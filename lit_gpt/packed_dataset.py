# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# Copyright Lightning AI. Licensed under the Apache License 2.0,
# see LICENSE file at https://github.com/Lightning-AI/litgpt/blob/main/LICENSE

# Very loosely inspired by indexed_dataset in Fairseq, Megatron
# https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/data/indexed_dataset.py


import os
import random
import struct
import json
import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info
import blobfile as bf

from utils import _parallel_ensure_local_paths


dtypes = {1: np.uint8, 2: np.int8, 3: np.int16, 4: np.int32, 5: np.int64, 6: np.float32, 7: np.float64, 8: np.uint16}


def code(dtype):
    for k in dtypes:
        if dtypes[k] == dtype:
            return k
    raise ValueError(dtype)


HDR_MAGIC = b"LITPKDS"
HDR_SIZE = 24  # bytes


class PackedDataset(IterableDataset):
    def __init__(
        self, filenames, n_chunks, block_size, seed=12345, shuffle=True, drop_last=True, num_processes=1, process_rank=0
    ):
        self._filenames = filenames
        self._n_chunks = n_chunks
        self._block_size = block_size
        self._seed = seed
        self._shuffle = shuffle
        self._drop_last = drop_last
        self._num_processes = num_processes
        self._process_rank = process_rank

    def __iter__(self):
        worker_info = get_worker_info()
        num_workers = worker_info.num_workers if worker_info is not None else 1
        worker_id = worker_info.id if worker_info is not None else 0
        num_shards = num_workers * self._num_processes
        shard_id = self._process_rank * num_workers + worker_id

        max_num_files = len(self._filenames) // num_shards * num_shards
        filenames = self._filenames[shard_id:max_num_files:num_shards]

        return PackedDatasetIterator(
            filenames=filenames,
            n_chunks=self._n_chunks,
            block_size=self._block_size,
            seed=self._seed,
            shuffle=self._shuffle,
            drop_last=self._drop_last,
        )


class PackedDatasetBuilder(object):
    def __init__(self, outdir, prefix, chunk_size, sep_token, dtype="auto", vocab_size=None):
        if dtype == "auto":
            if vocab_size is None:
                raise ValueError("vocab_size cannot be None when dtype='auto'")
            if vocab_size is not None and vocab_size < 65500:
                self._dtype = np.uint16
            else:
                self._dtype = np.int32
        else:
            self._dtype = dtype
        self._counter = 0
        self._chunk_size = chunk_size
        self._outdir = outdir
        self._prefix = prefix
        self._sep_token = sep_token
        self._arr = np.zeros(self._chunk_size, dtype=self._dtype)
        self._arr.fill(self._sep_token)
        self._idx = 0
        self._version = 1
        self._filenames = []
        self._total_samples = 0
        self._tokens = 0
    
    def _load_ckpt(self):
        ckptname = os.path.join(self._outdir, f"{self._prefix}.ckpt")
        if bf.exists(ckptname):
            with bf.BlobFile(ckptname, "r") as f:
                data = json.load(f)
                self._counter = data["counter"] + 1
                self._tokens = data["tokens"]
                self._total_samples = data["samples"]
            print(
                f"Loaded checkpoint from {ckptname}, counter={self._counter}, "
                f"tokens={self._tokens}, samples={self._total_samples}"
            )
            return True
        else:
            print(f"Checkpoint {ckptname} not found, starting fresh.")
            return False

    def _write_chunk(self):
        filename = os.path.join(self._outdir, f"{self._prefix}_{self._counter:010d}.bin")
        ckptname = os.path.join(self._outdir, f"{self._prefix}.ckpt")

        with bf.BlobFile(filename, "wb") as f:
            f.write(HDR_MAGIC)
            f.write(struct.pack("<Q", self._version))
            f.write(struct.pack("<B", code(self._dtype)))
            f.write(struct.pack("<Q", self._chunk_size))
            f.write(self._arr.tobytes(order="C"))

        with bf.BlobFile(ckptname, "w") as f:
            json.dump({"counter": self._counter, "tokens": self._tokens, "samples": self._total_samples}, f)

        self._filenames.append(filename)
        self._counter += 1
        self._arr.fill(self._sep_token)
        self._idx = 0

    @property
    def dtype(self):
        return self._dtype

    @property
    def filenames(self):
        return self._filenames.copy()

    def add_array(self, arr):
        while self._idx + arr.shape[0] > self._chunk_size:
            part_len = self._chunk_size - self._idx
            self._arr[self._idx : self._idx + part_len] = arr[:part_len]
            self._tokens += part_len
            self._write_chunk()
            arr = arr[part_len:]
            self._tokens += arr.shape[0]
 
        arr_len = arr.shape[0]
        self._arr[self._idx : self._idx + arr_len] = arr
        self._idx += arr_len
        self._total_samples += 1
        self._tokens += arr_len

    def write_reminder(self):
        self._write_chunk()


class PackedDatasetIterator:
    def __init__(self, filenames, n_chunks, block_size, seed, shuffle, drop_last):
        self._seed = seed
        self._shuffle = shuffle
        self._rng = np.random.default_rng(seed) if shuffle else None
        self._block_idxs = None

        self._drop_last = drop_last

        # TODO: instead of filenames, we could have a single text stream
        #       (or text file) with the sequence of all files to be
        #       fetched/loaded.
        self._filenames = filenames
        self._file_idx = 0

        self._n_chunks = n_chunks

        self._dtype = None
        self._block_size = block_size
        self._n_blocks = None
        
        self._mmaps = []
        self._buffers = []

        self._block_idxs = []
        self._curr_idx = 0

        self._load_n_chunks()

    def _read_header(self, path):
        with open(path, "rb") as f:
            magic = f.read(len(HDR_MAGIC))
            assert magic == HDR_MAGIC, "File doesn't match expected format."
            version = struct.unpack("<Q", f.read(8))
            assert version == (1,)
            (dtype_code,) = struct.unpack("<B", f.read(1))
            dtype = dtypes[dtype_code]
            (chunk_size,) = struct.unpack("<Q", f.read(8))
        return dtype, chunk_size

    def _close_mmaps(self):
        for mmap in self._mmaps:
            mmap._mmap.close()

    def _load_n_chunks(self): 
        self._close_mmaps()
        self._mmaps = [] 
        self._buffers = []

        if len(self._filenames) == 0:
            raise StopIteration

        if self._drop_last:
            if self._n_chunks > len(self._filenames[self._file_idx :]):
                self._file_idx = 0
        
            if self._n_chunks > len(self._filenames):
                n_chunks = len(self._filenames)
            else:
                n_chunks = self._n_chunks
        else:
            if self._file_idx == len(self._filenames):
                self._file_idx = 0
                
            if self._n_chunks > len(self._filenames):
                n_chunks = len(self._filenames)    
            elif self._n_chunks > len(self._filenames[self._file_idx :]) > 0:
                n_chunks = len(self._filenames[self._file_idx :])
            else:
                n_chunks = self._n_chunks
        # Fetch/copy all files for this batch in parallel
        batch_filenames = [self._filenames[self._file_idx + i] for i in range(n_chunks)]
        local_paths = _parallel_ensure_local_paths(batch_filenames, max_workers=n_chunks)

        print(f"Loaded {len(local_paths)} local paths")
        # Initialize header info from the first file if needed
        if self._dtype is None and local_paths:
            self._dtype, self._chunk_size = self._read_header(local_paths[0])
            self._n_blocks = self._chunk_size // self._block_size

        for local_path in local_paths:
            mmap = np.memmap(local_path, mode="r", order="C", offset=HDR_SIZE)
            self._mmaps.append(mmap)
            self._buffers.append(memoryview(mmap))

        self._file_idx += n_chunks
        n_all_blocks = n_chunks * self._n_blocks

        self._block_idxs = self._rng.permutation(n_all_blocks) if self._shuffle else range(n_all_blocks)

        self._curr_idx = 0

    def __del__(self):
        self._close_mmaps()
        del self._mmaps
        del self._buffers

    def __iter__(self):
        return self

    def __next__(self):
        if self._curr_idx >= len(self._block_idxs):
            self._load_n_chunks()
            # TODO: trigger fetching next next n_chunks if remote
        block_idx = self._block_idxs[self._curr_idx]
        chunk_id = block_idx // self._n_blocks
        buffer = self._buffers[chunk_id]
        elem_id = (block_idx % self._n_blocks) * self._block_size
        offset = np.dtype(self._dtype).itemsize * elem_id
        arr = np.frombuffer(buffer, dtype=self._dtype, count=self._block_size, offset=offset)
        self._curr_idx += 1
        return torch.from_numpy(arr.astype(np.int64))


class CombinedDataset(IterableDataset):
    def __init__(self, datasets, seed, weights=None):
        self._seed = seed
        self._datasets = datasets
        self._weights = weights
        n_datasets = len(datasets)
        if weights is None:
            self._weights = [1 / n_datasets] * n_datasets

    def __iter__(self):
        return CombinedDatasetIterator(self._datasets, self._seed, self._weights)


class CombinedDatasetIterator:
    def __init__(self, datasets, seed, weights):
        # Build paired list of (iterator, is_code_coda_flag) based on dataset source name
        self._datasets = []
        for ds in datasets:
            it = iter(ds)
            is_code = "msr_code" in ds._filenames[0]
            self._datasets.append((it, is_code))
        self._weights = weights
        self._rng = random.Random(seed)

    def __next__(self):
        # pick which dataset to draw from
        idx = self._rng.choices(range(len(self._datasets)), weights=self._weights, k=1)[0]
        dataset_it, is_code = self._datasets[idx]
        sample = next(dataset_it)
        return sample, is_code