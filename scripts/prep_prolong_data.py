# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# Copyright Lightning AI. Licensed under the Apache License 2.0,
# see LICENSE file at https://github.com/Lightning-AI/litgpt/blob/main/LICENSE

# prepare_prolong_64k_v2: Finished parallel implementation; may still require debugging
# python scripts/prepare_prolong_64k_v2.py --source_path ../../../../Datasets/prolong-data-64K/ --tokenizer_path Llama-2-7b-hf  --destination_path data/prolong_64K_v2 --split validation --percentage 1.0
# python scripts/prepare_prolong_64k_v2.py --source_path ../../../../Datasets/prolong-data-64K/ --tokenizer_path Llama-2-7b-hf  --destination_path data/prolong_64K_v2 --split train --percentage 1.0

# python prep_prolong_data.py --source_path prolong-data-64K/ --tokenizer_path Phi-3.5-mini-instruct --destination_path data/prolong_64K_v2 --split train --percentage 1.0

import json
import glob
import os
from pathlib import Path
import sys
from typing import List
import numpy as np
from tqdm import tqdm
from multiprocessing import Process, cpu_count
from streaming import StreamingDataset, Stream
import glob
from transformers import AutoTokenizer
from icecream import ic
import torch

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

import lit_gpt.packed_dataset as packed_dataset
from lit_gpt import Tokenizer

# Filename for SlimPajama
slimpajama_sets = {
    "train": "train/chunk*/*",
    "validation": "validation/chunk*/*",
    "test": "test/chunk*/*",
}


def prepare_full(
    source_path: Path,
    tokenizer_path: Path,
    destination_path: Path,
    chunk_size: int,
    split: str="train",
    filenames_subset: List[str] = None,
    process_id: int = 0,
    log_dir: str = None,
) -> None:
    import zstandard as zstd
    destination_path.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer(tokenizer_path)
    llama3_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")

    # Use the provided filenames_subset or default to all filenames
    filenames = filenames_subset
    
    if not filenames:
        raise RuntimeError(
            f"No files matching {slimpajama_sets[split]} found at {source_path}. \n"
            "Make sure you download the data..."
        )

    log_file = os.path.join(log_dir, f"processing_{process_id}.log")
    # completed_files = set()
    # if os.path.exists(log_file):
    #     with open(log_file, "r") as f:
    #         completed_files = set(json.load(f))

    # for split in ['validation', 'train']:
    prefix = f"{split}_prolong64K_{process_id}"
    print('prefix: ', prefix)
    builder = packed_dataset.PackedDatasetBuilder(
        outdir=destination_path,
        prefix=f"{split}_prolong64K_{process_id}",  # Use process_id to differentiate builders
        chunk_size=chunk_size,
        sep_token=tokenizer.bos_id,  # 128000, #
        dtype="auto",
        vocab_size=tokenizer.vocab_size,  # 128000 # 
    )
    # with zstd.open(open(filepath, "rb"), "rt", encoding="utf-8") as f:
    #     for row in tqdm(f):
    #         text = json.loads(row)["text"]
    #         #if json.loads(row)["meta"]["redpajama_set_name"] == "RedPajamaGithub":
    #         #    continue # we don't want to include the github data
    #         text_ids = tokenizer.encode(text)
    for i_sample, sample in enumerate(tqdm(filenames)):
        input_ids = sample['input_ids']
        # ic(input_ids.shape, input_ids[sample['indices'][0, 0]:sample['indices'][0, 0]+2], input_ids[sample['indices'][0, 1]-2:sample['indices'][0, 1]+2], tokenizer.bos_id, tokenizer.eos_id, llama3_tokenizer.bos_token_id, llama3_tokenizer.eos_token_id)
        for i_indice, each_indice in enumerate(sample['indices']):
            # ic('---------------- each doc start -------------------')
            # ic(each_indice, input_ids.shape)
            # ic(1, input_ids[0], input_ids[-1])
            document_input_ids = input_ids[each_indice[0]:each_indice[1]]
            # ic(document_input_ids.shape)
            # ic(document_input_ids[0], document_input_ids[-1])
            # ic(document_input_ids)
            start_with_128000 = document_input_ids[0] == 128000
            end_with_128001 = document_input_ids[-1] == 128001
            decoded_text = llama3_tokenizer.decode(document_input_ids)
            # print(decoded_text)
            text_ids = tokenizer.encode(decoded_text) # [:-1]
            # ic(1, text_ids)
            # ic(text_ids[:10], text_ids[-10:])
            if start_with_128000 and text_ids[0] != tokenizer.bos_id:
                # ic('start_with_128000 and text_ids[0] != tokenizer.bos_id')
                text_ids = torch.cat((torch.tensor([tokenizer.bos_id]), text_ids), dim=0)
            if not end_with_128001 and text_ids[-1] == tokenizer.eos_id:
                # ic('not end_with_128001 and text_ids[-1] == tokenizer.eos_id')
                text_ids = text_ids[:-1]
            # ic(2, text_ids, text_ids.shape)
            # ic(input_ids.shape, sample['indices'], text_ids.shape, text_ids[0:5], text_ids[-5:], tokenizer.bos_id, tokenizer.eos_id, llama3_tokenizer.bos_id, llama3_tokenizer.eos_id)
            builder.add_array(np.array(text_ids, dtype=builder.dtype))
        # if i_sample > 10:
        #     break 
        # Log
        # completed_files.add(filepath)
        # with open(log_file, "w") as f:
        #     json.dump(list(completed_files), f)

    # we throw away the final corpus to avoid meaningless corpus filled with bos_ids, see https://github.com/jzhang38/TinyLlama/issues/83 for more details
    # builder.write_reminder()


def prepare(
    source_path: Path = Path("data/RedPajama-Data-1T-Sample"),
    tokenizer_path: Path = Path("checkpoints/lit-llama/tokenizer.model"),
    destination_path: Path = Path("data/red_pajama_sample"),
    chunk_size: int = 2049 * 1024,
    split: str="train",
    percentage: float = 1.0,
    log_dir='prolong64K_log_dir',
) -> None:
    import time
    log_dir = os.path.join(log_dir, split)
    os.makedirs(log_dir, exist_ok=True)
    # filenames = glob.glob(os.path.join(source_path, slimpajama_sets[split]), recursive=True)
    # filenames = filenames[:int(len(filenames) * percentage)]
    #filenames = [f for f in os.listdir(source_path) if (os.path.isdir(os.path.join(source_path, f)) and f not in ['.git'])]
    filenames = ['book-65536', 'textbooks', 'thestackv1_concat_by_repo-65536']
    streams = []
    for filepath in filenames:
        # if filepath in completed_files:
        #     print(f"[Process {process_id}] Skipping {filepath}")
        #     continue
        
        print(f"Processing {filepath}")
        # streams = []
        path = os.path.join(source_path, filepath)
        print(f"Loading dataset from {path}")
        streams.append(Stream(remote=path, local=path))
        # dataset = StreamingDataset(streams=streams, shuffle=False, batch_size=1)
    train_val_dataset = StreamingDataset(streams=streams, shuffle=True, batch_size=1, shuffle_seed=9176)
    # if split == 'train':
    #     dataset = train_val_dataset[:int(len(train_val_dataset)*0.9)]
    # else:
    dataset = train_val_dataset #[int(len(train_val_dataset)*0.9):]
        
    print('len(dataset): {}, len(train_val_dataset): {}'.format(len(dataset), len(train_val_dataset)))
    num_processes = cpu_count() // 2 # min(cpu_count(), len(filenames))
    chunked_filenames = np.array_split(dataset, num_processes)
    # ic(chunked_filenames)
    for i, each_chunk in enumerate(chunked_filenames):
        print(i, 'len(each_chunk): {}'.format(len(each_chunk)))
    # paths = glob.glob(source_path+"/*/")
    # print(paths)

    processes = []
    start_time = time.time()

    for i, subset in enumerate(chunked_filenames):
        p = Process(target=prepare_full, args=(source_path, tokenizer_path, destination_path, chunk_size, split, list(subset), i, log_dir))
        processes.append(p)
        p.start()
        # time.sleep(10)  # Pauses for 10 seconds

    for p in processes:
        p.join()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    from jsonargparse import CLI
    CLI(prepare)