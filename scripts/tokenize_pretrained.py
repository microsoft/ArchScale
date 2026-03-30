# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# Copyright Lightning AI. Licensed under the Apache License 2.0,
# see LICENSE file at https://github.com/Lightning-AI/litgpt/blob/main/LICENSE

import json
import glob
import os
from pathlib import Path
import sys
from typing import List
import numpy as np
from tqdm import tqdm
from multiprocessing import Process, cpu_count
import random

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

import lit_gpt.packed_dataset as packed_dataset
from lit_gpt.tokenizer import Tokenizer
from transformers import AutoTokenizer

import pandas as pd

TEXT_KEYS = ["text", "file_contents", "content", "file_content"]  # priority order


def apply_chat_template(messages, eos_token) -> str:
    role_prefix = {
        "system": "<|system|>",
        "user": "<|user|>",
        "assistant": "<|assistant|>",
    }

    out_parts: List[str] = []
    for m in messages:
        role = m["role"]
        if role not in role_prefix:
            raise ValueError(f"Unknown role: {role!r}")
        prefix = role_prefix[role]
        content = m["content"]
        out_parts.append(f"{prefix}\n{content}<|end|>\n")

    return "".join(out_parts)


def extract_text(json_obj):
    for key in TEXT_KEYS:
        if key in json_obj:
            return json_obj[key]
    if "messages" in json_obj:
        text = apply_chat_template(json_obj["messages"], "<|endoftext|>")
        return text
    return None  # or raise an error if required


def read_jsonl_file(filepath):
    """Read JSONL file and yield text content from each line."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    json_obj = json.loads(line)
                    text = extract_text(json_obj)
                    if text:
                        yield text
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON in {filepath} at line {line_num + 1}: {e}")
                    continue
    except Exception as e:
        print(f"Error reading file {filepath}: {e}")


def read_parquet_file(filepath):
    """Read Parquet file and yield text content."""
    try:
        df = pd.read_parquet(filepath, engine='pyarrow')
        # Try different possible column names for text content
        text_column = None
        for col in TEXT_KEYS:
            if col in df.columns:
                text_column = col
                break
        
        if text_column is None:
            print(f"No text column found in {filepath}. Available columns: {df.columns.tolist()}")
            return
            
        for text in df[text_column]:
            if text and isinstance(text, str):
                yield text
    except Exception as e:
        print(f"Error reading parquet file {filepath}: {e}")


def prepare_full(
    source_path: Path,
    tokenizer_path: Path,
    destination_path: Path,
    chunk_size: int,
    filenames_subset: List[str] = None,
    process_id: int = 0
) -> None:

    destination_path.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer(tokenizer_path)
    # tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    # Use the provided filenames_subset or default to all filenames
    filenames = filenames_subset 

    builder = packed_dataset.PackedDatasetBuilder(
        outdir=destination_path,
        prefix=f"phi4data_{process_id}",  # Use process_id to differentiate builders
        chunk_size=chunk_size,
        sep_token=tokenizer.eos_id,
        dtype="auto",
        vocab_size=tokenizer.vocab_size,
    )

    load_ckpt = builder._load_ckpt()
    if load_ckpt:
        processed_samples = builder._total_samples - 1
    else:
        processed_samples = -1

    success_count = 0
    num_samples = 0
    for filepath in filenames:
        print(f"Processing {filepath}")
        
        # Determine file type and process accordingly
        file_ext = Path(filepath).suffix.lower()
        
        if file_ext == '.parquet':
            text_generator = read_parquet_file(filepath)
        elif file_ext == '.jsonl':
            text_generator = read_jsonl_file(filepath)
        else:
            print(f"Unsupported file type: {file_ext} for file {filepath}")
            continue
            
        try:
            for text in text_generator:
                if num_samples < processed_samples:
                    num_samples += 1
                    continue
                else:
                    # print("Start processing new samples")
                    num_samples = 0
                    processed_samples = -1
                
                if text and isinstance(text, str) and text.strip():
                    text_ids = tokenizer.encode(text)
                    builder.add_array(np.array(text_ids, dtype=builder.dtype))
            success_count += 1
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
            continue

    print(f"Processed ID {process_id} Processed {success_count} rows from {len(filenames)} files.")
    # we throw away the final corpus to avoid meaningless corpus filled with bos_ids, see https://github.com/jzhang38/TinyLlama/issues/83 for more details
    # builder.write_reminder()


def prepare(
    source_path: Path = Path("/home/v-zichongli/data-container/junhenghao/RAW_DATA/phi4_data_retokenization/synthetic_jsonl"),
    tokenizer_path: Path = Path("./phi3_tokenizer/"),
    destination_path: Path = None,
    chunk_size: int = 2049 * 8192,
    nproc: int = None,
    file_types: List[str] = None,
) -> None:
    import time
    if destination_path is None:
        destination_path = Path(str(source_path).replace("data_2B", "data_2B_tokenized"))

    # Default file types to process
    if file_types is None:
        file_types = ["*.parquet", "*.jsonl"]
    
    # Find files of specified types
    filenames = []
    for file_type in file_types:
        pattern = os.path.join(source_path, "**", file_type)
        found_files = glob.glob(pattern, recursive=True)
        filenames.extend(found_files)
    
    filenames = sorted(filenames)
    print(f"Found {len(filenames)} files in {source_path}")
    print(f"File types: {file_types}")
    print("Sample files:", filenames[:10])
    
    random.seed(43)                           # ② set the seed
    random.shuffle(filenames)
    
    if nproc is not None:
        num_processes = nproc
    else:
        num_processes = cpu_count()
    chunked_filenames = np.array_split(filenames, num_processes)
    # print(chunked_filenames)
    processes = []
    start_time = time.time()

    for i, subset in enumerate(chunked_filenames):
        if len(list(subset)) == 0:
            print(f"Skipping empty subset for process {i}")
            continue
        p = Process(target=prepare_full, args=(source_path, tokenizer_path, destination_path, chunk_size, list(subset), i))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    from jsonargparse import CLI
    CLI(prepare)