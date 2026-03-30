#!/usr/bin/env python3
"""
Optimized script to shuffle datasets stored as multiple parquet files.

This version includes performance optimizations and better memory management.
"""

import argparse
import os
import random
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional, Iterator, Tuple
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import heapq


def get_parquet_files(directory: Path) -> List[Path]:
    """Get all parquet files in the directory."""
    parquet_files = list(directory.glob("*.parquet"))
    if not parquet_files:
        raise ValueError(f"No parquet files found in {directory}")
    return sorted(parquet_files)


def estimate_memory_usage(parquet_files: List[Path]) -> int:
    """Estimate total memory usage of all parquet files."""
    total_size = 0
    for file_path in parquet_files:
        total_size += file_path.stat().st_size
    # Rough estimate: parquet compression is ~3-5x, so multiply by 4
    return total_size * 4


def chunked_file_reader(file_path: Path, chunk_size: int = 10000) -> Iterator[pd.DataFrame]:
    """Read parquet file in chunks."""
    parquet_file = pq.ParquetFile(file_path)
    for batch in parquet_file.iter_batches(batch_size=chunk_size):
        yield batch.to_pandas()


def efficient_global_shuffle(input_dir: Path, output_dir: Path, 
                           max_memory_gb: float = 8.0, seed: Optional[int] = None) -> None:
    """
    Memory-efficient global shuffle using external sorting approach.
    """
    print("Starting efficient global shuffle...")
    
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    parquet_files = get_parquet_files(input_dir)
    print(f"Found {len(parquet_files)} parquet files")
    
    # Estimate memory usage
    estimated_size_gb = estimate_memory_usage(parquet_files) / (1024**3)
    print(f"Estimated dataset size: {estimated_size_gb:.2f} GB")
    
    max_memory_bytes = int(max_memory_gb * 1024**3)
    
    if estimated_size_gb <= max_memory_gb * 0.8:  # Use 80% of available memory
        # Can fit in memory, use simple approach
        print("Dataset fits in memory, using in-memory shuffle...")
        global_shuffle_in_memory(input_dir, output_dir, seed)
    else:
        # Use external sorting approach
        print("Dataset too large for memory, using external sort...")
        external_sort_shuffle(input_dir, output_dir, max_memory_bytes, seed)


def global_shuffle_in_memory(input_dir: Path, output_dir: Path, seed: Optional[int] = None) -> None:
    """In-memory global shuffle for smaller datasets."""
    parquet_files = get_parquet_files(input_dir)
    
    # Load all data
    all_data = []
    for file_path in tqdm(parquet_files, desc="Loading files"):
        df = pd.read_parquet(file_path)
        all_data.append(df)
    
    # Combine and shuffle
    combined_df = pd.concat(all_data, ignore_index=True)
    shuffled_df = combined_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    
    # Write back maintaining original file sizes
    output_dir.mkdir(parents=True, exist_ok=True)
    original_sizes = [len(pd.read_parquet(f)) for f in parquet_files]
    
    start_idx = 0
    for i, (original_file, target_size) in enumerate(zip(parquet_files, original_sizes)):
        end_idx = min(start_idx + target_size, len(shuffled_df))
        chunk = shuffled_df.iloc[start_idx:end_idx]
        
        output_file = output_dir / original_file.name
        chunk.to_parquet(output_file, index=False)
        start_idx = end_idx


def external_sort_shuffle(input_dir: Path, output_dir: Path, 
                         max_memory_bytes: int, seed: Optional[int] = None) -> None:
    """
    External sorting approach for large datasets.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Phase 1: Create sorted chunks
        print("Phase 1: Creating shuffled chunks...")
        chunk_files = create_shuffled_chunks(input_dir, temp_path, max_memory_bytes, seed)
        
        # Phase 2: Merge chunks back to output
        print("Phase 2: Merging chunks...")
        merge_shuffled_chunks(chunk_files, output_dir, input_dir)


def create_shuffled_chunks(input_dir: Path, temp_dir: Path, 
                          max_memory_bytes: int, seed: Optional[int] = None) -> List[Path]:
    """Create shuffled temporary chunks."""
    parquet_files = get_parquet_files(input_dir)
    chunk_files = []
    current_chunk = []
    current_size = 0
    chunk_num = 0
    
    # Estimate rows per chunk based on memory limit
    avg_row_size = 1000  # bytes per row estimate
    max_rows_per_chunk = max_memory_bytes // avg_row_size
    
    for file_path in tqdm(parquet_files, desc="Processing into chunks"):
        for chunk_df in chunked_file_reader(file_path, chunk_size=10000):
            current_chunk.append(chunk_df)
            current_size += len(chunk_df)
            
            if current_size >= max_rows_per_chunk:
                # Write shuffled chunk
                combined_chunk = pd.concat(current_chunk, ignore_index=True)
                shuffled_chunk = combined_chunk.sample(frac=1.0, random_state=seed + chunk_num if seed else None)
                
                chunk_file = temp_dir / f"chunk_{chunk_num:04d}.parquet"
                shuffled_chunk.to_parquet(chunk_file, index=False)
                chunk_files.append(chunk_file)
                
                # Reset for next chunk
                current_chunk = []
                current_size = 0
                chunk_num += 1
    
    # Handle remaining data
    if current_chunk:
        combined_chunk = pd.concat(current_chunk, ignore_index=True)
        shuffled_chunk = combined_chunk.sample(frac=1.0, random_state=seed + chunk_num if seed else None)
        
        chunk_file = temp_dir / f"chunk_{chunk_num:04d}.parquet"
        shuffled_chunk.to_parquet(chunk_file, index=False)
        chunk_files.append(chunk_file)
    
    return chunk_files


def merge_shuffled_chunks(chunk_files: List[Path], output_dir: Path, original_dir: Path) -> None:
    """Merge shuffled chunks back to output files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get original file sizes for output distribution
    original_files = get_parquet_files(original_dir)
    original_sizes = [len(pd.read_parquet(f)) for f in original_files]
    
    # Read all chunks and merge
    all_chunks = []
    for chunk_file in tqdm(chunk_files, desc="Loading chunks"):
        chunk_df = pd.read_parquet(chunk_file)
        all_chunks.append(chunk_df)
    
    merged_df = pd.concat(all_chunks, ignore_index=True)
    
    # Distribute to output files
    start_idx = 0
    for i, (original_file, target_size) in enumerate(zip(original_files, original_sizes)):
        end_idx = min(start_idx + target_size, len(merged_df))
        chunk = merged_df.iloc[start_idx:end_idx]
        
        output_file = output_dir / original_file.name
        chunk.to_parquet(output_file, index=False)
        start_idx = end_idx


def streaming_shuffle(input_dir: Path, output_dir: Path, 
                     buffer_size: int = 100000, seed: Optional[int] = None) -> None:
    """
    Streaming shuffle using reservoir sampling for very large datasets.
    This maintains a buffer and continuously samples from incoming data.
    """
    print("Starting streaming shuffle...")
    
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    parquet_files = get_parquet_files(input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    reservoir = []
    total_seen = 0
    
    # First pass: build reservoir
    print("Building reservoir sample...")
    for file_path in tqdm(parquet_files, desc="Sampling files"):
        for chunk_df in chunked_file_reader(file_path):
            for _, row in chunk_df.iterrows():
                total_seen += 1
                
                if len(reservoir) < buffer_size:
                    reservoir.append(row.to_dict())
                else:
                    # Reservoir sampling
                    j = random.randint(0, total_seen - 1)
                    if j < buffer_size:
                        reservoir[j] = row.to_dict()
    
    # Convert reservoir to DataFrame and shuffle
    print(f"Shuffling reservoir of {len(reservoir)} samples...")
    reservoir_df = pd.DataFrame(reservoir)
    shuffled_reservoir = reservoir_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    
    # Write to single output file
    output_file = output_dir / "shuffled_sample.parquet"
    shuffled_reservoir.to_parquet(output_file, index=False)
    print(f"Written shuffled sample: {len(shuffled_reservoir)} rows")


def parallel_file_shuffle(input_dir: Path, output_dir: Path, 
                         max_workers: int = 4, seed: Optional[int] = None) -> None:
    """
    Parallel processing of individual files for faster shuffling.
    """
    print("Starting parallel file shuffle...")
    
    parquet_files = get_parquet_files(input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    def shuffle_single_file(args: Tuple[Path, Path, Optional[int]]) -> str:
        file_path, output_path, file_seed = args
        df = pd.read_parquet(file_path)
        shuffled_df = df.sample(frac=1.0, random_state=file_seed).reset_index(drop=True)
        shuffled_df.to_parquet(output_path, index=False)
        return f"Shuffled {file_path.name}: {len(shuffled_df)} rows"
    
    # Prepare arguments for parallel processing
    tasks = []
    for i, file_path in enumerate(parquet_files):
        output_path = output_dir / file_path.name
        file_seed = seed + i if seed is not None else None
        tasks.append((file_path, output_path, file_seed))
    
    # Execute in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(
            executor.map(shuffle_single_file, tasks),
            total=len(tasks),
            desc="Processing files"
        ))
    
    for result in results:
        print(result)


def main():
    parser = argparse.ArgumentParser(description="Optimized parquet dataset shuffling")
    parser.add_argument("--input_dir", type=str, required=True,
                       help="Directory containing parquet files to shuffle")
    parser.add_argument("--output_dir", type=str,
                       help="Output directory for shuffled files")
    parser.add_argument("--strategy", type=str, default="auto",
                       choices=["auto", "global", "file_level", "streaming", "parallel"],
                       help="Shuffling strategy to use")
    parser.add_argument("--max_memory_gb", type=float, default=8.0,
                       help="Maximum memory to use in GB")
    parser.add_argument("--buffer_size", type=int, default=100000,
                       help="Buffer size for streaming shuffle")
    parser.add_argument("--max_workers", type=int, default=4,
                       help="Number of parallel workers")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducible shuffling")
    parser.add_argument("--in_place", action="store_true",
                       help="Shuffle files in place (creates backup)")
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise ValueError(f"Input directory {input_dir} does not exist")
    
    if args.in_place:
        backup_dir = input_dir.parent / f"{input_dir.name}_backup"
        print(f"Creating backup at {backup_dir}")
        shutil.copytree(input_dir, backup_dir)
        output_dir = input_dir
    else:
        output_dir = Path(args.output_dir) if args.output_dir else Path(f"{input_dir}_shuffled")
    
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Strategy: {args.strategy}")
    print(f"Random seed: {args.seed}")
    
    # Execute shuffling strategy
    if args.strategy == "auto" or args.strategy == "global":
        efficient_global_shuffle(input_dir, output_dir, args.max_memory_gb, args.seed)
    elif args.strategy == "file_level" or args.strategy == "parallel":
        parallel_file_shuffle(input_dir, output_dir, args.max_workers, args.seed)
    elif args.strategy == "streaming":
        streaming_shuffle(input_dir, output_dir, args.buffer_size, args.seed)
    
    print("Shuffling completed successfully!")


if __name__ == "__main__":
    main() 