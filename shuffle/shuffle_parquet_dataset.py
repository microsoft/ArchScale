#!/usr/bin/env python3
"""
Script to shuffle datasets stored as multiple parquet files.

This script provides several strategies for shuffling:
1. Global shuffle: Load all data, shuffle globally, and write back
2. File-level shuffle: Shuffle the order of files and shuffle within each file
3. Memory-efficient shuffle: Process files in batches to handle large datasets

Usage:
    python shuffle_parquet_dataset.py --input_dir /path/to/dataset --strategy global
    python shuffle_parquet_dataset.py --input_dir /path/to/dataset --strategy file_level --output_dir /path/to/shuffled
    python shuffle_parquet_dataset.py --input_dir /path/to/dataset --strategy memory_efficient --batch_size 10000
"""

import argparse
import os
import random
import shutil
from pathlib import Path
from typing import List, Optional
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
import numpy as np


def get_parquet_files(directory: Path) -> List[Path]:
    """Get all parquet files in the directory."""
    parquet_files = list(directory.glob("*.parquet"))
    if not parquet_files:
        raise ValueError(f"No parquet files found in {directory}")
    return sorted(parquet_files)


def global_shuffle(input_dir: Path, output_dir: Path, seed: Optional[int] = None) -> None:
    """
    Load all parquet files, shuffle globally, and write back.
    Warning: This loads all data into memory - use for smaller datasets.
    """
    print("Starting global shuffle...")
    
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    parquet_files = get_parquet_files(input_dir)
    print(f"Found {len(parquet_files)} parquet files")
    
    # Load all data
    print("Loading all data into memory...")
    all_data = []
    total_rows = 0
    
    for file_path in tqdm(parquet_files, desc="Loading files"):
        df = pd.read_parquet(file_path)
        all_data.append(df)
        total_rows += len(df)
        print(f"Loaded {file_path.name}: {len(df)} rows")
    
    # Concatenate and shuffle
    print(f"Concatenating {total_rows} total rows...")
    combined_df = pd.concat(all_data, ignore_index=True)
    del all_data  # Free memory
    
    print("Shuffling data...")
    shuffled_df = combined_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    del combined_df  # Free memory
    
    # Write back to files with similar sizes
    print("Writing shuffled data back to files...")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Calculate rows per file based on original distribution
    original_sizes = [pd.read_parquet(f).shape[0] for f in parquet_files]
    
    start_idx = 0
    for i, (original_file, target_size) in enumerate(zip(parquet_files, original_sizes)):
        end_idx = min(start_idx + target_size, len(shuffled_df))
        chunk = shuffled_df.iloc[start_idx:end_idx]
        
        output_file = output_dir / f"shuffled_{original_file.name}"
        chunk.to_parquet(output_file, index=False)
        print(f"Written {output_file.name}: {len(chunk)} rows")
        
        start_idx = end_idx
        if start_idx >= len(shuffled_df):
            break


def file_level_shuffle(input_dir: Path, output_dir: Path, seed: Optional[int] = None) -> None:
    """
    Shuffle the order of files and shuffle rows within each file.
    More memory efficient than global shuffle.
    """
    print("Starting file-level shuffle...")
    
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    parquet_files = get_parquet_files(input_dir)
    print(f"Found {len(parquet_files)} parquet files")
    
    # Shuffle file order
    shuffled_files = parquet_files.copy()
    random.shuffle(shuffled_files)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i, file_path in enumerate(tqdm(shuffled_files, desc="Processing files")):
        # Load and shuffle each file
        df = pd.read_parquet(file_path)
        shuffled_df = df.sample(frac=1.0, random_state=seed + i if seed else None).reset_index(drop=True)
        
        # Write with new name to maintain shuffled order
        output_file = output_dir / f"shuffled_{i:04d}_{file_path.name}"
        shuffled_df.to_parquet(output_file, index=False)
        print(f"Processed {file_path.name} -> {output_file.name}: {len(shuffled_df)} rows")


def memory_efficient_shuffle(input_dir: Path, output_dir: Path, batch_size: int = 10000, 
                           seed: Optional[int] = None) -> None:
    """
    Memory-efficient shuffle using reservoir sampling and batch processing.
    Suitable for very large datasets that don't fit in memory.
    """
    print("Starting memory-efficient shuffle...")
    
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    parquet_files = get_parquet_files(input_dir)
    print(f"Found {len(parquet_files)} parquet files")
    
    # First pass: collect sample indices for shuffling
    print("First pass: analyzing data distribution...")
    file_info = []
    total_rows = 0
    
    for file_path in tqdm(parquet_files, desc="Analyzing files"):
        parquet_file = pq.ParquetFile(file_path)
        num_rows = parquet_file.metadata.num_rows
        file_info.append({
            'path': file_path,
            'rows': num_rows,
            'start_idx': total_rows
        })
        total_rows += num_rows
    
    print(f"Total rows across all files: {total_rows}")
    
    # Generate shuffled indices
    print("Generating shuffled indices...")
    shuffled_indices = np.random.permutation(total_rows)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process in batches
    print(f"Processing data in batches of {batch_size}...")
    batch_num = 0
    batch_data = []
    
    for i in tqdm(range(0, total_rows, batch_size), desc="Processing batches"):
        batch_indices = shuffled_indices[i:i + batch_size]
        batch_rows = []
        
        # Collect rows for this batch
        for idx in batch_indices:
            # Find which file this index belongs to
            file_idx = 0
            while file_idx < len(file_info) - 1 and idx >= file_info[file_idx + 1]['start_idx']:
                file_idx += 1
            
            local_idx = idx - file_info[file_idx]['start_idx']
            
            # Read specific row from file (this is simplified - in practice you'd want to batch these reads)
            df = pd.read_parquet(file_info[file_idx]['path'])
            row = df.iloc[local_idx:local_idx + 1]
            batch_rows.append(row)
        
        # Combine batch and write
        if batch_rows:
            batch_df = pd.concat(batch_rows, ignore_index=True)
            output_file = output_dir / f"shuffled_batch_{batch_num:04d}.parquet"
            batch_df.to_parquet(output_file, index=False)
            print(f"Written batch {batch_num}: {len(batch_df)} rows")
            batch_num += 1


def simple_file_shuffle(input_dir: Path, output_dir: Path, seed: Optional[int] = None) -> None:
    """
    Simple approach: shuffle each file individually in place or to output directory.
    Most memory efficient but least thorough shuffling.
    """
    print("Starting simple file shuffle...")
    
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    parquet_files = get_parquet_files(input_dir)
    print(f"Found {len(parquet_files)} parquet files")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for file_path in tqdm(parquet_files, desc="Shuffling files"):
        df = pd.read_parquet(file_path)
        shuffled_df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        
        output_file = output_dir / file_path.name
        shuffled_df.to_parquet(output_file, index=False)
        print(f"Shuffled {file_path.name}: {len(shuffled_df)} rows")


def main():
    parser = argparse.ArgumentParser(description="Shuffle parquet dataset files")
    parser.add_argument("--input_dir", type=str, required=True,
                       help="Directory containing parquet files to shuffle")
    parser.add_argument("--output_dir", type=str, 
                       help="Output directory for shuffled files (default: input_dir + '_shuffled')")
    parser.add_argument("--strategy", type=str, default="simple",
                       choices=["global", "file_level", "memory_efficient", "simple"],
                       help="Shuffling strategy to use")
    parser.add_argument("--batch_size", type=int, default=10000,
                       help="Batch size for memory-efficient strategy")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducible shuffling")
    parser.add_argument("--in_place", action="store_true",
                       help="Shuffle files in place (overwrites original files)")
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise ValueError(f"Input directory {input_dir} does not exist")
    
    if args.in_place:
        # Create backup first
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
    if args.strategy == "global":
        global_shuffle(input_dir, output_dir, args.seed)
    elif args.strategy == "file_level":
        file_level_shuffle(input_dir, output_dir, args.seed)
    elif args.strategy == "memory_efficient":
        memory_efficient_shuffle(input_dir, output_dir, args.batch_size, args.seed)
    elif args.strategy == "simple":
        simple_file_shuffle(input_dir, output_dir, args.seed)
    
    print("Shuffling completed successfully!")


if __name__ == "__main__":
    main() 