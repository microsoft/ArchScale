#!/usr/bin/env python3
"""
Script to create a sample parquet dataset for testing shuffling functionality.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse


def create_sample_dataset(output_dir: Path, num_files: int = 5, rows_per_file: int = 1000):
    """Create a sample dataset with multiple parquet files."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Creating sample dataset in {output_dir}")
    print(f"Files: {num_files}, Rows per file: {rows_per_file}")
    
    for i in range(num_files):
        # Generate sample data
        data = {
            'id': range(i * rows_per_file, (i + 1) * rows_per_file),
            'name': [f'person_{j}' for j in range(i * rows_per_file, (i + 1) * rows_per_file)],
            'age': np.random.randint(18, 80, rows_per_file),
            'salary': np.random.randint(30000, 150000, rows_per_file),
            'department': np.random.choice(['Engineering', 'Sales', 'Marketing', 'HR'], rows_per_file),
            'score': np.random.random(rows_per_file),
            'file_source': i  # To track which file each row came from originally
        }
        
        df = pd.DataFrame(data)
        
        # Save to parquet
        output_file = output_dir / f"data_{i:03d}.parquet"
        df.to_parquet(output_file, index=False)
        
        print(f"Created {output_file}: {len(df)} rows")
    
    print(f"Sample dataset created successfully!")
    print(f"Total rows: {num_files * rows_per_file}")


def main():
    parser = argparse.ArgumentParser(description="Create sample parquet dataset")
    parser.add_argument("--output_dir", type=str, default="sample_dataset",
                       help="Output directory for sample dataset")
    parser.add_argument("--num_files", type=int, default=5,
                       help="Number of parquet files to create")
    parser.add_argument("--rows_per_file", type=int, default=1000,
                       help="Number of rows per file")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    create_sample_dataset(output_dir, args.num_files, args.rows_per_file)


if __name__ == "__main__":
    main() 