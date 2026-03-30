# Parquet Dataset Shuffling Scripts

This repository contains Python scripts to shuffle datasets stored as multiple parquet files. Two versions are provided:

1. **`shuffle_parquet_dataset.py`**: Basic version with multiple shuffling strategies
2. **`shuffle_parquet_optimized.py`**: Optimized version with better memory management and performance

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Basic Usage

### Simple File-by-File Shuffle (Most Memory Efficient)

```bash
python shuffle_parquet_dataset.py --input_dir /path/to/dataset --strategy simple
```

This shuffles each parquet file individually. Most memory efficient but provides the least thorough shuffling.

### Global Shuffle (Best Shuffling Quality)

```bash
python shuffle_parquet_dataset.py --input_dir /path/to/dataset --strategy global --output_dir /path/to/output
```

Loads all data into memory, shuffles globally, and writes back. Provides the best shuffling but requires sufficient RAM.

### File-Level Shuffle (Balanced Approach)

```bash
python shuffle_parquet_dataset.py --input_dir /path/to/dataset --strategy file_level
```

Shuffles the order of files and shuffles rows within each file. Good balance between memory efficiency and shuffle quality.

## Optimized Version Usage

The optimized version automatically chooses the best strategy based on dataset size and available memory:

### Auto Strategy (Recommended)

```bash
python shuffle_parquet_optimized.py --input_dir /path/to/dataset --max_memory_gb 16
```

Automatically decides between in-memory and external sorting based on dataset size.

### Parallel Processing

```bash
python shuffle_parquet_optimized.py --input_dir /path/to/dataset --strategy parallel --max_workers 8
```

Uses multiple threads to process files in parallel for faster execution.

### Streaming Shuffle (For Very Large Datasets)

```bash
python shuffle_parquet_optimized.py --input_dir /path/to/dataset --strategy streaming --buffer_size 50000
```

Uses reservoir sampling for extremely large datasets that don't fit in memory.

## Command Line Options

### Basic Script Options

- `--input_dir`: Directory containing parquet files (required)
- `--output_dir`: Output directory (default: input_dir + '_shuffled')
- `--strategy`: Shuffling strategy (`simple`, `global`, `file_level`, `memory_efficient`)
- `--seed`: Random seed for reproducible results (default: 42)
- `--in_place`: Shuffle files in place (creates backup first)
- `--batch_size`: Batch size for memory-efficient strategy

### Optimized Script Options

- `--input_dir`: Directory containing parquet files (required)
- `--output_dir`: Output directory (default: input_dir + '_shuffled')
- `--strategy`: Strategy (`auto`, `global`, `parallel`, `streaming`)
- `--max_memory_gb`: Maximum memory to use in GB (default: 8.0)
- `--max_workers`: Number of parallel workers (default: 4)
- `--buffer_size`: Buffer size for streaming shuffle (default: 100000)
- `--seed`: Random seed for reproducible results (default: 42)
- `--in_place`: Shuffle files in place (creates backup first)

## Shuffling Strategies Explained

### 1. Simple Shuffle
- **Memory**: Very low
- **Speed**: Fast
- **Quality**: Basic (within-file shuffling only)
- **Use case**: Large datasets with limited memory

### 2. Global Shuffle
- **Memory**: High (loads all data)
- **Speed**: Medium
- **Quality**: Excellent (true global shuffle)
- **Use case**: Smaller datasets that fit in memory

### 3. File-Level Shuffle
- **Memory**: Low
- **Speed**: Medium
- **Quality**: Good (file order + within-file shuffling)
- **Use case**: General purpose, balanced approach

### 4. Memory-Efficient Shuffle
- **Memory**: Configurable
- **Speed**: Slow
- **Quality**: Excellent
- **Use case**: Large datasets with specific memory constraints

### 5. Streaming Shuffle (Optimized Version)
- **Memory**: Very low
- **Speed**: Slow
- **Quality**: Good (reservoir sampling)
- **Use case**: Extremely large datasets

### 6. Parallel Shuffle (Optimized Version)
- **Memory**: Low
- **Speed**: Very fast
- **Quality**: Good (within-file shuffling)
- **Use case**: Multi-core systems, time-sensitive applications

## Examples

### Example 1: Small Dataset (< 1GB)

```bash
# Use global shuffle for best quality
python shuffle_parquet_dataset.py \
    --input_dir ./small_dataset \
    --strategy global \
    --seed 123
```

### Example 2: Large Dataset (> 10GB)

```bash
# Use optimized version with auto strategy
python shuffle_parquet_optimized.py \
    --input_dir ./large_dataset \
    --strategy auto \
    --max_memory_gb 32 \
    --seed 123
```

### Example 3: Very Large Dataset (> 100GB)

```bash
# Use streaming shuffle
python shuffle_parquet_optimized.py \
    --input_dir ./huge_dataset \
    --strategy streaming \
    --buffer_size 200000 \
    --seed 123
```

### Example 4: Fast Processing with Multiple Cores

```bash
# Use parallel processing
python shuffle_parquet_optimized.py \
    --input_dir ./dataset \
    --strategy parallel \
    --max_workers 16 \
    --seed 123
```

### Example 5: In-Place Shuffling

```bash
# Shuffle files in place (creates backup)
python shuffle_parquet_optimized.py \
    --input_dir ./dataset \
    --in_place \
    --seed 123
```

## Performance Tips

1. **Memory Sizing**: For the auto strategy, set `--max_memory_gb` to about 70% of your available RAM
2. **Parallel Workers**: Set `--max_workers` to the number of CPU cores for file-level operations
3. **Batch Size**: For streaming shuffle, larger buffer sizes give better shuffling but use more memory
4. **SSD Storage**: Use SSD storage for temporary files when using external sorting strategies

## Dataset Structure

The scripts expect a directory containing `.parquet` files:

```
dataset/
├── file_001.parquet
├── file_002.parquet
├── file_003.parquet
└── ...
```

Output structure maintains the same format:

```
dataset_shuffled/
├── file_001.parquet  (shuffled content)
├── file_002.parquet  (shuffled content)
├── file_003.parquet  (shuffled content)
└── ...
```

## Error Handling

The scripts include error handling for common issues:

- Missing input directory
- No parquet files found
- Insufficient memory for global shuffle
- Disk space issues for external sorting

## Reproducibility

Use the `--seed` parameter to ensure reproducible shuffling:

```bash
python shuffle_parquet_optimized.py --input_dir ./dataset --seed 12345
```

Running the same command with the same seed will produce identical results.

## Memory Requirements

| Strategy | Memory Usage | Recommended For |
|----------|--------------|------------------|
| Simple | ~1 file size | Any dataset |
| Global | ~Total dataset size | < 50% of RAM |
| File-level | ~1 file size | General use |
| Streaming | ~Buffer size | Very large datasets |
| Parallel | ~1 file size × workers | Multi-core systems |

## Troubleshooting

### Out of Memory Errors
- Use `streaming` or `parallel` strategy
- Reduce `--max_memory_gb` parameter
- Reduce `--buffer_size` for streaming

### Slow Performance
- Use `parallel` strategy for multiple files
- Increase `--max_workers` for parallel processing
- Use SSD storage for temporary files

### Disk Space Issues
- Ensure 2x dataset size available disk space
- Use `--in_place` option to avoid duplicating data
- Clean up temporary directories manually if interrupted 