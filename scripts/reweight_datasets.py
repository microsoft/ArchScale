#!/usr/bin/env python3
"""
Script to create dataset_weights.json based on leaf folders from data_stats.txt
"""

import json
from typing import Dict, Set, List, Tuple


def parse_token_count(token_str: str) -> float:
    """
    Parse token count string like '1.79B' or '18.8B' into float number of tokens
    """
    token_str = token_str.strip()
    if token_str.endswith('B'):
        return float(token_str[:-1]) * 1e9
    elif token_str.endswith('M'):
        return float(token_str[:-1]) * 1e6
    elif token_str.endswith('K'):
        return float(token_str[:-1]) * 1e3
    else:
        return float(token_str)


def build_dataset_path(levels: list) -> str:
    """
    Build dataset path from levels, filtering out empty strings
    """
    non_empty_levels = [level.strip() for level in levels if level.strip()]
    return '/'.join(non_empty_levels)


def parse_data_stats_hierarchical(filename: str) -> List[Tuple[str, float, int]]:
    """
    Parse data_stats.txt and return list of (path, tokens, depth) tuples
    """
    entries = []
    
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Skip header line
    for line in lines[1:]:
        if not line.strip():
            continue
            
        parts = line.strip().split('\t')
        if len(parts) < 8:
            continue
            
        # Extract levels (Level_1 through Level_6)
        levels = parts[:6]
        tokens_str = parts[6]
        
        # Build the dataset path and calculate depth
        dataset_path = build_dataset_path(levels)
        depth = len([level for level in levels if level.strip()])
        
        if dataset_path:
            try:
                token_count = parse_token_count(tokens_str)
                entries.append((dataset_path, token_count, depth))
                print(f"Parsed: {dataset_path} (depth={depth}) -> {token_count:,.0f} tokens")
            except ValueError as e:
                print(f"Error parsing tokens for {dataset_path}: {tokens_str} ({e})")
    
    return entries


def find_leaf_folders(entries: List[Tuple[str, float, int]]) -> Dict[str, float]:
    """
    Identify leaf folders (those that don't have any children) and return their token counts
    """
    # Create a set of all paths
    all_paths = {entry[0] for entry in entries}
    
    # Find leaf folders - those that don't have any children
    leaf_folders = {}
    
    for path, tokens, depth in entries:
        # Check if this path has any children
        has_children = False
        for other_path in all_paths:
            if other_path != path and other_path.startswith(path + '/'):
                has_children = True
                break
        
        if not has_children:
            leaf_folders[path] = tokens
            print(f"Leaf folder: {path} -> {tokens:,.0f} tokens")
    
    return leaf_folders


def create_leaf_weights(data_stats_file: str, output_file: str = "dataset_weights_leaf.json", 
                       weighting_strategy: str = 'proportional'):
    """
    Create dataset weights JSON file based on leaf folders only
    
    Args:
        data_stats_file: Path to data_stats.txt
        output_file: Output JSON file path
        weighting_strategy: 'proportional' (default), 'sqrt', 'log', or 'uniform'
    """
    print(f"Parsing {data_stats_file}...")
    entries = parse_data_stats_hierarchical(data_stats_file)
    
    print(f"\nFinding leaf folders...")
    leaf_folders = find_leaf_folders(entries)
    
    if not leaf_folders:
        print("Error: No leaf folders found!")
        return
    
    print(f"\nFound {len(leaf_folders)} leaf folders")
    print(f"Total tokens in leaf folders: {sum(leaf_folders.values()):,.0f}")
    
    # Calculate weights based on strategy
    print(f"\nCalculating weights using '{weighting_strategy}' strategy...")
    
    if weighting_strategy == 'proportional':
        total_tokens = sum(leaf_folders.values())
        weights = {path: tokens / total_tokens for path, tokens in leaf_folders.items()}
    elif weighting_strategy == 'sqrt':
        import math
        sqrt_tokens = {path: math.sqrt(tokens) for path, tokens in leaf_folders.items()}
        total_sqrt = sum(sqrt_tokens.values())
        weights = {path: sqrt_val / total_sqrt for path, sqrt_val in sqrt_tokens.items()}
    elif weighting_strategy == 'log':
        import math
        log_tokens = {path: math.log(tokens + 1) for path, tokens in leaf_folders.items()}
        total_log = sum(log_tokens.values())
        weights = {path: log_val / total_log for path, log_val in log_tokens.items()}
    elif weighting_strategy == 'uniform':
        weights = {path: 1.0 / len(leaf_folders) for path in leaf_folders.keys()}
    else:
        raise ValueError(f"Unknown weighting strategy: {weighting_strategy}")
    
    # Create JSON structure with /phi4data suffix
    json_weights = {}
    for path, weight in weights.items():
        json_key = f"{path}/phi4data"
        json_weights[json_key] = weight
    
    # Save to file
    print(f"\nSaving weights to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(json_weights, f, indent=2, sort_keys=True)
    
    # Print statistics
    print(f"\nWeight statistics:")
    weight_values = list(weights.values())
    print(f"  Number of datasets: {len(weights)}")
    print(f"  Min weight: {min(weight_values):.6f}")
    print(f"  Max weight: {max(weight_values):.6f}")
    print(f"  Mean weight: {sum(weight_values) / len(weight_values):.6f}")
    print(f"  Sum of weights: {sum(weight_values):.6f}")
    
    # Show top 10 weighted datasets
    print(f"\nTop 10 datasets by weight:")
    sorted_weights = sorted([(path, weights[path], leaf_folders[path]) 
                           for path in weights.keys()], 
                          key=lambda x: x[1], reverse=True)
    for i, (path, weight, tokens) in enumerate(sorted_weights[:10]):
        print(f"  {i+1:2d}. {path:<50} weight={weight:.6f} tokens={tokens:,.0f}")
    
    print(f"\nGenerated {output_file} with {len(json_weights)} leaf datasets")
    return json_weights


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create dataset weights JSON for leaf folders only")
    parser.add_argument("--data_stats", default="data_stats.txt", 
                       help="Path to data_stats.txt file")
    parser.add_argument("--output", default="dataset_weights.json", 
                       help="Output JSON file path")
    parser.add_argument("--strategy", choices=['proportional', 'sqrt', 'log', 'uniform'], 
                       default='proportional',
                       help="Weighting strategy (default: proportional)")
    
    args = parser.parse_args()
    
    create_leaf_weights(args.data_stats, args.output, args.strategy) 