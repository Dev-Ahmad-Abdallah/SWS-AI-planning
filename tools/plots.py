"""
Plotting utilities for comparison and ablation studies.

Purpose: Generate matplotlib charts comparing A* vs PRM, ablation plots.

Inputs:
    - CSV log files from data/logs/
    - Output directory (docs/img/)

Outputs:
    - PNG plots with proper titles, axes, captions

Params:
    input_pattern: str - Glob pattern for CSV files
    output_dir: str - Output directory for plots
"""

import argparse
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os


def load_logs(input_pattern):
    """Load CSV logs matching pattern."""
    csv_files = glob.glob(input_pattern)
    
    if not csv_files:
        print(f"No CSV files found matching {input_pattern}")
        return None
    
    dfs = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            dfs.append(df)
        except Exception as e:
            print(f"Error loading {csv_file}: {e}")
    
    if not dfs:
        return None
    
    return pd.concat(dfs, ignore_index=True)


def plot_comparison(df, output_dir):
    """Generate comparison plots: A* vs PRM."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Group by planner
    grouped = df.groupby('planner')
    
    # Metrics to compare
    metrics = ['path_len', 'cpu_ms', 'efficiency', 'success', 'replans', 'min_clear']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        # Get data per planner
        a_star_data = grouped.get_group('a_star')[metric] if 'a_star' in grouped.groups else []
        prm_data = grouped.get_group('prm')[metric] if 'prm' in grouped.groups else []
        
        if len(a_star_data) > 0 or len(prm_data) > 0:
            data = [a_star_data, prm_data]
            labels = ['A*', 'PRM']
            
            # Box plot
            ax.boxplot(data, labels=labels)
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_title(f'{metric.replace("_", " ").title()} Comparison')
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/comparison.png", dpi=300, bbox_inches='tight')
    print(f"Saved comparison plot to {output_dir}/comparison.png")
    plt.close()
    
    # Bar chart for success rate
    fig, ax = plt.subplots(figsize=(8, 6))
    
    if 'a_star' in grouped.groups and 'prm' in grouped.groups:
        a_star_success = grouped.get_group('a_star')['success'].mean()
        prm_success = grouped.get_group('prm')['success'].mean()
        
        ax.bar(['A*', 'PRM'], [a_star_success, prm_success], color=['blue', 'orange'])
        ax.set_ylabel('Success Rate')
        ax.set_title('Success Rate Comparison')
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/success_rate.png", dpi=300, bbox_inches='tight')
    print(f"Saved success rate plot to {output_dir}/success_rate.png")
    plt.close()


def plot_ablation(df, output_dir):
    """Generate ablation plots."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Example: Plot efficiency vs path length
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for planner in df['planner'].unique():
        planner_df = df[df['planner'] == planner]
        ax.scatter(planner_df['path_len'], planner_df['efficiency'],
                  label=planner.upper(), alpha=0.6)
    
    ax.set_xlabel('Path Length')
    ax.set_ylabel('Efficiency')
    ax.set_title('Path Length vs Efficiency')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/ablation.png", dpi=300, bbox_inches='tight')
    print(f"Saved ablation plot to {output_dir}/ablation.png")
    plt.close()


def main():
    """Main plotting function."""
    parser = argparse.ArgumentParser(description="Generate plots from CSV logs")
    parser.add_argument('--in', '--input', dest='input_pattern',
                       default='data/logs/*.csv',
                       help='Input CSV file pattern (glob)')
    parser.add_argument('--out', '--output', dest='output_dir',
                       default='docs/img/',
                       help='Output directory for plots')
    
    args = parser.parse_args()
    
    # Load logs
    df = load_logs(args.input_pattern)
    
    if df is None:
        print("No data to plot")
        return
    
    print(f"Loaded {len(df)} log entries")
    
    # Generate plots
    plot_comparison(df, args.output_dir)
    plot_ablation(df, args.output_dir)
    
    print(f"\nPlots saved to {args.output_dir}")


if __name__ == "__main__":
    main()

