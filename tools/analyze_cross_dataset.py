import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_dataset_results(results_dir):
    """Load results from all datasets."""
    results = {}
    results_path = Path(results_dir)
    
    # Define dataset configurations
    datasets = {
        'flowers102': {
            'single_template': 'one_template',
            'multi_template': 'multi_templates'
        },
        'cifar10': {
            'single_template': 'one_template',
            'multi_template': 'multi_templates'
        },
        'caltech101': {
            'single_template': 'one_template',
            'multi_template': 'multi_templates'
        }
    }
    
    for dataset, configs in datasets.items():
        dataset_results = {}
        
        for template_type, subdir in configs.items():
            dataset_path = results_path / dataset / subdir
            if dataset_path.exists():
                dataset_results[template_type] = {}
                
                for csv_file in dataset_path.glob("*.csv"):
                    method_name = csv_file.stem.replace("_results", "")
                    df = pd.read_csv(csv_file)
                    
                    # Get clean performance
                    clean_data = df[df['attack'] == 'none']
                    if len(clean_data) > 0:
                        clean_row = clean_data.iloc[0]
                        dataset_results[template_type][method_name] = {
                            'top1': clean_row['top1'],
                            'macro_f1': clean_row['macro_f1'],
                            'ece': clean_row['ece']
                        }
        
        if dataset_results:
            results[dataset] = dataset_results
    
    return results

def create_cross_dataset_plot(results, output_dir):
    """Create cross-dataset performance comparison."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Extract data for plotting
    datasets = list(results.keys())
    methods = ['zshot', 'coop', 'tipadapter']
    method_labels = ['Zero-shot', 'CoOp', 'Tip-Adapter']
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    metrics = ['top1', 'macro_f1', 'ece']
    metric_labels = ['Top-1 Accuracy', 'Macro F1', 'ECE']
    
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[i]
        
        # Plot bars for each method across datasets
        x = np.arange(len(datasets))
        width = 0.25
        
        for j, (method, method_label) in enumerate(zip(methods, method_labels)):
            values = []
            for dataset in datasets:
                if 'single_template' in results[dataset] and method in results[dataset]['single_template']:
                    values.append(results[dataset]['single_template'][method][metric])
                else:
                    values.append(0)
            
            ax.bar(x + j * width, values, width, label=method_label)
        
        ax.set_xlabel('Dataset')
        ax.set_ylabel(label)
        ax.set_title(f'{label} Across Datasets')
        ax.set_xticks(x + width)
        ax.set_xticklabels([d.replace('102', '-102') for d in datasets])
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'cross_dataset_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Cross-dataset plot saved to: {output_path / 'cross_dataset_comparison.png'}")

def create_dataset_summary_table(results, output_file):
    """Create summary table of results across datasets."""
    summary_data = []
    
    for dataset, dataset_results in results.items():
        for template_type, methods in dataset_results.items():
            for method, metrics in methods.items():
                summary_data.append({
                    'Dataset': dataset.replace('102', '-102'),
                    'Template Type': template_type.replace('_', ' ').title(),
                    'Method': method.replace('_', ' ').title(),
                    'Top-1': metrics['top1'],
                    'Macro F1': metrics['macro_f1'],
                    'ECE': metrics['ece']
                })
    
    df = pd.DataFrame(summary_data)
    df = df.round(4)
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"Dataset summary saved to: {output_file}")
    
    # Print table
    print("\n=== Cross-Dataset Performance Summary ===")
    print(df.to_string())
    
    return df

def create_robustness_comparison(results, output_dir):
    """Create robustness comparison across datasets."""
    output_path = Path(output_dir)
    
    # This would require loading the full attack results
    # For now, create a placeholder
    print("Robustness comparison across datasets - requires full attack data")
    print("This would show how different datasets respond to attacks")

def main():
    parser = argparse.ArgumentParser(description='Analyze results across datasets')
    parser.add_argument('--results_dir', type=str, default='reports/tables',
                       help='Results directory containing dataset subdirectories')
    parser.add_argument('--output_dir', type=str, default='reports/figs',
                       help='Output directory for plots')
    parser.add_argument('--output_file', type=str, default='reports/tables/cross_dataset_summary.csv',
                       help='Output file for summary table')
    args = parser.parse_args()
    
    print("=== Cross-Dataset Analysis ===")
    print(f"Results directory: {args.results_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Output file: {args.output_file}")
    
    # Load results
    results = load_dataset_results(args.results_dir)
    
    if not results:
        print("❌ No dataset results found!")
        return
    
    print(f"Found results for datasets: {list(results.keys())}")
    
    # Create plots and tables
    create_cross_dataset_plot(results, args.output_dir)
    create_dataset_summary_table(results, args.output_file)
    create_robustness_comparison(results, args.output_dir)
    
    print("✅ Cross-dataset analysis complete!")

if __name__ == "__main__":
    main()
