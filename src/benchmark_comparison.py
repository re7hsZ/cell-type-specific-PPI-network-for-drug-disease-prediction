"""
Benchmark Comparison Script

Orchestrates the execution of Cell-Type Specific and Global PPI models,
collects evaluation metrics, and generates a comparative visualization.
"""

import os
import subprocess
import json
import matplotlib.pyplot as plt
import numpy as np

def run_benchmark():
    """
    Runs both model variants (Cell-Type Specific and Global PPI) and generates
    a comparison plot of their performance metrics.
    """
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_script = os.path.join(base_path, "src", "train.py")
    results_dir = os.path.join(base_path, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    celltype = "acinar_cell_of_salivary_gland"
    epochs = 10
    
    # Execute Cell-Type Specific Model
    print(f"Running Cell-Type Specific Model ({celltype})...")
    ct_results_file = os.path.join(results_dir, "celltype_results.json")
    subprocess.run([
        "python", train_script,
        "--celltype", celltype,
        "--epochs", str(epochs),
        "--save_results", ct_results_file
    ], check=True)
    
    # Execute Global PPI Model
    print("Running Global PPI Model...")
    global_results_file = os.path.join(results_dir, "global_results.json")
    subprocess.run([
        "python", train_script,
        "--use_global_ppi",
        "--epochs", str(epochs),
        "--save_results", global_results_file
    ], check=True)
    
    # Load Results
    with open(ct_results_file, 'r') as f:
        ct_metrics = json.load(f)
    with open(global_results_file, 'r') as f:
        global_metrics = json.load(f)
        
    # Generate Visualization
    visualize_comparison(ct_metrics, global_metrics, results_dir)

def visualize_comparison(ct_metrics, global_metrics, output_dir):
    """
    Generates a grouped bar chart comparing performance metrics.
    """
    metrics = list(ct_metrics.keys())
    ct_values = [ct_metrics[m] for m in metrics]
    global_values = [global_metrics[m] for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, ct_values, width, label='Cell-Type Specific')
    rects2 = ax.bar(x + width/2, global_values, width, label='Global PPI')
    
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    
    ax.bar_label(rects1, padding=3, fmt='%.3f')
    ax.bar_label(rects2, padding=3, fmt='%.3f')
    
    fig.tight_layout()
    
    output_path = os.path.join(output_dir, "comparison_results.png")
    plt.savefig(output_path)
    print(f"Comparison plot saved to {output_path}")

if __name__ == "__main__":
    run_benchmark()
