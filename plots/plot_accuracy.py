import json
import os
import re
import matplotlib.pyplot as plt
import numpy as np

# Set global font sizes for LaTeX
plt.rcParams.update({
    'font.size': 24,
    'axes.labelsize': 24,
    'axes.titlesize': 24,
    'xtick.labelsize': 24,
    'ytick.labelsize': 24,
    'legend.fontsize': 24,
    'figure.titlesize': 24
})

def extract_swa_size(filename):
    # Extract the number after 'swa' in the filename
    match = re.search(r'swa(\d+)', filename)
    return int(match.group(1)) if match else None

def extract_model_name(filename):
    # Extract the model name between 'mup_' and '_d16'
    match = re.search(r'mup_(.+?)_d16', filename)
    return match.group(1) if match else None

def plot_accuracy_vs_swa():
    # Directory containing the JSON files
    results_dir = 'prolong_results_32k'
    #results_dir = 'slim_results_32k'
    # Dictionary to store results for each model
    model_results = {}
    transformer_results = {}
    
    # Process each JSON file
    for filename in os.listdir(results_dir):
        if not filename.endswith('.json'):
            continue
            
        # Extract model name and SWA size
        model_name = extract_model_name(filename)
        #print(model_name)
        swa_size = extract_swa_size(filename)
        
        if model_name is None:
            continue
            
        # Read the JSON file
        with open(os.path.join(results_dir, filename), 'r') as f:
            data = json.load(f)
            
        # Get the accuracy and std for the last entry (assuming it's the final result)
        if data:
            final_result = data[-1]
            mean_acc = final_result['mean_acc']
            std_acc = final_result['std_acc']
            
            # Store results
            if "transformer" in model_name.lower() and (not "transformerls" in model_name.lower()):
                if model_name not in transformer_results:
                    transformer_results[model_name] = {'mean_acc': mean_acc, 'std_acc': std_acc}
            else:
                if model_name not in model_results:
                    model_results[model_name] = {'swa_sizes': [], 'accuracies': [], 'stds': []}
                
                model_results[model_name]['swa_sizes'].append(swa_size)
                model_results[model_name]['accuracies'].append(mean_acc)
                model_results[model_name]['stds'].append(std_acc)
    
    # Create the plot with larger figure size
    plt.figure(figsize=(16, 12))
    
    # # Define colors for each model
    # colors = {
    #     'sambay': 'blue',
    #     'sambayoco': 'red',
    #     'sambayda': 'green',
    #     'transformerls': 'purple',
    #     'tie_rbase_transformerls': 'orange',
    #     'tie_rbase_prolong_transformerls': 'orange'
    # }
    
    mmap={
        "tie_prolong_varlen_sambay": "SambaY",
        "tie_prolong_varlen_sambayoco": "Samba+YOCO",
        "tie_prolong_varlen_sambayda": "SambaY+DA",
        "tie_rbase_prolong_varlen_transformerls": "TransformerLS",
        "transformer": "Transformer++",
        "tie_prolong_varlen_swayoco": "SWA+YOCO",
    }
    model_results["tie_prolong_varlen_swayoco"] = {
        "swa_sizes": [64, 128, 256, 512, 1024, 2048],
        "accuracies": [13.28/100, 37.50/100, 30.47/100, 13.28/100, 16.41/100, 8.59/100],
        "stds": [3.40/100, 4.42/100, 7.45/100, 2.59/100, 4.62/100, 4.62/100],
    }
    # mmap={
    #     "sambay": "SambaY",
    #     "sambayoco": "Samba+YOCO",
    #     "sambayda": "SambaY+DA",
    #     "transformerls": "TransformerLS (RoPE Base: 1e4)",
    #     "tie_rbase_transformerls": "TransformerLS",
    #     "transformer": "Transformer++",
    # }
    # Plot each model's results
    for model_name, results in model_results.items():
        if not "varlen" in model_name:
            continue
        print(model_name,results)
        # Sort by SWA size
        sorted_indices = np.argsort(results['swa_sizes'])
        swa_sizes = np.array(results['swa_sizes'])[sorted_indices]
        accuracies = np.array(results['accuracies'])[sorted_indices]
        stds = np.array(results['stds'])[sorted_indices]
        
        # Plot with error bars
        plt.errorbar(swa_sizes, accuracies, yerr=stds, 
                    label=mmap.get(model_name, model_name), #'_'.join(model_name.split('_')[:-1]), model_name.split('_')[-1]), 
                    marker='o', 
                    markersize=16,
                    capsize=12, 
                    capthick=3, 
                    elinewidth=3,
                    linewidth=3)
                    # color=colors.get(model_name, None))
    
    # Add horizontal lines for transformer results
    for model_name, result in transformer_results.items():
        if not "varlen" in model_name:
            continue
        plt.axhline(y=result['mean_acc'], 
                   color='gray', 
                   linestyle='--', 
                   alpha=0.5,
                   linewidth=2,
                   label=f"{mmap.get(model_name.split('_')[-1], model_name.split('_')[-1])}")
        # Add error bands
        plt.fill_between([64, 2048], 
                        [result['mean_acc'] - result['std_acc'], result['mean_acc'] - result['std_acc']],
                        [result['mean_acc'] + result['std_acc'], result['mean_acc'] + result['std_acc']],
                        color='gray', alpha=0.1)
    
    # Customize the plot
    plt.xlabel('Sliding Window Size', fontsize=24)
    plt.ylabel('Accuracy', fontsize=24)
    #plt.title('Accuracy vs Sliding Window Size on Phonebook 32K', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7, linewidth=1.5)
    plt.legend(fontsize=22, loc='lower left', bbox_to_anchor=(0.0, 0.0))
    
    # Use log scale for x-axis since SWA sizes are exponentially increasing
    plt.xscale('log', base=2)
    
    # Set x-axis ticks to match the SWA sizes
    plt.xticks([64, 128, 256, 512, 1024, 2048], 
               ['64', '128', '256', '512', '1024', '2048'])
    # Set y-axis limits from 0 to 1
    plt.ylim(0, 1)
    # Adjust layout and save with higher DPI
    plt.tight_layout()
    plt.savefig(results_dir+'_accuracy_vs_swa.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    plot_accuracy_vs_swa() 