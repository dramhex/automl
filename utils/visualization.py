import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_scatter(df: pd.DataFrame, features: list, target: str):
    '''Makes scatter plots for every feature vs target'''
    num_plots = len(features)
    fig, axes = plt.subplots(nrows=1, ncols=num_plots, figsize=(5 * num_plots, 5))

    if num_plots == 1:
        axes = [axes]

    for i, feature in enumerate(features):
        axes[i].scatter(df[feature], df[target])
        axes[i].set_xlabel(feature)
        axes[i].set_ylabel(target)
        axes[i].set_title(f'Scatter Plot of {feature} vs {target}')
    
    plt.tight_layout()
    return fig, axes

def save_plot(fig, dataset_name: str, features: list, target: str):
    output_dir = f'plots/{dataset_name}'
    os.makedirs(output_dir, exist_ok=True)
    
    # Saves each feature in a distinct file
    base_filename = '_'.join(features) + f'_vs_{target}.png'
    plot_filename = f'{output_dir}/{base_filename}'
    fig.savefig(plot_filename)
    
    plt.close(fig)