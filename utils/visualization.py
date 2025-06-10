import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_scatter_matrix(df: pd.DataFrame, features: list, target: str) -> None:
    num_plots = len(features)
    fig, axes = plt.subplots(nrows=1, ncols=num_plots, figsize=(5 * num_plots, 5))

    if num_plots == 1:
        axes = [axes]  # Update the unique axis from the list

    # Make a folder to save the plots
    output_dir = 'plots'
    os.makedirs(output_dir, exist_ok=True)

    for i, feature in enumerate(features):
        axes[i].scatter(df[feature], df[target])
        axes[i].set_xlabel(feature)
        axes[i].set_ylabel(target)
        axes[i].set_title(f'Scatter Plot of {feature} vs {target}')

        # Save each plot as an image
        plot_filename = f'{output_dir}/{feature}_vs_{target}.png'
        plt.savefig(plot_filename)
    
    plt.tight_layout()
    plt.show()