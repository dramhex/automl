import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_scatter_matrix(df: pd.DataFrame, features: list, target: str) -> None:
    num_plots = len(features)
    fig, axes = plt.subplots(nrows=1, ncols=num_plots, figsize=(5 * num_plots, 5))

    # Gérer le cas où il n'y a qu'un seul graphique
    if num_plots == 1:
        axes = [axes]  # Mettre l'objet axe unique dans une liste

    for i, feature in enumerate(features):
        axes[i].scatter(df[feature], df[target])
        axes[i].set_xlabel(feature)
        axes[i].set_ylabel(target)
        axes[i].set_title(f'Scatter Plot of {feature} vs {target}')
    
    plt.tight_layout()
    plt.show()