import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D 

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

def plot_regression_result(df: pd.DataFrame, features: list, target: str, model, model_type: str):
    """
    Affiche la droite (2D) ou le plan (3D) de régression selon le modèle.
    """
    if model_type == "simple_linear" and len(features) == 1:
        x = df[features[0]].values
        y = df[target].values
        plt.figure(figsize=(8, 6))
        plt.scatter(x, y, color='blue', label='Data')
        # Trie x pour que la droite soit bien tracée
        sort_idx = np.argsort(x)
        x_sorted = x[sort_idx]
        y_pred = model.coef_[0] * x_sorted + model.intercept_
        plt.plot(x_sorted, y_pred, color='red', label='Regression line')
        plt.xlabel(features[0])
        plt.ylabel(target)
        plt.title('Simple Linear Regression')
        plt.xlim(x.min(), x.max())
        plt.ylim(y.min(), y.max())
        plt.legend()
        plt.show()

    elif model_type == "multiple_linear" and len(features) == 2:
        x1 = df[features[0]]
        x2 = df[features[1]]
        y = df[target]
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x1, x2, y, color='blue', label='Data')
        # Plan de régression
        x1_grid, x2_grid = np.meshgrid(
            np.linspace(x1.min(), x1.max(), 20),
            np.linspace(x2.min(), x2.max(), 20)
        )
        X_grid = np.c_[x1_grid.ravel(), x2_grid.ravel()]
        y_pred = model.predict(X_grid).reshape(x1_grid.shape)
        ax.plot_surface(x1_grid, x2_grid, y_pred, color='red', alpha=0.5)
        ax.set_xlabel(features[0])
        ax.set_ylabel(features[1])
        ax.set_zlabel(target)
        ax.set_title('Multiple Linear Regression (2 features)')
        plt.show()
    else:
        print("Visualisation de la régression non supportée pour ce cas.")
