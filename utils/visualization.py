import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D 

max_points = 100 #For undersampling, depends on your GPU tho

def plot_scatter(df: pd.DataFrame, features: list, target: str, models: list):
    '''Makes scatter plots for every feature vs target, 3 plots per row max'''
    max_cols = 3
    num_plots = len(features)
    ncols = min(num_plots, max_cols)
    nrows = (num_plots + max_cols - 1) // max_cols

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 5 * nrows))
    axes = np.array(axes).reshape(-1)  # Flatten in case of single row/col

    for i, feature in enumerate(features):
        ax = axes[i]
        x = df[feature].values
        y = df[target].values
        # Undersampling
        if len(x) > max_points:
            idx = np.random.choice(len(x), max_points, replace=False)
            x = x[idx]
            y = y[idx]

        data_plot = pd.DataFrame({feature: x, target: y})
        sns.scatterplot(data=data_plot, x=feature, y=target, ax=ax, color='royalblue', s=30, edgecolor=None)
        ax.set_xlabel(feature)
        ax.set_ylabel(target, fontsize=3)
        ax.set_title(f'Scatter Plot of {feature} vs {target}')

        if models and len(models) > i and models[i] is not None:
            model = models[i]
            x_sorted = np.sort(x)
            y_pred = model.predict(x_sorted.reshape(-1, 1))
            ax.plot(x_sorted, y_pred, color='crimson', label='Regression line')
            ax.legend()

    # Hide unused subplots if any
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()
    return fig

def save_plot(fig, dataset_name: str, features: list, target: str, model_type: str = None):
    # Ajoute le type de modèle dans le chemin si précisé
    if model_type:
        output_dir = f'plots/{dataset_name}/{model_type}'
    else:
        output_dir = f'plots/{dataset_name}'
    os.makedirs(output_dir, exist_ok=True)

    # Shorten filename if too many features
    if len(features) == 1:
        feature_part = features[0]
    elif len(features) == 2:
        feature_part = f"{features[0]}_{features[1]}"
    else:
        feature_part = f"{features[0]}_and_more"

    base_filename = f'{feature_part}_vs_{target}.png'
    plot_filename = f'{output_dir}/{base_filename}'
    fig.savefig(plot_filename)
    print(f"Plot saved in '{plot_filename}'")
    plt.close(fig)

def plot_3d(df: pd.DataFrame, features: list, target: str, model):
    """
    Affiche un scatter 3D et le plan de régression pour une régression multiple à 2 features.
    """
    x1 = df[features[0]].values
    x2 = df[features[1]].values
    y = df[target].values

    # Undersampling
    max_points = 100
    if len(x1) > max_points:
        idx = np.random.choice(len(x1), max_points, replace=False)
        x1_plot = x1[idx]
        x2_plot = x2[idx]
        y_plot = y[idx]
    else:
        x1_plot = x1
        x2_plot = x2
        y_plot = y

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x1_plot, x2_plot, y_plot, color='royalblue', s=20, alpha=0.7, label='Data')

    # Plan de régression
    grid_points = 20
    x1_grid, x2_grid = np.meshgrid(
        np.linspace(x1.min(), x1.max(), grid_points),
        np.linspace(x2.min(), x2.max(), grid_points)
    )
    X_grid = np.c_[x1_grid.ravel(), x2_grid.ravel()]
    y_pred = model.predict(X_grid).reshape(x1_grid.shape)
    ax.plot_surface(x1_grid, x2_grid, y_pred, color='crimson', alpha=0.4, edgecolor='none')
    ax.set_xlabel(features[0])
    ax.set_ylabel(features[1])
    ax.set_zlabel(target)
    ax.set_title('Multiple Linear Regression (2 features)')

    plt.tight_layout()
    plt.show()
    return fig