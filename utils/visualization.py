import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D 

max_points = 100 #For undersampling, depends on your GPU tho

def plot_scatter(df: pd.DataFrame, features: list, target: str, models: list):
    '''Makes scatter plots for every feature vs target'''
    num_plots = len(features)
    fig, axes = plt.subplots(nrows=1, ncols=num_plots, figsize=(5 * num_plots, 5))

    if num_plots == 1:
        axes = [axes]

    for i, feature in enumerate(features):
        x = df[feature].values
        y = df[target].values
        # Undersampling
        if len(x) > max_points:
            idx = np.random.choice(len(x), max_points, replace=False)
            x = x[idx]
            y = y[idx]

        data_plot = pd.DataFrame({feature: x, target: y})
        sns.scatterplot(data=data_plot, x=feature, y=target, ax=axes[i], color='royalblue', s=30, edgecolor=None)
        axes[i].set_xlabel(feature)
        axes[i].set_ylabel(target)
        axes[i].set_title(f'Scatter Plot of {feature} vs {target}')
    
        if models and len(models) > i and models[i] is not None:
            model = models[i]
            x_sorted = np.sort(x)
            y_pred = model.predict(x_sorted.reshape(-1, 1))
            axes[i].plot(x_sorted, y_pred, color='crimson', label='Regression line')
            axes[i].legend()

    plt.tight_layout()
    plt.show()
    return fig, axes

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

def plot_regression_result(df: pd.DataFrame, features: list, target: str, model, model_type: str, dataset_name):
    """
    Affiche la droite (2D) ou le plan (3D) de régression selon le modèle.
    """
    if model_type == "simple_linear" and len(features) == 1:
        x = df[features[0]].values
        y = df[target].values

        #Undersampling if needed
        if len(x) > max_points:
            idx = np.random.choice(len(x), max_points, replace=False)
            x_plot = x[idx]
            y_plot = y[idx]
        else:
            x_plot = x
            y_plot = y

        plt.figure(figsize=(8, 6))
        plt.scatter(x_plot, y_plot, color='blue', label='Data')
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
        if dataset_name:
            fig = plt.gcf()
            save_plot(fig, dataset_name, features, target, model_type)
        plt.show()

    elif model_type == "multiple_linear" and len(features) == 2:
        x1 = df[features[0]]
        x2 = df[features[1]]
        y = df[target]

        #Undersampling if needed
        if len(x1) > max_points*4:
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
        ax.scatter(x1_plot, x2_plot, y_plot, color='blue', label='Data')
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
        if dataset_name:
            fig = plt.gcf()
            save_plot(fig, dataset_name, features, target, model_type)
        plt.show()

    else:
        print("Visualisation de la régression non supportée pour ce cas.")
