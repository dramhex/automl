import os
from utils.file_loader import load_csv
from settings import (
    get_columns, set_features, set_target,
    select_model
)
from utils.visualization import plot_scatter, save_plot, plot_regression_result
from models.linear_regression import train_simple_linear, train_multiple_linear

def process_dataset(file_path: str):
    df = load_csv(file_path)
    print(df.head())

    dataset_name = os.path.splitext(os.path.basename(file_path))[0]
    model_type = select_model()
    columns = get_columns(df)

    if not columns:
        print("No valid columns available for the selected model.")
        return

    features = set_features(columns, model_type)
    target = set_target(columns, features, model_type)

    print(f'Selected features: {features}')
    print(f'Selected target: {target}')

    if model_type == "simple_linear":
        trained_model = train_simple_linear(df, features, target)
    elif model_type == "multiple_linear":
        trained_model = train_multiple_linear(df, features, target)
    
    visualize_and_save(df, features, target, dataset_name)

    plot_regression_result(df, features, target, trained_model, model_type)

def visualize_and_save(df, features, target, dataset_name):
    user_input = input('Do you want to visualize the data? (yes/no): ').strip().lower()
    if user_input == 'yes' and len(features) > 0:
        fig, _ = plot_scatter(df, features, target)
        save_plot(fig, dataset_name, features, target)
        print(f"Plots saved in 'plots/{dataset_name}' directory.")
    else:
        print("No features selected for visualization.")