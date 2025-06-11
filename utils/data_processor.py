import os
from utils.file_loader import load_csv
from settings import (
    get_columns, set_features, set_target,
    select_model, filter_continuous_columns
)
from utils.visualization import plot_scatter, save_plot

def process_dataset(file_path: str):
    df = load_csv(file_path)
    print(df.head())

    dataset_name = os.path.splitext(os.path.basename(file_path))[0]
    model, columns = prepare_model_and_columns(df)

    if not columns:
        print("No valid columns available for the selected model.")
        return

    features = set_features(columns)
    target = set_target(columns, features)

    print(f'Selected features: {features}')
    print(f'Selected target: {target}')
    
    visualize_and_save(df, features, target, dataset_name)

def prepare_model_and_columns(df):
    model = select_model()
    columns = get_columns(df)
    if model == "linear_regression":
        columns = filter_continuous_columns(df)
    return model, columns

def visualize_and_save(df, features, target, dataset_name):
    user_input = input('Do you want to visualize the data? (yes/no): ').strip().lower()
    if user_input == 'yes' and len(features) > 0:
        fig, _ = plot_scatter(df, features, target)
        save_plot(fig, dataset_name, features, target)
        print(f"Plots saved in 'plots/{dataset_name}' directory.")
    else:
        print("No features selected for visualization.")