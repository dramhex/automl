import os
from utils.file_loader import load_csv
from settings import settings_instance as settings

from utils.visualization import plot_scatter, plot_3d, save_plot 
from models.regression import train_models

def process_dataset(file_path: str):
    df = load_csv(file_path)
    dataset_name = os.path.splitext(os.path.basename(file_path))[0]
    columns = settings.get_columns(df)

    if not columns:
        print("No valid columns available for the selected model.")
        return

    model_type = settings.select_model()

    

    target = settings.set_target()
    features = settings.set_features()

    print(f'Selected features: {features}')
    print(f'Selected target: {target}')

    settings.display_settings

    models, splits = train_models(df, features, target, model_type)

    # Visualisation
    if model_type == "linear_regression":
        user_input = input('Do you want to visualize the data? (yes/no): ').strip().lower()
        if user_input == 'yes' and len(features) > 0:
            fig = plot_scatter(df, features, target, models=models)
            save_plot(fig, dataset_name, features, target, model_type=model_type)
    elif model_type == "multiple_linear" and len(features) == 2:
        user_input = input('Do you want to visualize the 3D regression? (yes/no): ').strip().lower()
        if user_input == 'yes':
            # models[0] is the only model for multiple linear regression
            fig = plot_3d(df, features, target, models[0])
            save_plot(fig, dataset_name, features, target, model_type=model_type)
    elif model_type == "multiple_linear" and len(features) > 2:
        print("Visualization is not supported for multiple linear regression with more than 2 features.")