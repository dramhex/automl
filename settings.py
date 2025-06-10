import pandas as pd
import numpy as np

def select_model() -> str:
    print("Select a model:")
    print("1. Linear Regression")
    # Add other models here if needed
    choice = input("Enter the number of the model you want to use: ").strip()

    if choice == "1":
        return "linear_regression"
    else:
        print("Invalid choice. Please try again.")
        return select_model()

def filter_continuous_columns(df: pd.DataFrame) -> list:
    continuous_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    if not continuous_columns:
        print("No continuous columns available.")
        return []
    return continuous_columns

def get_columns(data_frame: pd.DataFrame) -> list:
    columns = list(data_frame)
    return columns

def set_features(columns: list) -> list:
    #Let user chose one or more features from the columns of the dataset
    while True:
        print(f'\nAvailable features: {columns}')
        user_input = input('Enter one or more features (separated with commas): ').strip()

        if not user_input:
            print('Please enter at least one feature.')
            continue

        features = [feature.strip() for feature in user_input.split(',')]
        valid_features = [f for f in features if f in columns]
        
        if not valid_features:
            print('No valid features entered. Please try again.')
            continue

        if len(valid_features) >= len(columns):
            print('You must leave at least one column for the target.')
            continue

        return valid_features
    
def set_target(columns: list, selected_features: list) -> str:
    #Let user chose a feature from the columns of the dataset
    remaining_columns = [col for col in columns if col not in selected_features]

    while True:
        print(f'\nAvailable columns for target: {remaining_columns}')
        user_input = input('Enter the target column: ').strip()

        if user_input in remaining_columns:
            return user_input
        else:
            print('Invalid target selection. Please select a valid column.')