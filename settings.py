import pandas as pd
import numpy as np

def select_model() -> str:
    print("Select a learning model:")
    print("1. Simple Linear Regression")
    print("2. Multiple Linear Regression")
    choice = input("Enter the number of the model you want to use: ").strip()
    if choice == "1":
        return "simple_linear"
    elif choice == "2":
        return "multiple_linear"
    else:
        print("Invalid choice. Please try again.")
        return select_model()

def get_columns(df: pd.DataFrame) -> dict:
    continuous_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = df.select_dtypes(exclude=[np.number]).columns.tolist()

    columns_by_type = {
        'continuous': continuous_columns,
        'categorical': categorical_columns,
        'all': df.columns.tolist()
    }

    print(f"Columns by type: {columns_by_type}")
    return columns_by_type

def filter_columns(columns_by_type: dict, model) -> list:
    if model in ('simple_linear', 'multiple_linear'):
        columns = columns_by_type['continuous']
    else:
        columns = columns_by_type['all']
    return columns

def set_features(columns_by_type: dict, model: str) -> list:
    columns = filter_columns(columns_by_type, model)
    
    while True:
        print("\nAvailable features:")
        for idx, col in enumerate(columns, 1):
            print(f"{idx}. {col}")

        user_input = input('Enter one or more feature numbers (separated by spaces): ').strip()
        indices = user_input.split()

        try:
            selected_indices = [int(index) - 1 for index in indices]
        except ValueError:
            print("Invalid input. Please enter numbers only.")
            continue

        valid_features = [columns[i] for i in selected_indices if 0 <= i < len(columns)]
        
        if not valid_features:
            print('No valid features entered. Please try again.')
            continue

        if model == "simple_linear" and len(valid_features) != 1:
            print('Simple Linear Regression requires exactly one feature.')
            continue

        if len(valid_features) >= len(columns):
            print('You must leave at least one column for the target.')
            continue

        return valid_features

def set_target(columns_by_type: dict, selected_features: list, model: str) -> str:
    columns = filter_columns(columns_by_type, model)

    remaining_columns = [col for col in columns if col not in selected_features]

    while True:
        print("\nAvailable columns for target:")
        for idx, col in enumerate(remaining_columns, 1):
            print(f"{idx}. {col}")

        user_input = input('Enter the number of the target column: ').strip()
        
        try:
            index = int(user_input) - 1
        except ValueError:
            print("Invalid input. Please enter a number.")
            continue
        
        if 0 <= index < len(remaining_columns):
            return remaining_columns[index]
        else:
            print('Invalid target selection. Please select a valid column.')