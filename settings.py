import pandas as pd
import numpy as np
import time

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

        self.columns_by_type = {
            'continuous': continuous_columns,
            'categorical': categorical_columns,
            'all': df.columns.tolist()
        }

        print(f"Columns by type: {self.columns_by_type}")

        return self.columns_by_type

    def filter_columns(self) -> list:
            if self.model_type in ('linear_regression', 'non_linear_regression'):
                return self.columns_by_type['continuous']
            else:
                return self.columns_by_type['all']

    def select_model(self) -> str:
        print("Select a learning model:")
        for i, name in enumerate(self.models_names, 1):
            print(f"{i}. {name.replace('_', ' ').title()}")
        while True:
            try:
                choice = int(input("Enter the number of the model you want to use: ").strip())
                if 1 <= choice <= len(self.models_names):
                    self.model_type = self.models_names[choice - 1]
                    return self.model_type
                else:
                    print(f"Enter a number between 1 and {len(self.models_names)}")
            except ValueError:
                print("Enter a valid number.")

    def set_target(self) -> str:
        columns = self.filter_columns()

        while True:
            print("\nAvailable columns for target:")
            for idx, col in enumerate(columns, 1):
                print(f"{idx}. {col}")

            user_input = input('Enter the number of the target column: ').strip()
            
            try:
                index = int(user_input) - 1
            except ValueError:
                print("Invalid input. Please enter a number.")
                continue
            
            if 0 <= index < len(columns):
                self.target = columns[index]
                return columns[index]
            else:
                print('Invalid target selection. Please select a valid column.')

    def set_features(self) -> list:
        columns = self.filter_columns()
        
        remaining_columns = [col for col in columns if col not in self.target]

        while True:
            print("\nAvailable features:")
            for idx, col in enumerate(remaining_columns, 1):
                print(f"{idx}. {col}")

            print("You can select one or more features (at least one).")
            user_input = input('Enter one or more feature numbers (separated by spaces): ').strip()
            indices = user_input.split()

            try:
                selected_indices = [int(index) - 1 for index in indices]
            except ValueError:
                print("Invalid input. Please enter numbers only.")
                time.sleep(2)
                continue

            valid_features = [remaining_columns[i] for i in selected_indices if 0 <= i < len(remaining_columns)]

            if not valid_features:
                print('No valid features entered. Please try again.')
                time.sleep(2)
                continue

            self.features = valid_features
            return valid_features

    def display_settings(self):
        print("\nCurrent settings:")
        print(f"Model Type: {self.model_type}")
        print(f"Target: {self.target}")
        print(f"Features: {self.features}")

settings_instance = Settings()