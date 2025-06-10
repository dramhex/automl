import pandas as pd

def get_columns(data_frame: pd.DataFrame) -> list:
    columns = list(data_frame)
    return columns

def set_features(columns: list) -> list:
    while True:
        print(f'\n Available features : \n {columns}')
        user_input = input('Enter one or more features (space-separated): ').strip()

        if not user_input:
            print('Please enter at least one feature')
            continue

        features = user_input.split()

        invalid = [f for f in features if f not in columns]
        if invalid:
            print(f'Invalid features : {invalid}. \n Please choose from {columns}')
            continue

        return features