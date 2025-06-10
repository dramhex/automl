from utils.file_loader import load_csv
from settings import get_columns, set_features, set_target
from utils.visualization import plot_scatter_matrix

def main():
    df = load_csv('data/sample_salary_dataset.csv')
    print(df.head())

    columns = get_columns(df)
    features = set_features(columns)
    target = set_target(columns, features)

    print(f'Selected features: {features}')
    print(f'Selected target: {target}')

    user_input = input('Do you want to visualize the data? (yes/no): ').strip().lower()
    if user_input == 'yes':
        if len(features) > 0:
            plot_scatter_matrix(df, features, target)
        else:
            print("No features selected for visualization.")

if __name__ == '__main__':
    main()