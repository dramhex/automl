from utils.file_loader import load_csv
from settings import get_columns, set_features, set_target, select_model, filter_continuous_columns
from utils.visualization import plot_scatter_matrix

def main():
    df = load_csv('data/sample_salary_dataset.csv')
    print(df.head())

    model = select_model()

    columns = get_columns(df)

    # Filter the columns if the model is a linear regression
    if model == "linear_regression":
        columns = filter_continuous_columns(df)

    if not columns:
        print("No valid columns available for the selected model.")
        return
    
    features = set_features(columns)
    target = set_target(columns, features)

    print(f'Selected features: {features}')
    print(f'Selected target: {target}')

    user_input = input('Do you want to visualize the data? (yes/no): ').strip().lower()
    if user_input == 'yes':
        plot_scatter_matrix(df, features, target)

if __name__ == '__main__':
    main()