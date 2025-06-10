from utils.file_loader import load_csv
from settings import get_columns, set_features
def main():
    df = load_csv('data/sample_salary_dataset.csv')
    print(df.head())

    columns = get_columns(df)

    print(set_features(columns))

if __name__ == '__main__':
    main()