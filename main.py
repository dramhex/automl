from utils.file_loader import load_csv

def main():
    df = load_csv('data/sample_dataset.csv')
    print(df.head())

if __name__ == '__main__':
    main()