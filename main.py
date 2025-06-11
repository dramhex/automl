from utils.data_processor import process_dataset

def main():
    file_path = 'data/weatherHistory.csv'
    #file_path = 'data/sample_salary_dataset.csv'
    
    process_dataset(file_path)

if __name__ == '__main__':
    main()