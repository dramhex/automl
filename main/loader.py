import pandas as pd

class DataLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.data = None
        self.columns = []  # Store column names

    def load_data(self) -> pd.DataFrame:
        """Load data from a CSV file."""
        try:
            self.data = pd.read_csv(self.file_path)
            self.columns = list(self.data.columns)  # Update columns after loading
            return self.data
        except FileNotFoundError:
            print(f"File not found: {self.file_path}")
            self.columns = []
            return pd.DataFrame()
        except pd.errors.EmptyDataError:
            print("No data found in the file.")
            self.columns = []
            return pd.DataFrame()
        except Exception as e:
            print(f"An error occurred while loading the data: {e}")
            self.columns = []
            return pd.DataFrame()
    
    def get_columns(self) -> list:
        """Return the list of column names from the loaded data."""
        if self.data is not None:
            return self.columns
        else:
            print("Data not loaded yet.")
            return []