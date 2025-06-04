import matplotlib.pyplot as plt

from loader import DataLoader

class DataVisualizer:
    def __init__(self, data):
        self.data = data

    def plot_data(self, x_column: str, y_column: str):
        """Plot data using specified columns."""
        if self.data is not None and not self.data.empty:
            plt.figure(figsize=(10, 6))
            plt.scatter(self.data[x_column], self.data[y_column], marker='o')
            plt.title(f'{y_column} vs {x_column}')
            plt.xlabel(x_column)
            plt.ylabel(y_column)
            plt.grid()
            plt.show()
        else:
            print("No data available to plot.")