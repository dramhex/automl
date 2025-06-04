import matplotlib.pyplot as plt

from loader import DataLoader

class DataVisualizer:
    def __init__(self, data):
        self.data = data

    def plot_data(self, x_columns, y_column: str):
        """Plot scatterplots of multiple X columns vs a single Y column in one window."""
        if self.data is not None and not self.data.empty:
            plt.figure(figsize=(10, 6))
            for x_column in x_columns:
                plt.scatter(self.data[x_column], self.data[y_column], label=f'{x_column} vs {y_column}', alpha=0.7)
            plt.title(f'{y_column} vs ' + ', '.join(x_columns))
            plt.xlabel('X Columns')
            plt.ylabel(y_column)
            plt.legend()
            plt.grid()
            plt.show()
        else:
            print("No data available to plot.")