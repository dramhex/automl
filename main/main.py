import visualizer, loader, learn, gui

# Load csv file
# describe
# ask for settings
# visualize
# type of regression
# learn
# show results

print("Starting the main program...")

def main():
    # Initialize the DataLoader with a file path
    file_path = "file_path"  # Replace with your actual file path
    data_loader = loader.DataLoader(file_path)
    
    # Load the data
    data = data_loader.load_data()
    print("Data loaded successfully.")
    print(data.head())  # Display the first few rows of the data
    print(data.describe())  # Display a summary of the data
    
    if data.empty:
        print("No data loaded. Exiting.")
        return
    
    # Initialize the DataVisualizer with the DataLoader
    visualizer_instance = visualizer.DataVisualizer(data)
    
    # Ask for columns to plot
    x_column = input(f"Available columns: {data_loader.get_columns()}\nEnter X column: ")
    y_column = input("Enter Y column: ")
    
    # Plot the data
    visualizer_instance.plot_data(x_column, y_column)
    
    '''
    # Learn from the data (placeholder for learning functionality)
    learn.learn_from_data(data, x_column, y_column)
    '''

main()