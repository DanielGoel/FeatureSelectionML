import pandas as pd
import matplotlib.pyplot as plt
import os

# Function to count occurrences of each label in the CSV and plot the bar graph
def plot_label_counts(dataset_name, 
                    csv_file, save_path):
    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_file)

    # Map the labels to the corresponding categories
#    label_mapping = {
#        0: 'Normal',
#        3: 'PID Gain', 4: 'PID Gain',
#        5: 'PID Reset Rate', 6: 'PID Reset Rate',
#        7: 'PID Rate',
#        1: 'Setpoint', 2: 'Setpoint'
#    }

    # Create a new column 'category' based on the label column
#    df['category'] = df['Label'].map(label_mapping)

    df['category'] = df['Label']
    # Count the occurrences of each category
    category_counts = df['category'].value_counts().sort_index()

    # Create the bar chart
    plt.figure(figsize=(10, 6))
    ax = category_counts.plot(kind='bar', color='dodgerblue', width=0.8) 

    # Add the count on top of each bar
    for bar in ax.patches:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval + 1000, f'{int(yval)}', ha='center', fontweight='bold')

    # Customize the plot
    plt.xlabel('Attack Category')
    plt.ylabel('Count')
    plt.title('Attack Counts by Category')
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Ensure the directory exists
    os.makedirs(save_path, exist_ok=True)

    # Save the plot to the specified directory
    save_file = os.path.join(save_path, f'label_counts_plot_{os.path.basename(dataset_name)}.png')
    plt.savefig(save_file)
    print(f"Plot saved to {save_file}")

    # Close the plot to free up memory
    plt.close()

csv_files = {
    "function": "GaspipelineDatasets/NewGasFilteredFunctionMinMax_Remapped.csv",
    "command": "GaspipelineDatasets/NewGasFilteredCommandMinMax_Remapped.csv",
    "all": "GaspipelineDatasets/NewGasFilteredAllMinMax.csv",
    "response": "GaspipelineDatasets/NewGasFilteredResponseNNNoOHEMulti.csv"
}
save_directory = 'results/figures'
for dataset_name, csv_file in csv_files.items():
    plot_label_counts(dataset_name, csv_file, save_directory)