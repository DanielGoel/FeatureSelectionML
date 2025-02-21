import pandas as pd

# Load the dataset
df = pd.read_csv("results/metrics/evaluation_log.csv")  # Update with your actual file path

# Ensure all rows are printed
pd.set_option("display.max_rows", None)  # Show all rows
pd.set_option("display.max_columns", None)  # Show all columns (if needed)
pd.set_option("display.expand_frame_repr", False)  # Prevent line wrapping

# Print the entire "Feature Selection" column
print(df["Feature Selection"])
df["Feature Selection"] = df["Feature Selection"].fillna("None")
print(df["Feature Selection"])