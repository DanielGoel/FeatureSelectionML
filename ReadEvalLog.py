import pandas as pd
import matplotlib.pyplot as plt
import os

# Load the dataset
df = pd.read_csv("results/metrics/evaluation_log.csv")

# Filter to only include rows where "Input File" is "NewGasFilteredFunctionMinMax.csv"
df = df[df["Input File"] == "NewGasFilteredAllMinMax.csv"]

# Handle missing feature selection values
df["Feature Selection"] = df["Feature Selection"].fillna("None")

# Create a directory for saving figures if it doesn't exist
os.makedirs("results/figures", exist_ok=True)

# ------------------- Generate the Last Graph: Remove Duplicate "None" Entries for the Same Model Type ------------------- #

# Define relevant feature selection methods
filtered_feature_selection = ["RFE", "SPFS", "None"]

# Filter F1 Scores >= 0.95 and ensure only selected Feature Selection methods are included
df_best_filtered = df[df["F1 Score"] >= 0.80]
df_best_filtered = df_best_filtered[df_best_filtered["Feature Selection"].isin(filtered_feature_selection)]

# Drop duplicate entries to ensure each model type appears only once
df_best_filtered = df_best_filtered.drop_duplicates(subset=["Rank Method", "Feature Selection", "Model Type"], keep="first")

# Create label column
df_best_filtered["Label"] = df_best_filtered.apply(
    lambda x: f"All Features, {x['Model Type']}" if x["Feature Selection"] == "None" 
    else f"{x['Rank Method']}, {x['Feature Selection']}, {x['Model Type']}", axis=1
)

# Ensure labels are unique
df_best_filtered = df_best_filtered.sort_values("F1 Score", ascending=False)

# Plot bar chart for best F1 score per Feature Selection Method
plt.figure(figsize=(12, 7))
bars = plt.bar(df_best_filtered["Label"], df_best_filtered["F1 Score"], color="royalblue")

# Add text labels on top of each bar
for i, bar in enumerate(bars):
    yval = bar.get_height()
    num_features = df_best_filtered.iloc[i]["Number of Features"]
    plt.text(bar.get_x() + bar.get_width()/2, yval, f"{yval:.4f} ({num_features})", 
             ha='center', va='bottom', fontsize=10, fontweight='bold', rotation=45)

# Formatting
plt.xlabel("Feature Selection Method")
plt.ylabel("Best F1 Score")
plt.title("Best F1 Scores: Rank Method, Feature Selection Method, Model Type")
plt.xticks(rotation=45, ha="right")
plt.ylim(0.80, 1.10)
plt.grid(axis="y", linestyle="--", alpha=0.6)

# Save and show
plt.savefig("results/figures/Best_F1_Score_Rank_Feature_Model2.png", dpi=300, bbox_inches="tight")
plt.show()

# ------------------- Generate the Second Graph: F1 Score vs. Number of Features ------------------- #

# Ensure "Number of Features" column is numeric
df["Number of Features"] = pd.to_numeric(df["Number of Features"], errors="coerce")

# Drop any rows where "Number of Features" is NaN
df = df.dropna(subset=["Number of Features"])

# Find the best F1 score for each unique number of features
df_best_f1 = df.loc[df.groupby("Number of Features")["F1 Score"].idxmax()]

# Sort by number of features
df_best_f1 = df_best_f1.sort_values("Number of Features")

# Create a directory for saving figures if it doesn't exist
os.makedirs("results/figures", exist_ok=True)

# Plot F1 Score vs. Number of Features
plt.figure(figsize=(12, 8))
plt.plot(df_best_f1["Number of Features"], df_best_f1["F1 Score"], marker="o", linestyle="-", color="orange", label="F1 Score")


# Formatting
plt.xticks(df_best_f1["Number of Features"])
plt.yticks([0.8, 0.85, 0.9, 0.95, 1.0])
plt.ylim(0.8, 1.0)
plt.xlabel("Number of Features")
plt.ylabel("F1 Score")
plt.title("Impact of Feature Reduction on Model Performance")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()

# Save and show
plt.savefig("results/figures/F1_Score_vs_Num_Features.png", dpi=300, bbox_inches="tight")
plt.show()
