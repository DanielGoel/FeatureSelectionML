import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns

# Load the dataset
df = pd.read_csv("results/metrics/evaluation_log_3.csv")
filenames = ['function', 'response', 'all', 'command']

for file in filenames:
    # Create a filtered copy for this file
    df_subset = df[df["Input File"] == file].copy()

    # If there are no matching rows, skip this iteration
    if df_subset.empty:
        print(f"No data found for '{file}', skipping...")
        continue

    # Handle missing feature selection values
    df_subset["Feature Selection"] = df_subset["Feature Selection"].fillna("None")

    # Create a directory for saving figures if it doesn't exist
    os.makedirs("results/figures", exist_ok=True)

    # ------------------- Generate Bar Chart ------------------- #

    # Define relevant feature selection methods
    filtered_feature_selection = ["RFE", "SPFS", "None"]

    # Filter F1 Scores >= 0.80 and ensure only selected Feature Selection methods are included
    df_best_filtered = df_subset[df_subset["F1 Score"] >= 0.80]
    df_best_filtered = df_best_filtered[df_best_filtered["Feature Selection"].isin(filtered_feature_selection)]

    # Drop duplicate entries to ensure each model type appears only once
    df_best_filtered = df_best_filtered.drop_duplicates(subset=["Rank Method", "Feature Selection", "Model Type"], keep="first")

    # Debugging: Print the filtered data
    print(f"Filtered Data for Bar Chart ({file}):")
    print(df_best_filtered[["Rank Method", "Feature Selection", "Model Type", "F1 Score", "Number of Features"]])

    # Check if the dataframe is empty before plotting
    if not df_best_filtered.empty:
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
        plt.title(f"Best F1 Scores for {file}")
        plt.xticks(rotation=45, ha="right")
        plt.ylim(0.955, .985)
        plt.grid(axis="y", linestyle="--", alpha=0.6)

        # Save and show
        plt.savefig(f"results/figures/Best_F1_Score_Rank_Feature_Model_{file}.png", dpi=300, bbox_inches="tight")
    else:
        print(f"No data available for the bar chart ({file})!")

    # ------------------- Feature Reduction Graph ------------------- #

    # Extract relevant data for feature reduction plot
    df_feature_reduction = df_subset[df_subset["Feature Selection"] != "None"]

    # Check if "Number of Features" exists
    if "Number of Features" in df_feature_reduction.columns:
        df_feature_reduction = df_feature_reduction.groupby("Number of Features")["F1 Score"].max().reset_index()

        # Debugging: Print the filtered data
        print(f"Filtered Data for Feature Reduction Graph ({file}):")
        print(df_feature_reduction)

        # Check if the dataframe is empty before plotting
        if not df_feature_reduction.empty:
            # Plot the impact of feature reduction on model performance
            plt.figure(figsize=(10, 6))
            plt.plot(df_feature_reduction["Number of Features"], df_feature_reduction["F1 Score"], 
                    marker='o', linestyle='-', color='orange', label="F1 Score")

            # Formatting
            plt.xlabel("Number of Features")
            plt.ylabel("F1 Score")
            plt.title(f"Impact of Feature Reduction on {file}")
            plt.legend()
            plt.grid(True, linestyle="--", alpha=0.6)

            # Ensure all integers appear on the x-axis
            min_features = df_feature_reduction["Number of Features"].min()
            max_features = df_feature_reduction["Number of Features"].max()
            plt.xticks(range(min_features, max_features + 1))  # Set x-axis ticks to all integers

            # Save and show
            plt.savefig(f"results/figures/Feature_Reduction_Impact_{file}.png", dpi=300, bbox_inches="tight")
        else:
            print(f"No data available for the feature reduction graph ({file})!")
    else:
        print(f"Column 'Number of Features' not found in the dataset for {file}!")



    # Extract relevant data
    df_speed_vs_f1 = df_subset[["Model Type", "F1 Score", "Evaluation Time", "Number of Features"]].dropna()

    # Check if necessary columns exist
    if not df_speed_vs_f1.empty:
        plt.figure(figsize=(10, 6))

        # Create scatter plot with bubble size as number of features
        sns.scatterplot(data=df_speed_vs_f1, x="F1 Score", y="Evaluation Time", 
                        size="Number of Features", hue="Model Type", palette="viridis", alpha=0.7, edgecolor="black")

        # Formatting
        plt.xlabel("F1 Score")
        plt.ylabel("Evaluation Time (s)")
        plt.title("Model Speed vs Best Performance (Bubble Size = Features)")
        plt.legend(title="Model Type", bbox_to_anchor=(1.05, 1), loc='upper left')

        # Save and show
        plt.savefig(f"results/figures/Model_Speed_vs_Performance_Bubble_{file}.png", dpi=300, bbox_inches="tight")
    else:
        print("No data available for the speed vs performance graph!")
