import pandas as pd
import matplotlib.pyplot as plt
import os

# Load the dataset
df = pd.read_csv("results/metrics/evaluation_log.csv")


df["Feature Selection"] = df["Feature Selection"].fillna("None")

# Create a directory for saving figures if it doesn't exist
os.makedirs("results/figures", exist_ok=True)


# ----------------------------------- First Graph: Best F1 Score per Rank Method & Model Type ----------------------------------- #

# Map "randomforest" and "xgboost" to "WFI RandomForest" and "WFI XGBoost"
df["Rank Method"] = df["Rank Method"].replace({"randomforest": "WFI RandomForest", "xgboost": "WFI XGBoost"})

# Get best F1 Score for each combination of Rank Method and Model Type
df_best_ranking_model = df.loc[df.groupby(["Rank Method", "Model Type"])["F1 Score"].idxmax(), 
                               ["Rank Method", "Model Type", "F1 Score"]]

# Sort for better visualization
df_best_ranking_model = df_best_ranking_model[df_best_ranking_model["F1 Score"] > 0.95]
df_best_ranking_model = df_best_ranking_model.sort_values("F1 Score", ascending=False)

# Plot bar chart for best F1 score per Ranking Method and Model Type
plt.figure(figsize=(12, 6))
bars = plt.bar(df_best_ranking_model.apply(lambda x: f"{x['Rank Method']} ({x['Model Type']})", axis=1), 
               df_best_ranking_model["F1 Score"], color="royalblue")

# Add text labels on top of each bar
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, f"{yval:.8f}", ha='center', va='bottom', fontsize=10, fontweight='bold', rotation=90)

# Formatting
plt.xlabel("Ranking Method & Model Type")
plt.ylabel("Best F1 Score")
plt.title("Best F1 Score for Each Ranking Method and Model Type (F1 Score > 0.95)")
plt.xticks(rotation=45, ha="right")
plt.ylim(0.98, 1.0)
plt.grid(axis="y", linestyle="--", alpha=0.6)

# Save and show
plt.savefig("results/figures/best_f1_ranking_model_95+.png", dpi=300, bbox_inches="tight")
plt.show()

# ----------------------------------- Second Graph: Best F1 Score per Rank Method, Model Type & Feature Selection ----------------------------------- #

# Get best F1 Score for each combination of Rank Method, Model Type, and Feature Selection
df_best_all = df.loc[df.groupby(["Rank Method", "Model Type", "Feature Selection", "Number of Features"])["F1 Score"].idxmax(), 
                     ["Rank Method", "Model Type", "Feature Selection", "F1 Score", "Number of Features"]]

# Sort for better visualization
df_best_all = df_best_all[df_best_all["F1 Score"] > 0.98]  # Filter F1 Scores > 98%
df_best_all = df_best_all.sort_values("F1 Score", ascending=False)  # Sort descending

# Plot bar chart for best F1 score per Ranking Method, Model Type, and Feature Selection
plt.figure(figsize=(14, 6))
bars = plt.bar(df_best_all.apply(lambda x: f"{x['Rank Method']}, {x['Feature Selection']}, {x['Model Type']}", axis=1), 
               df_best_all["F1 Score"], color="royalblue")

# Add text labels on top of each bar
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, f"{yval:.8f}", ha='center', va='bottom', fontsize=10, fontweight='bold', rotation=90)

# Formatting
plt.xlabel("Ranking Method, Model Type & Feature Selection")
plt.ylabel("Best F1 Score")
plt.title("Best F1 Score for Each Combination of Ranking Method, Model Type, and Feature Selection (F1 Score > 0.98)")
plt.xticks(rotation=90, ha="right")
plt.ylim(0.98, 1.0)
plt.grid(axis="y", linestyle="--", alpha=0.6)

# Save and show
plt.savefig("results/figures/best_f1_ranking_model_feature_selection_95+.png", dpi=300, bbox_inches="tight")
plt.show()

## ----------------------------------- Third Graph: Ordered by Feature Selection Methods ----------------------------------- #
#
## Define a custom order for feature selection methods
#feature_selection_order = ["RFE", "SPFS", "None", "remove_1", "remove_2"]
#
## Filter the best performing F1 Scores for these feature selection methods
#df_best_feature_selection = df_best_all[df_best_all["F1 Score"] >= 0.95]
#df_best_feature_selection = df_best_feature_selection[df_best_feature_selection["Feature Selection"].isin(feature_selection_order)]
#
#
## Sort by feature selection order
#df_best_feature_selection["Feature Selection"] = pd.Categorical(df_best_feature_selection["Feature Selection"], 
#                                                                categories=feature_selection_order, ordered=True)
#df_best_feature_selection = df_best_feature_selection.sort_values("F1 Score", ascending=False)
#
## Plot bar chart for best F1 score per Feature Selection Method
#plt.figure(figsize=(12, 6))
#bars = plt.bar(df_best_feature_selection.apply(lambda x: f"{x['Rank Method']}, {x['Feature Selection']}, {x['Model Type']}", axis=1), 
#               df_best_feature_selection["F1 Score"], color="royalblue")
#
## Add text labels on top of each bar
#for bar in bars:
#    yval = bar.get_height()
#    plt.text(bar.get_x() + bar.get_width()/2, yval, f"{yval:.4f}", ha='center', va='bottom', fontsize=10, fontweight='bold', rotation=90)
#
## Formatting
#plt.xlabel("Feature Selection Method")
#plt.ylabel("Best F1 Score")
#plt.title("Best F1 Score Ordered by Feature Selection Method (F1 Score > 0.95)")
#plt.xticks(rotation=45, ha="right")
#plt.ylim(0.95, 1.0)
#plt.grid(axis="y", linestyle="--", alpha=0.6)
#
## Save and show
#plt.savefig("results/figures/best_f1_feature_selection_ordered_95+.png", dpi=300, bbox_inches="tight")
#plt.show()
#
## ----------------------------------- Fourth Graph: Feature Selection Excluding "remove_1" and "remove_2" ----------------------------------- #
#
## Define the feature selection methods to include
#filtered_feature_selection = ["RFE", "SPFS", "None"]
#
## Filter the best-performing F1 Scores for the specified feature selection methods
#df_best_filtered = df_best_all[df_best_all["F1 Score"] >= 0.95]
#df_best_filtered = df_best_filtered[df_best_filtered["Feature Selection"].isin(filtered_feature_selection)]
#
## Sort by feature selection order
#
#df_best_filtered["Feature Selection"] = pd.Categorical(df_best_filtered["Feature Selection"], 
#                                                       categories=filtered_feature_selection, ordered=True)
#df_best_filtered = df_best_filtered.sort_values("F1 Score", ascending=False)
#
## Plot bar chart for best F1 score per Feature Selection Method (excluding "remove_1" and "remove_2")
#plt.figure(figsize=(12, 6))
#bars = plt.bar(df_best_filtered.apply(lambda x: f" {x['Rank Method']}, {x['Feature Selection']}, {x['Model Type']}", axis=1), 
#               df_best_filtered["F1 Score"], color="royalblue")
#
## Add text labels on top of each bar
#for bar in bars:
#    yval = bar.get_height()
#    plt.text(bar.get_x() + bar.get_width()/2, yval, f"{yval:.8f}", ha='center', va='bottom', fontsize=10, fontweight='bold', rotation=56)
#
## Formatting
#plt.xlabel("Feature Selection Method")
#plt.ylabel("Best F1 Score")
#plt.title("Best F1 Score for Selected Feature Selection Methods (Excluding all 'remove_x) (F1 Score > 0.95)")
#plt.xticks(rotation=45, ha="right")
#plt.ylim(0.98, 1.0)
#plt.grid(axis="y", linestyle="--", alpha=0.6)
#
## Save and show
#plt.savefig("results/figures/best_f1_feature_selection_filtered_95+.png", dpi=300, bbox_inches="tight")
#plt.show()
#

# ------------------- Fifth Graph: Remove Duplicate "None" Entries for the Same Model Type ------------------- #

# Define relevant feature selection methods
filtered_feature_selection = ["RFE", "SPFS", "None"]

# Filter F1 Scores >= 0.95 and ensure only selected Feature Selection methods are included
df_best_filtered = df_best_all[df_best_all["F1 Score"] >= 0.95]
df_best_filtered = df_best_filtered[df_best_filtered["Feature Selection"].isin(filtered_feature_selection)]
print(df_best_filtered.columns.tolist())

# Separate "None" rows and drop duplicates based only on "Model Type"
df_none_filtered = df_best_filtered[df_best_filtered["Feature Selection"] == "None"].drop_duplicates(subset=["Model Type"], keep="first")

# Keep all other feature selection methods
df_other_filtered = df_best_filtered[df_best_filtered["Feature Selection"] != "None"]

# Combine the filtered data
df_best_filtered = pd.concat([df_none_filtered, df_other_filtered])
df_best_filtered = df_best_filtered.sort_values("F1 Score", ascending=False)

df_best_filtered["Label"] = df_best_filtered.apply(
    #lambda x: f"{x['Feature Selection']}, {x['Model Type']}" if x["Feature Selection"] == "None" 
    lambda x: f"All Features, {x['Model Type']}" if x["Feature Selection"] == "None" 
    else f"{x['Rank Method']}, {x['Feature Selection']}, {x['Model Type']}", axis=1
)


# Plot bar chart for best F1 score per Feature Selection Method
plt.figure(figsize=(12, 6))
bars = plt.bar(df_best_filtered["Label"], df_best_filtered["F1 Score"], color="royalblue")

# Add text labels on top of each bar
for i, bar in enumerate(bars):
    yval = bar.get_height()
    num_features = df_best_filtered.iloc[i]["Number of Features"]
    plt.text(bar.get_x() + bar.get_width()/2, yval, f"{yval:.4f} ({num_features})", 
             ha='center', va='bottom', fontsize=10, fontweight='bold', rotation=56)
# Formatting
plt.xlabel("Feature Selection Method")
plt.ylabel("Best F1 Score")
plt.title("Best F1 Scores: Rank Method, Feature Selection Method, Model Type")
plt.xticks(rotation=45, ha="right")
plt.ylim(0.98, 1.0)
plt.grid(axis="y", linestyle="--", alpha=0.6)

# Save and show
plt.savefig("results/figures/Best_F1_Score_Rank_Feature_Model2.png", dpi=300, bbox_inches="tight")
plt.show()