import os
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from ranking import DataAnalysisRanker
from feature_selection import FeatureSelector

# List of CSV files to process
csv_files = {
    #"function2": "GaspipelineDatasets/NewGasFilteredFunctionMinMax_Remapped.csv",
    #"command": "GaspipelineDatasets/NewGasFilteredCommandMinMax_Remapped.csv",
    "all": "GaspipelineDatasets/NewGasFilteredAllMinMax.csv",
    #"response": "GaspipelineDatasets/NewGasFilteredResponseNNNoOHEMulti.csv"
}

ranking_methods = ["WFI-XGB"]
model_types = ["XGB"]
feature_selection_methods = ["RFE", "SPFS", "None"]

use_std_dev = use_abs_diff = use_skewness = use_kurtosis = True

# ✅ **Step 1: Split Each CSV File into 80/20 if not already split**
for dataset_name, input_file in csv_files.items():
    train_file = f"GaspipelineDatasets/{dataset_name}_train_80.csv"
    test_file = f"GaspipelineDatasets/{dataset_name}_test_20.csv"

    if not (os.path.exists(train_file) and os.path.exists(test_file)):
        df = pd.read_csv(input_file)
        target_column = "Label"
        
        if target_column not in df.columns:
            raise KeyError(f"⚠️ Target column '{target_column}' not found in dataset: {dataset_name}")

        X = df.drop(columns=[target_column])
        y = df[target_column]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        train_df = X_train.copy()
        train_df[target_column] = y_train
        test_df = X_test.copy()
        test_df[target_column] = y_test

        train_df.to_csv(train_file, index=False)
        test_df.to_csv(test_file, index=False)

# ✅ **Step 2: Process Each Dataset with Feature Selection**
for dataset_name in csv_files.keys():
    input_file = f"GaspipelineDatasets/{dataset_name}_train_80.csv"
    
    for ranking_method in ranking_methods:
        ranking_file = f'results/Rankings/Ranking_{ranking_method}_{dataset_name}.csv'
        
        ranker = DataAnalysisRanker(ranking_file, input_file, ranking_file)
        ranker.analyze(ranking_method, use_std_dev, use_abs_diff, use_skewness, use_kurtosis)
        ranker.save_results()
        
        rankings_df = pd.read_csv(ranking_file)
        ranked_features = rankings_df.iloc[:, 0].tolist()

        for model_type in model_types:
            for feature_selection_method in feature_selection_methods:
                selector = FeatureSelector(dataset_name, ranking_file, input_file, model_type)
                final_selected_features = selector.perform_feature_selection(feature_selection_method)
                
                print(f"Selected features for {dataset_name} ({ranking_method}, {model_type}, {feature_selection_method}):\n{final_selected_features}\n")
                
