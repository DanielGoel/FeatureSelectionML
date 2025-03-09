import os
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from ranking import DataAnalysisRanker
from feature_selection import FeatureSelector
from model_testing import ModelEvaluator

# List of CSV files to process
csv_files = {
    "function": "GaspipelineDatasets/NewGasFilteredFunctionMinMax_Remapped.csv",
    "command": "GaspipelineDatasets/NewGasFilteredCommandMinMax_Remapped.csv",
    "all": "GaspipelineDatasets/NewGasFilteredAllMinMax.csv",
    "response": "GaspipelineDatasets/NewGasFilteredResponseNNNoOHEMulti.csv"
}

metrics_log_file = "results/metrics/Evaluation_log_2.csv"
model_dir = "results/models/"
ranking_methods = ["WFI-RF", "SP", "WFI-XGB"]
model_types = ["XGB", "RF"]
average_types = ["micro", "macro", "weighted"]
feature_selection_methods = ["RFE", "SPFS", "None"]

use_std_dev = use_abs_diff = use_skewness = use_kurtosis = True

# ✅ **Step 2: Process Each Dataset with Feature Selection and Model Training**
for dataset_name, csv_file in csv_files.items():  # Now use split datasets 
    for ranking_method in ranking_methods:
        input_file = csv_file
        ranking_file = f'results/Rankings/Ranking_{ranking_method}_{dataset_name}.csv'
        
        # Step 1: Rank Features (with timing)
        start_time_ranking = time.time()
        ranker = DataAnalysisRanker(ranking_file, input_file, ranking_file)
        ranker.analyze(ranking_method, use_std_dev, use_abs_diff, use_skewness, use_kurtosis)
        ranker.save_results()
        ranking_time_taken = time.time() - start_time_ranking
        
        # Load ranked features
        rankings_df = pd.read_csv(ranking_file)
        ranked_features = rankings_df.iloc[:, 0].tolist()
        total_features = len(ranked_features)
        print(f"Processing dataset: {dataset_name} with {total_features} features using {ranking_method}.")
        
        # Standard evaluation for each model type and feature selection method
        for model_type in model_types:
            for feature_selection_method in feature_selection_methods:
                # Step 2: Feature Selection (with timing)
                start_time_selection = time.time()
                selector = FeatureSelector(dataset_name, ranking_file, input_file)
                final_model, selected_features = selector.perform_feature_selection(model_type, feature_selection_method)
                selection_time_taken = time.time() - start_time_selection
                
                model_filename = f"{dataset_name}_{model_type}_{feature_selection_method}.pkl"
                model_path = os.path.join(model_dir, model_filename)
                
                for average_type in ["micro", "macro", "weighted"]:
                    evaluator = ModelEvaluator(dataset_name, input_file, average_type, selected_features,
                                               model_type, feature_selection_method, model_path, ranking_method, metrics_log_file)
                    evaluation_time_taken = evaluator.train_and_evaluate(ranking_time_taken, selection_time_taken)
                    total_time_taken = ranking_time_taken + selection_time_taken + evaluation_time_taken
                    print(f"Total time for {dataset_name} using {ranking_method}, {model_type}, {feature_selection_method}, {average_type}: {total_time_taken:.2f} seconds")
            
            # Step 3: Iterative Feature Removal (removing 1 to total_features - 1 features)
            for num_remove in range(1, total_features):
                # Use all but the last 'num_remove' features from the ranked list
                selected_features = ranked_features[:-num_remove]
                feature_selection_method = f"rem_{num_remove}"
                model_filename = f"{dataset_name}_{model_type}_{feature_selection_method}.pkl"
                model_path = os.path.join(model_dir, model_filename)
                
                # Manually set the selected features and save the model
                selector.selected_features = selected_features
                final_model = selector.train_and_save_model(selected_features, model_type, model_path)
                
                for average_type in average_types:
                    evaluator = ModelEvaluator(dataset_name, input_file, average_type, selected_features,
                                               model_type, feature_selection_method, model_path, ranking_method, metrics_log_file)
                    evaluator.train_and_evaluate(ranking_time_taken, selection_time_taken)
                    
        print(f"Completed processing for {dataset_name} dataset.\n")
