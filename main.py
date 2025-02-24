import os
import pandas as pd
from ranking import DataAnalysisRanker
from feature_selection import FeatureSelector
from model_testing import ModelEvaluator

# List of CSV files to process
csv_files = {
    "function": "GaspipelineDatasets/NewGasFilteredFunctionMinMax.csv",
    
}

model_dir = 'results/models/'
ranking_methods = ["SP"]  # Added WFI-RF and WFI-XGB
use_std_dev = use_abs_diff = use_skewness = use_kurtosis = True

for dataset_name, input_file in csv_files.items():
    for ranking_method in ranking_methods:
        ranking_file = f'results/Rankings/Ranking_{ranking_method}_{dataset_name}.csv'
        
        # Step 1: Rank features
        ranker = DataAnalysisRanker(ranking_file, input_file, ranking_file)
        ranker.analyze(ranking_method, use_std_dev, use_abs_diff, use_skewness, use_kurtosis)
        ranker.save_results()
        
        # Load ranked features
        rankings_df = pd.read_csv(ranking_file)
        ranked_features = rankings_df.iloc[:, 0].tolist()
        total_features = len(ranked_features)
        print(f"Processing dataset: {dataset_name} with {total_features} features using {ranking_method}.")
        
        for model_type in ["RF"]:
            for feature_selection_method in ["RFE", "SPFS", "None"]:
                # Step 2: Select Features
                selector = FeatureSelector(dataset_name, ranking_file, input_file)
                selector.perform_feature_selection(model_type, feature_selection_method)
                
                model_filename = f"{dataset_name}_{model_type}_{feature_selection_method}.pkl"
                model_path = os.path.join(model_dir, model_filename)
                
                for average_type in ["micro", "macro", "weighted"]:
                    evaluator = ModelEvaluator(input_file, average_type, selector.selected_features, model_type, feature_selection_method, model_path, ranking_method)
                    evaluator.train_and_evaluate()
                    
            # Step 3: Remove 1 to (total_features - 1) features iteratively
            for num_remove in range(1, total_features):
                selected_features = ranked_features[:-num_remove]
                feature_selection_method = f"rem_{num_remove}"
                model_filename = f"{model_type}_{feature_selection_method}.pkl"
                model_path = os.path.join(model_dir, model_filename)
                
                selector.selected_features = selected_features
                selector.train_and_save_model(selected_features, model_type, model_path)
                
                for average_type in ["micro", "macro", "weighted"]:
                    evaluator = ModelEvaluator(input_file, average_type, selected_features, model_type, feature_selection_method, model_path, ranking_method)
                    evaluator.train_and_evaluate()
    
    print(f"Completed processing for {dataset_name} dataset.\n")