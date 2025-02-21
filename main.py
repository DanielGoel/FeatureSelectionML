from ranking import DataAnalysisRanker
from feature_selection import FeatureSelector
from model_testing import ModelEvaluator
import pandas as pd
import os

#def main():
#    input_file = "GaspipelineDatasets/NewGasFilteredFunctionMinMax.csv"
#    ranking_file = 'results/Rankings/Ranking_Function.csv'
#    model_dir = 'results/models/'
#
#
#    use_std_dev = True
#    use_abs_diff = True
#    use_skewness = True
#    use_kurtosis = True
#
#    for rank_method in ["SP"]:
#        if rank_method == "SP":
#            ranking_file = 'results/Rankings/Ranking_SP_Function.csv'
#        if rank_method == "randomforest":
#            ranking_file = 'results/Rankings/Ranking_RandomForest_Function.csv'
#        if rank_method == "xgboost":
#            ranking_file = 'results/Rankings/Ranking_XGBoost_Function.csv'
#        for model_type in ["randomforest", "xgboost"]:
#            for feature_selection_method in ["RFE", "SPFS", "None"]:
#                # Step 1: Rank features
#                ranker = DataAnalysisRanker(ranking_file, input_file, ranking_file)
#                ranker.analyze(rank_method, use_std_dev, use_abs_diff, use_skewness, use_kurtosis)
#                ranker.save_results()
#                # Step 2: Select features based on ranking
#                selector = FeatureSelector(ranking_file, input_file)
#                selector.perform_feature_selection(model_type, feature_selection_method)
#
#                if feature_selection_method != "None":
#                    model_filename = f"{model_type}_{feature_selection_method}.pkl"
#                else:
#                    model_filename = f"{model_type}_Baseline.pkl"  # Default model without feature selection
#                
#                model_path = os.path.join(model_dir, model_filename)
#
#                if not os.path.exists(model_path):
#                    print(f"⚠️ Warning: Model file {model_path} not found. Skipping this combination.")
#                    continue
#
#                for average_type in ["micro", "macro", "weighted"]:
#                    # Step 3: Train and evaluate the model, log results, and save the model
#                    evaluator = ModelEvaluator(input_file, average_type, selector.selected_features, model_type, feature_selection_method, model_path, rank_method)
#                    evaluator.train_and_evaluate()
#                    print(f"Model: {model_type}, Feature Selection: {feature_selection_method}, Average Type: {average_type}")
#                    print("--------------------------------------------------")

#def main():
#    input_file = "GaspipelineDatasets/NewGasFilteredFunctionMinMax.csv"
#    model_dir = 'results/models/'
#
#    use_std_dev = True
#    use_abs_diff = True
#    use_skewness = True
#    use_kurtosis = True
#
#    for rank_method in ["SP"]:
#        ranking_file = f'results/Rankings/Ranking_{rank_method}_Function.csv'
#        
#        # Step 1: Rank features
#        ranker = DataAnalysisRanker(ranking_file, input_file, ranking_file)
#        ranker.analyze(rank_method, use_std_dev, use_abs_diff, use_skewness, use_kurtosis)
#        ranker.save_results()
#
#        # Load ranked features
#        rankings_df = pd.read_csv(ranking_file)
#        print(f"ranked features: {rankings_df}")
#        ranked_features = rankings_df.iloc[:, 0].tolist()
#        print(f"Ranked Features: {ranked_features}")
#
#        for model_type in ["randomforest", "xgboost"]:
#            for num_remove in range(6):  # Remove 0 to 5 least important features
#                selected_features = ranked_features[:-num_remove] if num_remove > 0 else ranked_features
#                
#                for num_keep in range(1, 6):  # Keep only top 1 to 5 features
#                    limited_features = ranked_features[:num_keep]
#                    
#                    # Step 2: Select Features
#                    selector = FeatureSelector(ranking_file, input_file)
#                    selector.selected_features = selected_features if num_remove > 0 else limited_features
#                    
#                    feature_selection_method = f"remove_{num_remove}" if num_remove > 0 else f"keep_{num_keep}"
#                    model_filename = f"{model_type}_{feature_selection_method}.pkl"
#                    model_path = os.path.join(model_dir, model_filename)
#                    
#                    selector.train_and_save_model(selector.selected_features, model_type, model_path)
#                    
#                    
#                    for average_type in ["micro", "macro", "weighted"]:
#                        # Step 3: Train and Evaluate Model
#                        evaluator = ModelEvaluator(input_file, average_type, selector.selected_features, model_type, feature_selection_method, model_path, rank_method)
#                        print(f"features" , selector.selected_features)
#                        evaluator.train_and_evaluate()
#                        
#                        print(f"Model: {model_type}, Features: {feature_selection_method}, Average Type: {average_type}")
#                        print("--------------------------------------------------")

    #model_type = "randomforest"  # "xgboost" or "randomforest"
    #feature_selection_method = "none"  # "RFE" or "SPFS" or "None"
    #average_type = "weighted"  # "none", "micro", "macro", "weighted"
#
    ## Step 1: Rank features
    #ranker = DataAnalysisRanker(input_file, ranking_file)
    #ranker.analyze(use_std_dev, use_abs_diff, use_skewness, use_kurtosis)
    #ranker.save_results()
#
    ## Step 2: Select features based on ranking
    #selector = FeatureSelector(ranking_file, input_file)
    #selector.perform_feature_selection(model_type, feature_selection_method)
#
    ## Step 3: Train and evaluate the model, log results, and save the model
    #evaluator = ModelEvaluator(input_file, average_type,  selector.selected_features, model_type, feature_selection_method)
    #evaluator.train_and_evaluate()


#def main():
#    input_file = "GaspipelineDatasets/NewGasFilteredFunctionMinMax.csv"
#    model_dir = 'results/models/'
#
#    use_std_dev = True
#    use_abs_diff = True
#    use_skewness = True
#    use_kurtosis = True
#
#    for rank_method in ["SP"]:
#        ranking_file = f'results/Rankings/Ranking_{rank_method}_Function.csv'
#        
#        # Step 1: Rank features
#        ranker = DataAnalysisRanker(ranking_file, input_file, ranking_file)
#        ranker.analyze(rank_method, use_std_dev, use_abs_diff, use_skewness, use_kurtosis)
#        ranker.save_results()
#
#        # Load ranked features
#        rankings_df = pd.read_csv(ranking_file)
#        ranked_features = rankings_df.iloc[:, 0].tolist()
#        print(f"Ranked Features for {rank_method}: {ranked_features}")
#
#        for model_type in ["randomforest", "xgboost"]:
#            for feature_selection_method in ["RFE", "SPFS", "None"]:
#                # Step 2: Select Features
#                selector = FeatureSelector(ranking_file, input_file)
#                selector.perform_feature_selection(model_type, feature_selection_method)
#                selected_features = selector.selected_features
#
#                if feature_selection_method != "None":
#                    model_filename = f"{model_type}_{feature_selection_method}.pkl"
#                else:
#                    model_filename = f"{model_type}_Baseline.pkl"
#                
#                model_path = os.path.join(model_dir, model_filename)
#                
#                if not os.path.exists(model_path):
#                    print(f"⚠️ Warning: Model file {model_path} not found. Skipping this combination.")
#                    continue
#
#                for average_type in ["micro", "macro", "weighted"]:
#                    # Step 3: Train and Evaluate Model
#                    evaluator = ModelEvaluator(input_file, average_type, selected_features, model_type, feature_selection_method, model_path, rank_method)
#                    evaluator.train_and_evaluate()
#                    print(f"Model: {model_type}, Feature Selection: {feature_selection_method}, Average Type: {average_type}")
#                    print("--------------------------------------------------")
#
#            for num_remove in range(6):  # Remove 0 to 5 least important features
#                modified_features = ranked_features[:-num_remove] if num_remove > 0 else ranked_features
#                
#                for num_keep in range(1, 6):  # Keep only top 1 to 5 features
#                    limited_features = ranked_features[:num_keep]
#                    feature_selection_method = f"remove_{num_remove}" if num_remove > 0 else f"keep_{num_keep}"
#                    
#                    model_filename = f"{model_type}_{feature_selection_method}.pkl"
#                    model_path = os.path.join(model_dir, model_filename)
#                    
#                    selector = FeatureSelector(ranking_file, input_file)
#                    selector.selected_features = modified_features if num_remove > 0 else limited_features
#                    selector.train_and_save_model(selector.selected_features, model_type, model_path)
#                    
#                    for average_type in ["micro", "macro", "weighted"]:
#                        evaluator = ModelEvaluator(input_file, average_type, selector.selected_features, model_type, feature_selection_method, model_path, rank_method)
#                        evaluator.train_and_evaluate()
#                        print(f"Model: {model_type}, Features: {feature_selection_method}, Average Type: {average_type}")
#                        print("--------------------------------------------------")
#
#if __name__ == "__main__":
#    main()


import pandas as pd
import os
from ranking import DataAnalysisRanker
from feature_selection import FeatureSelector
from model_testing import ModelEvaluator

def main():
    input_file = "GaspipelineDatasets/NewGasFilteredFunctionMinMax.csv"
    model_dir = 'results/models/'

    use_std_dev = True
    use_abs_diff = True
    use_skewness = True
    use_kurtosis = True

    for rank_method in ["SP", "randomforest", "xgboost"]:
        ranking_file = f'results/Rankings/Ranking_{rank_method}_Function.csv'
        
        # Step 1: Rank features
        ranker = DataAnalysisRanker(ranking_file, input_file, ranking_file)
        ranker.analyze(rank_method, use_std_dev, use_abs_diff, use_skewness, use_kurtosis)
        ranker.save_results()

        # Load ranked features
        rankings_df = pd.read_csv(ranking_file)
        ranked_features = rankings_df.iloc[:, 0].tolist()
        print(f"Ranked Features for {rank_method}: {ranked_features}")

        for model_type in ["randomforest", "xgboost"]:
            for num_features in range(6, 10):  # Only select feature counts between 6 and 9
                modified_features = ranked_features[:num_features]  # Keep only top-ranked features

                feature_selection_method = f"keep_{num_features}"
                model_filename = f"{model_type}_{feature_selection_method}.pkl"
                model_path = os.path.join(model_dir, model_filename)

                selector = FeatureSelector(ranking_file, input_file)
                selector.selected_features = modified_features
                selector.train_and_save_model(selector.selected_features, model_type, model_path)

                for average_type in ["micro", "macro", "weighted"]:
                    evaluator = ModelEvaluator(input_file, average_type, selector.selected_features, model_type, feature_selection_method, model_path, rank_method)
                    evaluator.train_and_evaluate()
                    print(f"Model: {model_type}, Features: {feature_selection_method}, Average Type: {average_type}")
                    print("--------------------------------------------------")

if __name__ == "__main__":
    main()