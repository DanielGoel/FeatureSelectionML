import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

class DataAnalysisRanker:
    def __init__(self, ranking_file, input_file, output_file):
        self.ranking_file = ranking_file
        self.input_file = input_file
        self.output_file = output_file
        self.df = pd.read_csv(input_file)
        self.results_df = None

    def compute_standard_deviation(self):
        return self.df.std()

    def compute_abs_diff_mean_median(self):
        return np.abs(self.df.mean() - self.df.median())

    def compute_skewness(self):
        return self.df.apply(skew)

    def compute_kurtosis(self):
        return self.df.apply(kurtosis)

    def analyze(self, rank_method, use_std_dev, use_abs_diff, use_skewness, use_kurtosis):
        metrics = {}
        ranks = {}

        if rank_method == 'randomforest':
            feature_importances_df = self.wfi_RandomForest()
            feature_importances_df = feature_importances_df.set_index('Name')
            self.df = self.df[feature_importances_df.index]
        
        elif rank_method == 'xgboost':
            feature_importances_df = self.wfi_XGBoost()
            feature_importances_df = feature_importances_df.set_index('Name')
            self.df = self.df[feature_importances_df.index]
        
        else:
            if 'Label' in self.df.columns:
                self.df = self.df.drop(columns=['Label'])
            if use_std_dev:
                metrics['Standard Deviation'] = self.compute_standard_deviation()
                ranks['Standard Deviation Rank'] = metrics['Standard Deviation'].rank(ascending=True)

            if use_abs_diff:
                metrics['Absolute Difference'] = self.compute_abs_diff_mean_median()
                ranks['Absolute Difference Rank'] = metrics['Absolute Difference'].rank(ascending=True)

            if use_skewness:
                metrics['Skewness'] = self.compute_skewness()
                ranks['Skewness Rank'] = metrics['Skewness'].rank(ascending=False)

            if use_kurtosis:
                metrics['Kurtosis'] = self.compute_kurtosis()
                ranks['Kurtosis Rank'] = metrics['Kurtosis'].rank(ascending=True)

            # Combine all metrics and ranks into a single DataFrame
            OutputData = {**metrics, **ranks}
            self.results_df = pd.DataFrame({
                'Name': self.df.columns,
                **OutputData
            })

            if ranks:
                total_rank = sum(ranks.values())
                self.results_df['Total Rank'] = total_rank
                self.feature_importances_df = self.results_df.sort_values('Total Rank', ascending=True)
        self.feature_importances_df = self.feature_importances_df.reset_index()
        #print(feature_importances_df)
        self.feature_importances_df = self.feature_importances_df.set_index('Name')
        self.save_results()
    
    def wfi_RandomForest(self):

        target_column = 'Label'
        y = self.df[target_column]
        X = self.df.drop(columns=[target_column])
        print(X.columns)


        model = RandomForestClassifier(random_state=42)
        model.fit(X, y)

        self.feature_importances = model.feature_importances_
        self.feature_importances_df = pd.DataFrame({
            'Name': X.columns,
            'Random Forest Feature Importance': self.feature_importances
        }).sort_values(by='Random Forest Feature Importance', ascending=False)  # Sorting by importance
        return self.feature_importances_df
    
    def wfi_XGBoost(self):

        target_column = 'Label'
        y = self.df[target_column]
        X = self.df.drop(columns=[target_column])
        
        # Convert class labels to a sequential range
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)  # Converts [0, 3, 4, 5, 6, 7] → [0, 1, 2, 3, 4, 5]

        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        model.fit(X, y_encoded)  # Use encoded labels

        self.feature_importances = model.feature_importances_

        self.feature_importances_df = pd.DataFrame({
            'Name': X.columns,
            'XGBoost Feature Importance': self.feature_importances
        }).sort_values(by='XGBoost Feature Importance', ascending=False) 

        return self.feature_importances_df.sort_values(by='XGBoost Feature Importance', ascending=False)


    def save_results(self):
        df_to_save = self.feature_importances_df
        df_to_save.to_csv(self.ranking_file, index=True)
        print(f"Results saved to {self.ranking_file}")