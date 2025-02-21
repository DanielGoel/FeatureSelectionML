import pandas as pd
import os
import pickle
import time
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix

class ModelEvaluator:
    def __init__(self, input_file, average_type, selected_features, model_type, feature_selection_method, model_path, rank_method):
        self.rank_method = rank_method
        self.input_file = input_file
        self.feature_selection_method = feature_selection_method
        self.selected_features = selected_features
        self.average_type = average_type
        self.model_type = model_type
        self.model_path = model_path
        self.df = pd.read_csv(input_file)
        self.target_column = "Label"
        self.metrics_log_file = "results/metrics/evaluation_log.csv"
        self.removed_features = list(set(self.df.columns) - set(self.selected_features)- {self.target_column})
        
    def load_model(self):
        """Load pre-trained model from a file"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at {self.model_path}")

        with open(self.model_path, 'rb') as model_file:
            model = pickle.load(model_file)

        return model

    def evaluate_model(self, y_true, y_pred):
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average=self.average_type, zero_division=0)
        recall = recall_score(y_true, y_pred, average=self.average_type, zero_division=0)
        f1 = f1_score(y_true, y_pred, average=self.average_type)

        cm = confusion_matrix(y_true, y_pred)
        fpr = cm[0][1] / cm[0].sum() if cm[0].sum() > 0 else 0
        fnr = cm[1][0] / cm[1].sum() if cm[1].sum() > 0 else 0

        return {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'FPR': fpr,
            'FNR': fnr
        }

    def train_and_evaluate(self):
        """Perform evaluation using 10-Fold Cross-Validation"""
        start_time = time.time()
        
        if self.feature_selection_method == "none":
            self.selected_features = self.df.columns.tolist()
            self.selected_features.remove(self.target_column)  # Ensure target is removed
            

        X = self.df[self.selected_features]
        y = self.df[self.target_column]

        # Ensure y is encoded correctly for XGBoost
        if "xgboost" in self.model_type.lower():
            y = y.astype('category').cat.codes  # Convert labels to numerical format

        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        model = self.load_model()
        metrics_data = []

        print("Performing 10-Fold Cross-Validation...")
        
        for fold, (train_index, val_index) in enumerate(kf.split(X, y), start=1):
            X_train, X_val = X.iloc[train_index], X.iloc[val_index]
            y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        
            # Ensure feature names match the ones used during training
            if hasattr(model, "feature_names_in_"):
                X_val = X_val[model.feature_names_in_]

            # Use pre-trained model for prediction
            y_pred = model.predict(X_val)

            metrics = self.evaluate_model(y_val, y_pred)
            metrics_data.append(metrics)
            print(f"Fold {fold}: F1 Score = {metrics['F1 Score']:.4f}")

        # Compute average metrics
        avg_metrics = pd.DataFrame(metrics_data).mean(numeric_only=True)
        total_time_taken = time.time() - start_time

        # Create a log entry
        log_entry = pd.DataFrame({
            'Input File': [os.path.basename(self.input_file)],
            'Rank Method': [self.rank_method],
            'Feature Selection': [self.feature_selection_method],
            'Model Type': [self.model_type],
            'Average Type': [self.average_type],
            'Number of Features': [len(self.selected_features)],
            'Removed Features': [", ".join(self.removed_features)],
            'Accuracy': [avg_metrics['Accuracy']],
            'Precision': [avg_metrics['Precision']],
            'Recall': [avg_metrics['Recall']],
            'F1 Score': [avg_metrics['F1 Score']],
            'FPR': [avg_metrics['FPR']],
            'FNR': [avg_metrics['FNR']],
            'Time Taken': [total_time_taken]
        })

        # Append results to the log file
        os.makedirs(os.path.dirname(self.metrics_log_file), exist_ok=True)
        if not os.path.exists(self.metrics_log_file):
            log_entry.to_csv(self.metrics_log_file, index=False)
        else:
            log_entry.to_csv(self.metrics_log_file, mode='a', header=False, index=False)

        print(f"✅ Results appended to {self.metrics_log_file}")
