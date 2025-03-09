import pandas as pd
import os
import pickle
import time
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier

class ModelEvaluator:
    def __init__(self, og_file, input_file, average_type, selected_features, model_type, 
                 feature_selection_method, model_path, rank_method, metrics_log_file):
        self.og_file = og_file
        self.rank_method = rank_method
        self.input_file = input_file
        self.feature_selection_method = feature_selection_method
        self.selected_features = selected_features
        self.average_type = average_type
        self.model_type = model_type
        self.model_path = model_path
        self.df = pd.read_csv(input_file)
        self.target_column = "Label"
        self.metrics_log_file = metrics_log_file
        self.removed_features = list(set(self.df.columns) - set(self.selected_features) - {self.target_column})

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

    def train_and_evaluate(self, ranking_time_taken, selection_time_taken):
        """Train on 80%, Apply K-Fold CV, Save Best Model, Then Test on 20%"""
        start_time_evaluation = time.time()

        if self.feature_selection_method == "none":
            self.selected_features = self.df.columns.tolist()
            self.selected_features.remove(self.target_column)  # Ensure target is removed
            
        X = self.df[self.selected_features]
        y = self.df[self.target_column]

        # Encode labels for XGBoost
        if "xgboost" in self.model_type.lower():
            y = y.astype('category').cat.codes

        # **Step 1: Split dataset into 80% training, 20% final test**
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # **Step 2: Apply K-Fold CV on the 80% training set**
        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        best_f1_score = 0
        best_model = None

        print("Performing 10-Fold Cross-Validation on 80% training data...")
        for fold, (train_index, val_index) in enumerate(kf.split(X_train, y_train), start=1):
            X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
            y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

            # Train either XGBoost or Random Forest
            if "xgboost" in self.model_type.lower():
                model = xgb.XGBClassifier(seed=42)
            else:
                model = RandomForestClassifier(random_state=42)

            model.fit(X_train_fold, y_train_fold)
            y_pred_fold = model.predict(X_val_fold)

            metrics = self.evaluate_model(y_val_fold, y_pred_fold)
            f1 = metrics['F1 Score']

            print(f"Fold {fold}: F1 Score = {f1:.4f}")

            # Save the best model based on F1 Score
            if f1 > best_f1_score:
                best_f1_score = f1
                best_model = model  # Keep the best-performing model

        print(f"\n✅ Best Model Selected with F1 Score: {best_f1_score:.4f}")

        # **Step 3: Test the Best Model on 20% Final Test Data**
        y_test_pred = best_model.predict(X_test)
        final_test_metrics = self.evaluate_model(y_test, y_test_pred)

        evaluation_time_taken = time.time() - start_time_evaluation
        total_time_taken = ranking_time_taken + selection_time_taken + evaluation_time_taken

        # **Step 4: Log Results**
        log_entry = pd.DataFrame({
            'Input File': [os.path.basename(self.og_file)],
            'Rank Method': [self.rank_method],
            'Feature Selection': [self.feature_selection_method],
            'Model Type': [self.model_type],
            'Average Type': [self.average_type],
            'Number of Features': [len(self.selected_features)],
            'Removed Features': [", ".join(self.removed_features)],
            'Best K-Fold F1 Score': [best_f1_score],
            'Final Test Accuracy': [final_test_metrics['Accuracy']],
            'Final Test Precision': [final_test_metrics['Precision']],
            'Final Test Recall': [final_test_metrics['Recall']],
            'Final Test F1 Score': [final_test_metrics['F1 Score']],
            'Final Test FPR': [final_test_metrics['FPR']],
            'Final Test FNR': [final_test_metrics['FNR']],
            'Ranking Time Taken': [ranking_time_taken],
            'Feature Selection Time Taken': [selection_time_taken],
            'Evaluation Time Taken': [evaluation_time_taken],  
            'Total Time Taken': [total_time_taken] 
        })

        # Append results to the log file
        os.makedirs(os.path.dirname(self.metrics_log_file), exist_ok=True)
        if not os.path.exists(self.metrics_log_file):
            log_entry.to_csv(self.metrics_log_file, index=False)
        else:
            log_entry.to_csv(self.metrics_log_file, mode='a', header=False, index=False)

        print(f"✅ Results appended to {self.metrics_log_file}")

        # **Step 5: Save the Best Model**
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        with open(self.model_path, 'wb') as model_file:
            pickle.dump(best_model, model_file)

        print(f"✅ Best trained model saved to {self.model_path}")

        return evaluation_time_taken  # Return evaluation time for tracking
