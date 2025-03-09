import pandas as pd
import os
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import time
from sklearn.metrics import f1_score
import pickle

class FeatureSelector:
    def __init__(self, dataset_name, ranking_file, input_file):
        self.ranking_file = ranking_file
        self.input_file = input_file
        self.dataset_name = dataset_name
        self.df = pd.read_csv(input_file)
        self.selected_features = []
        self.metrics_dir = "results/metrics/"
        self.model_dir = "results/models/"
        os.makedirs(self.metrics_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)

        # Load feature ranking order
        self.ranked_features = self.load_ranked_features()

    def load_ranked_features(self):
        rankings_df = pd.read_csv(self.ranking_file)
        ranked_features = rankings_df.iloc[:, 0].tolist()  # First column contains feature names
        return [feature for feature in ranked_features if feature in self.df.columns]  # Keep only valid features

    def perform_feature_selection(self, model_type, method):
        if method == "RFE":
            return self.perform_rfe(model_type)  
        elif method == "SPFS":
            return self.perform_spfs(model_type) 
        else:
            print("⚠️ Skipped Feature Selection. Using all features.")
            return self.perform_baseline(model_type) 



    def perform_rfe(self, model_type):
        target_column = 'Label'
        
        if target_column not in self.df.columns:
            raise KeyError(f"Column '{target_column}' not found in dataset!")

        y = self.df[target_column]
        X = self.df.drop(columns=[target_column])  # Use ranked feature order

        # Select Model Type
        if model_type == 'XGB':
            y = y.astype('category').cat.codes  
            model = xgb.XGBClassifier(seed=42)
        else:
            model = RandomForestClassifier(random_state=42)

        selected_features = self.ranked_features.copy()  # Start with all ranked features
        best_f1_score = 0  

        print("Starting Recursive Feature Elimination...")

        # Evaluate the model with all features first
        X_train, X_test, y_train, y_test = train_test_split(X[selected_features], y, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        best_f1_score = self.get_f1_score(y_test, y_pred)['F1 Score']
        print(f"Initial F1 Score with all features: {best_f1_score:.4f}")

        for feature in reversed(self.ranked_features):
            if feature in selected_features:
                temp_features = selected_features.copy()
                temp_features.remove(feature)
                print(f"Testing without feature: {feature}")

                # Train with updated feature set
                X_train, X_test, y_train, y_test = train_test_split(X[temp_features], y, test_size=0.2, random_state=42)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                current_f1_score = self.get_f1_score(y_test, y_pred)['F1 Score']

                if current_f1_score >= best_f1_score:
                    best_f1_score = current_f1_score
                    selected_features = temp_features
                    print(f"Removing {feature} improved or maintained F1 Score to {current_f1_score:.4f}")
                else:
                    print(f"Keeping {feature} as removing it decreased F1 Score to {current_f1_score:.4f}")

        # Save final RFE model
        self.selected_features = selected_features
        self.save_final_model(model, model_type, "RFE")
        print(f"Final selected features: {self.selected_features}")
        print(f"Final F1 Score: {best_f1_score:.4f}")
        return model, self.selected_features 

    def perform_spfs(self, model_type):
        self.selected_features = []
        target_column = 'Label'
        
        if target_column not in self.df.columns:
            raise KeyError(f"Column '{target_column}' not found in dataset!")

        y = self.df[target_column]
        X = self.df.drop(columns=[target_column])

        metrics_data = []

        print("Starting SPFS...")
        self.ranked_features = [feature for feature in self.ranked_features if feature != target_column]
        for feature in self.ranked_features:
            self.selected_features.append(feature)
            X_selected = X[self.selected_features]
            X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

            if model_type == 'XGB':
                y_train = y_train.astype('category').cat.codes
                y_test = y_test.astype('category').cat.codes
                model = xgb.XGBClassifier(seed=42)
            else:
                model = RandomForestClassifier(random_state=42)

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            f1_score = self.get_f1_score(y_test, y_pred)['F1 Score']

            if f1_score > max([m['F1 Score'] for m in metrics_data] or [0]):
                print(f"Feature {feature} kept. New best F1 score = {f1_score:.4f}\n")
            else:
                self.selected_features.remove(feature)
                print(f"Feature {feature} removed. F1 score did not improve.")

        # Save final SPFS model
        self.save_final_model(model, model_type, "SPFS")
        print(f"Final selected features: {self.selected_features}")
        return model, self.selected_features
    
    def perform_baseline(self, model_type):

        # Use all features except 'Label'
        self.selected_features = self.df.columns.tolist()
        self.selected_features.remove('Label')

        # Extract features and target
        X = self.df[self.selected_features]
        y = self.df['Label']

        # Encode y if using XGBoost
        if model_type.lower() == "xgb":
            y = y.astype('category').cat.codes  # Convert labels to numerical codes
            model = xgb.XGBClassifier(seed=42)
        else:
            model = RandomForestClassifier(random_state=42)

        # Split data (80% train, 20% test)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the model
        model.fit(X_train, y_train)

        # Save the trained model
        model_path = f"results/models/{self.dataset_name}_{model_type}_None.pkl"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        with open(model_path, 'wb') as model_file:
            pickle.dump(model, model_file)

        print(f"✅ Baseline Model ({model_type}) saved to {model_path}")
        return model, self.selected_features

    def get_f1_score(self, y_true, y_pred):
        f1 = f1_score(y_true, y_pred, average='weighted')
        return {'F1 Score': f1}

    def save_final_model(self, model, model_type, method):
        model_filename = f"{self.dataset_name}_{model_type}_{method}.pkl"
        model_path = os.path.join(self.model_dir, model_filename)

        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

        print(f"Model saved to {model_path}")

    def train_and_save_model(self, selected_features, model_type, model_path):
        print(f"Training model using selected features: {selected_features}")
        target_column = 'Label'
        
        y = self.df[target_column]
        X = self.df[selected_features]  # Use selected features!
    
        # Select model type
        if model_type.lower() == "xgb":
            y = y.astype('category').cat.codes
            model = xgb.XGBClassifier(seed=42)
        else:
            model = RandomForestClassifier(random_state=42)
    
        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model.fit(X_train, y_train)
        
        # Save model
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
    
        print(f"✅ Model saved to {model_path}")
        return model
