# src/models/product_model.py
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import shap
import pandas as pd
import numpy as np

class ProductLifecycleModel:
    def __init__(self, experiment_name="nexocommerce-product-lifecycle"):
        mlflow.set_experiment(experiment_name)
        self.model = None
        self.feature_columns = None
        
    def prepare_features(self, df):
        """Prepara features para o modelo"""
        feature_cols = [
            'price', 'stock_quantity', 'sales_last_30d', 
            'views_last_30d', 'rating', 'num_reviews',
            'days_since_launch', 'discount_percentage', 'return_rate',
            'conversion_rate', 'revenue'
        ]
        
        X = df[feature_cols]
        y = df['lifecycle_stage']
        
        self.feature_columns = feature_cols
        return X, y
    
    def train(self, df, params=None):
        """Treina o modelo com tracking no MLflow"""
        if params is None:
            params = {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'random_state': 42
            }
        
        X, y = self.prepare_features(df)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        with mlflow.start_run(run_name="product_lifecycle_training"):
            # Log parameters
            mlflow.log_params(params)
            
            # Train model
            self.model = RandomForestClassifier(**params)
            self.model.fit(X_train, y_train)
            
            # Predictions
            y_pred = self.model.predict(X_test)
            y_pred_proba = self.model.predict_proba(X_test)
            
            # Metrics
            accuracy = accuracy_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
            
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("roc_auc", roc_auc)
            
            # Feature importance
            feature_importance = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            mlflow.log_dict(feature_importance.to_dict(), "feature_importance.json")
            
            # SHAP values
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(X_test)
            
            # Log model
            mlflow.sklearn.log_model(
                self.model, 
                "model",
                registered_model_name="product_lifecycle_model"
            )
            
            print(f"✅ Model trained - Accuracy: {accuracy:.4f}, ROC-AUC: {roc_auc:.4f}")
            print(f"\n📊 Classification Report:\n{classification_report(y_test, y_pred)}")
            
            return {
                'accuracy': accuracy,
                'roc_auc': roc_auc,
                'feature_importance': feature_importance
            }