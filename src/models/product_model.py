#%%
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)
import mlflow
import mlflow.sklearn
from typing import Dict, Tuple, Optional, Any
import logging
import joblib
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProductLifecycleModel:
    """Model for predicting product lifecycle actions"""
    
    def __init__(self, model_type: str = "random_forest"):
        """
        Initialize model
        
        Args:
            model_type: Type of model ('random_forest' or 'gradient_boosting')
        """
        self.model_type = model_type
        self.model = None
        self.feature_names = None
        self.feature_importance = None
        self.classes = ["DESCONTINUAR", "MANTER", "PROMOVER"]
        
        # Initialize model
        if model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1
            )
        elif model_type == "gradient_boosting":
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
    
    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
        log_mlflow: bool = True
    ) -> Dict[str, Any]:
        """
        Train the model
        
        Args:
            X: Features
            y: Target variable
            test_size: Proportion of test set
            log_mlflow: Whether to log to MLflow
            
        Returns:
            Dictionary with training metrics
        """
        logger.info(f"Training {self.model_type} model...")
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
        
        # Start MLflow run
        if log_mlflow:
            mlflow.start_run(run_name=f"{self.model_type}_training")
            mlflow.log_param("model_type", self.model_type)
            mlflow.log_param("n_features", len(self.feature_names))
            mlflow.log_param("train_size", len(X_train))
            mlflow.log_param("test_size", len(X_test))
        
        # Train model
        self.model.fit(X_train, y_train)
        logger.info("Model training completed")
        
        # Predictions
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)
        y_pred_proba_test = self.model.predict_proba(X_test)
        
        # Calculate metrics
        metrics = self._calculate_metrics(
            y_train, y_pred_train, y_test, y_pred_test, y_pred_proba_test
        )
        
        # Feature importance
        self.feature_importance = pd.DataFrame({
            "feature": self.feature_names,
            "importance": self.model.feature_importances_
        }).sort_values("importance", ascending=False)
        
        # Log to MLflow
        if log_mlflow:
            # Log metrics
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            # Log feature importance
            mlflow.log_dict(
                self.feature_importance.to_dict(orient="records"),
                "feature_importance.json"
            )
            
            # Log model
            mlflow.sklearn.log_model(
                self.model,
                "model",
                registered_model_name=f"product_lifecycle_{self.model_type}"
            )
            
            # Log confusion matrix
            cm = confusion_matrix(y_test, y_pred_test)
            mlflow.log_dict(
                {"confusion_matrix": cm.tolist()},
                "confusion_matrix.json"
            )
            
            mlflow.end_run()
        
        logger.info(f"Training completed. Test Accuracy: {metrics['test_accuracy']:.4f}")
        
        return metrics
    
    def _calculate_metrics(
        self,
        y_train: pd.Series,
        y_pred_train: np.ndarray,
        y_test: pd.Series,
        y_pred_test: np.ndarray,
        y_pred_proba_test: np.ndarray
    ) -> Dict[str, float]:
        """Calculate evaluation metrics"""
        
        metrics = {
            # Train metrics
            "train_accuracy": accuracy_score(y_train, y_pred_train),
            "train_precision": precision_score(y_train, y_pred_train, average="weighted"),
            "train_recall": recall_score(y_train, y_pred_train, average="weighted"),
            "train_f1": f1_score(y_train, y_pred_train, average="weighted"),
            
            # Test metrics
            "test_accuracy": accuracy_score(y_test, y_pred_test),
            "test_precision": precision_score(y_test, y_pred_test, average="weighted"),
            "test_recall": recall_score(y_test, y_pred_test, average="weighted"),
            "test_f1": f1_score(y_test, y_pred_test, average="weighted"),
        }
        
        # ROC-AUC (multi-class)
        try:
            metrics["test_roc_auc"] = roc_auc_score(
                y_test, y_pred_proba_test, multi_class="ovr", average="weighted"
            )
        except Exception as e:
            logger.warning(f"Could not calculate ROC-AUC: {e}")
            metrics["test_roc_auc"] = 0.0
        
        return metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions
        
        Args:
            X: Features
            
        Returns:
            Array of predictions
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict probabilities
        
        Args:
            X: Features
            
        Returns:
            Array of probabilities for each class
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        return self.model.predict_proba(X)
    
    def predict_with_confidence(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Make predictions with confidence scores
        
        Args:
            X: Features
            
        Returns:
            DataFrame with predictions and confidence
        """
        predictions = self.predict(X)
        probabilities = self.predict_proba(X)
        
        # Get confidence (max probability)
        confidence = probabilities.max(axis=1)
        
        # Create result dataframe
        result = pd.DataFrame({
            "prediction": predictions,
            "prediction_label": [self.classes[p] for p in predictions],
            "confidence": confidence,
            "prob_descontinuar": probabilities[:, 0],
            "prob_manter": probabilities[:, 1],
            "prob_promover": probabilities[:, 2]
        })
        
        return result
    
    def get_feature_importance(self, top_n: int = 10) -> pd.DataFrame:
        """
        Get top N most important features
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importance
        """
        if self.feature_importance is None:
            raise ValueError("Model not trained yet")
        
        return self.feature_importance.head(top_n)
    
    def save_model(self, filepath: str):
        """Save model to disk"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            "model": self.model,
            "feature_names": self.feature_names,
            "feature_importance": self.feature_importance,
            "model_type": self.model_type,
            "classes": self.classes
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model from disk"""
        model_data = joblib.load(filepath)
        
        self.model = model_data["model"]
        self.feature_names = model_data["feature_names"]
        self.feature_importance = model_data["feature_importance"]
        self.model_type = model_data["model_type"]
        self.classes = model_data["classes"]
        
        logger.info(f"Model loaded from {filepath}")
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Evaluate model on given data
        
        Args:
            X: Features
            y: True labels
            
        Returns:
            Dictionary with evaluation results
        """
        y_pred = self.predict(X)
        y_pred_proba = self.predict_proba(X)
        
        results = {
            "accuracy": accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred, average="weighted"),
            "recall": recall_score(y, y_pred, average="weighted"),
            "f1": f1_score(y, y_pred, average="weighted"),
            "classification_report": classification_report(
                y, y_pred, target_names=self.classes
            ),
            "confusion_matrix": confusion_matrix(y, y_pred)
        }
        
        try:
            results["roc_auc"] = roc_auc_score(
                y, y_pred_proba, multi_class="ovr", average="weighted"
            )
        except:
            results["roc_auc"] = None
        
        return results

#%%

if __name__ == "__main__":
    # Test the model
    from data_loader import DataLoader
    from data_processing import DataProcessor
    
    # Load and process data
    loader = DataLoader()
    df = loader.load_data(source="synthetic", n_samples=1000)
    
    processor = DataProcessor()
    processed_df, features = processor.process_pipeline(df, is_training=True)
    
    # Prepare data
    X = processed_df[features]
    y = processed_df["lifecycle_action"]
    
    # Train model
    model = ProductLifecycleModel(model_type="random_forest")
    metrics = model.train(X, y, log_mlflow=False)
    
    print("\nTraining Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    print("\nTop 10 Feature Importance:")
    print(model.get_feature_importance(top_n=10))
    
    # Test predictions
    sample = X.head(5)
    predictions = model.predict_with_confidence(sample)
    print("\nSample Predictions:")
    print(predictions)