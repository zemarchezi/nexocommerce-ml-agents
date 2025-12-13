#%%
import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from datetime import datetime
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import json

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from pipeline.data_loader import DataLoader
from pipeline.data_processing import DataProcessor
from models.product_model import ProductLifecycleModel

def convert_to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    return obj
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TrainingPipeline:
    """
    Complete training pipeline for Product Lifecycle Model

    Features:
    - Data loading from multiple sources
    - Automated feature engineering
    - Model training with hyperparameter tuning
    - Cross-validation
    - MLflow experiment tracking
    - Model versioning and registry
    - Performance evaluation
    - Model persistence
    """

    def __init__(
        self,
        experiment_name: str = "product_lifecycle_model",
        mlflow_tracking_uri: str = "./mlruns",
        random_state: int = 42
    ):
        """
        Initialize Training Pipeline

        Args:
            experiment_name: MLflow experiment name
            mlflow_tracking_uri: MLflow tracking URI
            random_state: Random seed for reproducibility
        """
        self.experiment_name = experiment_name
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.random_state = random_state

        # Initialize components
        self.data_loader = DataLoader()
        self.data_processor = DataProcessor()
        self.model = None

        # Setup MLflow
        self._setup_mlflow()

        # Pipeline metadata
        self.metadata = {
            "pipeline_version": "1.0.0",
            "created_at": datetime.now().isoformat(),
            "random_state": random_state
        }

        logger.info(f"Training Pipeline initialized - Experiment: {experiment_name}")

    def _setup_mlflow(self):
        """Setup MLflow tracking"""
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        logger.info(f"MLflow tracking URI: {self.mlflow_tracking_uri}")
        logger.info(f"MLflow experiment: {self.experiment_name}")

    def load_data(
        self,
        source: str = "synthetic",
        dataset_name: Optional[str] = None,
        file_path: Optional[str] = None,
        n_samples: int = 1000
    ) -> pd.DataFrame:
        """
        Load data from specified source

        Args:
            source: Data source ('kaggle', 'local', 'synthetic')
            dataset_name: Kaggle dataset name (if source='kaggle')
            file_path: Local file path (if source='local')
            n_samples: Number of samples for synthetic data

        Returns:
            DataFrame with raw data
        """
        logger.info(f"Loading data from source: {source}")

        if source == "kaggle":
            if not dataset_name:
                dataset_name = "aimlveera/counterfeit-product-detection-dataset"
            df = self.data_loader.load_data(source="kaggle", dataset_name=dataset_name)

        elif source == "local":
            if not file_path:
                raise ValueError("file_path required for local source")
            df = self.data_loader.load_data(source="local", file_path=file_path)

        elif source == "synthetic":
            df = self.data_loader.load_data(source="synthetic", n_samples=n_samples)

        else:
            raise ValueError(f"Unknown source: {source}")

        logger.info(f"Data loaded successfully - Shape: {df.shape}")

        # Log data statistics
        self._log_data_statistics(df, stage="raw")

        return df

    def _log_data_statistics(self, df: pd.DataFrame, stage: str):
        """Log data statistics"""
        stats = {
            f"{stage}_rows": len(df),
            f"{stage}_columns": len(df.columns),
            f"{stage}_missing_values": df.isnull().sum().sum(),
            f"{stage}_duplicates": df.duplicated().sum()
        }

        logger.info(f"Data statistics ({stage}): {stats}")

        return stats

    def preprocess_data(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        validation_size: float = 0.1
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list]:
        """
        Preprocess data and split into train/val/test

        Args:
            df: Raw dataframe
            test_size: Test set proportion
            validation_size: Validation set proportion

        Returns:
            Tuple of (train_df, val_df, test_df, feature_names)
        """
        logger.info("Starting data preprocessing...")

        # Process data
        processed_df, features = self.data_processor.process_pipeline(
            df,
            is_training=True,
            create_target_var=True
        )

        logger.info(f"Features created: {len(features)}")
        logger.info(f"Feature names: {features}")

        # Check target variable
        if "lifecycle_action" not in processed_df.columns:
            raise ValueError("Target variable 'lifecycle_action' not found")

        # Log target distribution
        target_dist = processed_df["lifecycle_action"].value_counts().to_dict()
        logger.info(f"Target distribution: {target_dist}")

        # Split data
        train_df, temp_df = train_test_split(
            processed_df,
            test_size=(test_size + validation_size),
            random_state=self.random_state,
            stratify=processed_df["lifecycle_action"]
        )

        val_size_adjusted = validation_size / (test_size + validation_size)
        val_df, test_df = train_test_split(
            temp_df,
            test_size=(1 - val_size_adjusted),
            random_state=self.random_state,
            stratify=temp_df["lifecycle_action"]
        )

        logger.info(f"Data split - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

        # Save processed data
        self._save_processed_data(train_df, val_df, test_df, features)

        return train_df, val_df, test_df, features

    def _save_processed_data(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        features: list
    ):
        """Save processed data to disk"""
        data_dir = Path("data/processed")
        data_dir.mkdir(parents=True, exist_ok=True)

        train_df.to_csv(data_dir / "train.csv", index=False)
        val_df.to_csv(data_dir / "val.csv", index=False)
        test_df.to_csv(data_dir / "test.csv", index=False)

        # Save feature names
        with open(data_dir / "features.json", "w") as f:
            json.dump({"features": features}, f, indent=2)

        logger.info(f"Processed data saved to {data_dir}")

    def train_model(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        features: list,
        model_type: str = "random_forest",
        hyperparameter_tuning: bool = False,
        cv_folds: int = 5
    ) -> ProductLifecycleModel:
        """
        Train model with optional hyperparameter tuning

        Args:
            train_df: Training dataframe
            val_df: Validation dataframe
            features: List of feature names
            model_type: Model type ('random_forest' or 'gradient_boosting')
            hyperparameter_tuning: Whether to perform hyperparameter tuning
            cv_folds: Number of cross-validation folds

        Returns:
            Trained model
        """
        logger.info(f"Training {model_type} model...")

        # Prepare data
        X_train = train_df[features]
        y_train = train_df["lifecycle_action"]
        X_val = val_df[features]
        y_val = val_df["lifecycle_action"]

        # Start MLflow run
        with mlflow.start_run(run_name=f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):

            # Log parameters
            mlflow.log_param("model_type", model_type)
            mlflow.log_param("n_features", len(features))
            mlflow.log_param("train_size", len(train_df))
            mlflow.log_param("val_size", len(val_df))
            mlflow.log_param("hyperparameter_tuning", hyperparameter_tuning)
            mlflow.log_param("cv_folds", cv_folds)

            # Initialize model
            self.model = ProductLifecycleModel(model_type=model_type)

            # Hyperparameter tuning
            if hyperparameter_tuning:
                logger.info("Performing hyperparameter tuning...")
                best_model = self._hyperparameter_tuning(
                    X_train, y_train, model_type, cv_folds
                )
                mlflow.log_params({f"best_{k}": v for k, v in best_model.best_params_.items()})

                # Update model with best estimator from GridSearchCV
                self.model.model = best_model.best_estimator_
                self.model.is_trained = True
                logger.info("Model updated with best hyperparameters from GridSearchCV")
            else:
                # Train model with default parameters
                logger.info("Training model with default parameters...")
                self.model.train(X_train, y_train, log_mlflow=False)

            # Cross-validation
            logger.info(f"Performing {cv_folds}-fold cross-validation...")
            cv_scores = cross_val_score(
                self.model.model,
                X_train,
                y_train,
                cv=cv_folds,
                scoring='accuracy',
                n_jobs=-1
            )

            logger.info(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
            mlflow.log_metric("cv_accuracy_mean", cv_scores.mean())
            mlflow.log_metric("cv_accuracy_std", cv_scores.std())

            # Evaluate on validation set
            logger.info("Evaluating on validation set...")
            val_metrics = self.model.evaluate(X_val, y_val)

            # Log validation metrics
            for metric_name, metric_value in val_metrics.items():
                # Only log numeric metrics
                if isinstance(metric_value, (int, float)):
                    mlflow.log_metric(f"val_{metric_name}", metric_value)
                elif isinstance(metric_value, str):
                    # Log text reports as artifacts
                    mlflow.log_text(metric_value, f"val_{metric_name}.txt")

            # Generate predictions for detailed analysis
            y_val_pred = self.model.model.predict(X_val)

            # Classification report
            class_report = classification_report(
                y_val,
                y_val_pred,
                target_names=["DESCONTINUAR", "MANTER", "PROMOVER"],
                output_dict=True
            )

            logger.info("\nClassification Report (Validation):")
            logger.info(classification_report(
                y_val,
                y_val_pred,
                target_names=["DESCONTINUAR", "MANTER", "PROMOVER"]
            ))

            # Log per-class metrics
            for class_name, metrics in class_report.items():
                if isinstance(metrics, dict):
                    for metric_name, value in metrics.items():
                        mlflow.log_metric(f"val_{class_name}_{metric_name}", value)

            # Confusion matrix
            cm = confusion_matrix(y_val, y_val_pred)
            logger.info(f"\nConfusion Matrix (Validation):\n{cm}")

            # Feature importance
            feature_importance = self.model.get_feature_importance()
            logger.info("\nTop 10 Most Important Features:")
            for idx, row in feature_importance.iterrows():
                feat = row['feature']
                importance = row['importance']
                logger.info(f"  {feat}: {importance:.4f}")
                mlflow.log_metric(f"feature_importance_{feat}", float(importance))

            # Log model
            mlflow.sklearn.log_model(
                self.model.model,
                "model",
                registered_model_name=f"{model_type}_product_lifecycle"
            )

            # Log artifacts
            self._log_artifacts(features, feature_importance, class_report, cm)

            logger.info("Model training completed successfully!")

        return self.model

    def _hyperparameter_tuning(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        model_type: str,
        cv_folds: int
    ) -> GridSearchCV:
        """
        Perform hyperparameter tuning using GridSearchCV

        Args:
            X_train: Training features
            y_train: Training target
            model_type: Model type
            cv_folds: Number of CV folds

        Returns:
            Fitted GridSearchCV object
        """
        if model_type == "random_forest":
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            base_model = ProductLifecycleModel(model_type="random_forest").model

        elif model_type == "gradient_boosting":
            param_grid = {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'min_samples_split': [2, 5, 10]
            }
            base_model = ProductLifecycleModel(model_type="gradient_boosting").model

        else:
            raise ValueError(f"Unknown model type: {model_type}")

        logger.info(f"Grid search with {len(param_grid)} parameters...")

        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=cv_folds,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(X_train, y_train)

        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best CV score: {grid_search.best_score_:.4f}")

        return grid_search

    def _log_artifacts(
        self,
        features: list,
        feature_importance: list,
        class_report: dict,
        confusion_matrix: np.ndarray
    ):
        """Log artifacts to MLflow"""
        artifacts_dir = Path("artifacts")
        artifacts_dir.mkdir(exist_ok=True)

        # Save features
        with open(artifacts_dir / "features.json", "w") as f:
            json.dump({"features": features}, f, indent=2)

        # Save feature importance
        with open(artifacts_dir / "feature_importance.json", "w") as f:
            json.dump(
                {"feature_importance": feature_importance.to_dict('records')},
                f,
                indent=2
            )

        # Save classification report
        with open(artifacts_dir / "classification_report.json", "w") as f:
            json.dump(class_report, f, indent=2)

        # Save confusion matrix
        np.save(artifacts_dir / "confusion_matrix.npy", confusion_matrix)

        # Log to MLflow
        mlflow.log_artifacts(str(artifacts_dir))

        logger.info(f"Artifacts logged to MLflow")

    def evaluate_test_set(
        self,
        test_df: pd.DataFrame,
        features: list
    ) -> Dict[str, Any]:
        """
        Evaluate model on test set

        Args:
            test_df: Test dataframe
            features: Feature names

        Returns:
            Test metrics
        """
        logger.info("Evaluating on test set...")

        X_test = test_df[features]
        y_test = test_df["lifecycle_action"]

        # Evaluate
        test_metrics = self.model.evaluate(X_test, y_test)

        # Predictions
        y_test_pred = self.model.model.predict(X_test)

        # Classification report
        class_report = classification_report(
            y_test,
            y_test_pred,
            target_names=["DESCONTINUAR", "MANTER", "PROMOVER"]
        )

        logger.info("\n" + "="*80)
        logger.info("TEST SET EVALUATION")
        logger.info("="*80)
        logger.info(f"\nTest Metrics:")
        for metric_name, metric_value in test_metrics.items():
            if isinstance(metric_value, (int, float)):
                logger.info(f"  {metric_name}: {metric_value:.4f}")
            else:
                logger.info(f"  {metric_name}:\n{metric_value}")

        logger.info(f"\nClassification Report (Test):")
        logger.info(class_report)

        # Confusion matrix
        cm = confusion_matrix(y_test, y_test_pred)
        logger.info(f"\nConfusion Matrix (Test):\n{cm}")

        return test_metrics

    def save_processor(self, processor_path: str):
        """Save the data processor"""
        import pickle
        os.makedirs(os.path.dirname(processor_path), exist_ok=True)
        
        processor_data = {
            'scaler': self.data_processor.scaler,
            'label_encoders': self.data_processor.label_encoders,
            'feature_names': self.data_processor.feature_names
        }
        
        with open(processor_path, 'wb') as f:
            pickle.dump(processor_data, f)
        
        logger.info(f"Processor saved to {processor_path}")

    def save_model(self, model_path: str = "models/product_lifecycle_model.pkl"):
        """
        Save trained model to disk

        Args:
            model_path: Path to save model
        """
        model_dir = Path(model_path).parent
        model_dir.mkdir(parents=True, exist_ok=True)

        self.model.save_model(model_path)
        logger.info(f"Model saved to {model_path}")

    def run_complete_pipeline(
        self,
        source: str = "synthetic",
        dataset_name: Optional[str] = None,
        file_path: Optional[str] = None,
        n_samples: int = 1000,
        model_type: str = "random_forest",
        hyperparameter_tuning: bool = False,
        test_size: float = 0.2,
        validation_size: float = 0.1,
        cv_folds: int = 5,
        save_model_path: str = "models/product_lifecycle_model.pkl"
    ) -> Dict[str, Any]:
        """
        Run complete training pipeline

        Args:
            source: Data source
            dataset_name: Kaggle dataset name
            file_path: Local file path
            n_samples: Number of synthetic samples
            model_type: Model type
            hyperparameter_tuning: Enable hyperparameter tuning
            test_size: Test set size
            validation_size: Validation set size
            cv_folds: Cross-validation folds
            save_model_path: Path to save model

        Returns:
            Pipeline results
        """
        logger.info("\n" + "="*80)
        logger.info("STARTING COMPLETE TRAINING PIPELINE")
        logger.info("="*80)

        start_time = datetime.now()

        try:
            # 1. Load data
            logger.info("\n[1/5] Loading data...")
            df = self.load_data(
                source=source,
                dataset_name=dataset_name,
                file_path=file_path,
                n_samples=n_samples
            )

            # 2. Preprocess data
            logger.info("\n[2/5] Preprocessing data...")
            train_df, val_df, test_df, features = self.preprocess_data(
                df,
                test_size=test_size,
                validation_size=validation_size
            )

            # 3. Train model
            logger.info("\n[3/5] Training model...")
            self.model = self.train_model(
                train_df,
                val_df,
                features,
                model_type=model_type,
                hyperparameter_tuning=hyperparameter_tuning,
                cv_folds=cv_folds
            )

            # 4. Evaluate on test set
            logger.info("\n[4/5] Evaluating on test set...")
            test_metrics = self.evaluate_test_set(test_df, features)

            # 5. Save model
            if save_model_path:
                logger.info(f"\n[5/5] Saving model to {save_model_path}...")
                self.model.save_model(save_model_path)

                # Save the data processor as well
                processor_path = save_model_path.replace('.pkl', '_processor.pkl')
                self.save_processor(processor_path)
                logger.info(f"Model and processor saved successfully")

            # Calculate total time
            end_time = datetime.now()
            total_time = (end_time - start_time).total_seconds()

            # Prepare results
            results = {
                "status": "SUCCESS",
                "pipeline_version": self.metadata["pipeline_version"],
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "total_time_seconds": total_time,
                "data_source": source,
                "model_type": model_type,
                "hyperparameter_tuning": hyperparameter_tuning,
                "data_splits": {
                    "train": len(train_df),
                    "validation": len(val_df),
                    "test": len(test_df),
                    "total": len(df)
                },
                "features": {
                    "count": len(features),
                    "names": features
                },
                "test_metrics": test_metrics,
                "model_path": save_model_path,
                "mlflow_experiment": self.experiment_name
            }

            logger.info("\n" + "="*80)
            logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
            logger.info("="*80)
            logger.info(f"Total time: {total_time:.2f} seconds")
            logger.info(f"Model saved to: {save_model_path}")
            logger.info(f"Test Accuracy: {test_metrics['accuracy']:.4f}")

            return results

        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            raise


def main():
    """Main function for CLI"""
    parser = argparse.ArgumentParser(description="Train Product Lifecycle Model")

    parser.add_argument(
        "--source",
        type=str,
        default="synthetic",
        choices=["kaggle", "local", "synthetic"],
        help="Data source"
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help="Kaggle dataset name"
    )
    parser.add_argument(
        "--file-path",
        type=str,
        default=None,
        help="Local file path"
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=1000,
        help="Number of synthetic samples"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="random_forest",
        choices=["random_forest", "gradient_boosting"],
        help="Model type"
    )
    parser.add_argument(
        "--hyperparameter-tuning",
        action="store_true",
        help="Enable hyperparameter tuning"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test set size"
    )
    parser.add_argument(
        "--validation-size",
        type=float,
        default=0.1,
        help="Validation set size"
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Cross-validation folds"
    )
    parser.add_argument(
        "--save-model-path",
        type=str,
        default="models/product_lifecycle_model.pkl",
        help="Path to save model"
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="product_lifecycle_model",
        help="MLflow experiment name"
    )

    args = parser.parse_args()

    # Create pipeline
    pipeline = TrainingPipeline(experiment_name=args.experiment_name)

    # Run pipeline
    results = pipeline.run_complete_pipeline(
        source=args.source,
        dataset_name=args.dataset_name,
        file_path=args.file_path,
        n_samples=args.n_samples,
        model_type=args.model_type,
        hyperparameter_tuning=args.hyperparameter_tuning,
        test_size=args.test_size,
        validation_size=args.validation_size,
        cv_folds=args.cv_folds,
        save_model_path=args.save_model_path
    )

    # Save results
    results_path = Path("results")
    results_path.mkdir(exist_ok=True)

    results_file = results_path / f"training_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, "w") as f:
        serializable_results = convert_to_serializable(results)
        json.dump(serializable_results, f, indent=2)

    logger.info(f"\nResults saved to: {results_file}")
#%%

if __name__ == "__main__":
    main()