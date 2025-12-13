#%%

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import logging
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """Load and prepare data from various sources"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        
        # Create directories if they don't exist
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
    
    def load_kaggle_dataset(self, dataset_name: str = "aimlveera/counterfeit-product-detection-dataset") -> pd.DataFrame:
        """
        Load dataset from Kaggle
        
        Args:
            dataset_name: Kaggle dataset identifier
            
        Returns:
            DataFrame with raw data
        """
        try:
            import kaggle
            
            logger.info(f"Downloading dataset from Kaggle: {dataset_name}")
            
            # Download dataset
            kaggle.api.dataset_download_files(
                dataset_name,
                path=str(self.raw_dir),
                unzip=True
            )
            
            # Find CSV files
            csv_files = list(self.raw_dir.glob("*.csv"))
            
            if not csv_files:
                raise FileNotFoundError("No CSV files found in downloaded dataset")
            
            # Load the first CSV file
            df = pd.read_csv(csv_files[0])
            logger.info(f"Loaded {len(df)} rows from {csv_files[0].name}")
            
            return df
            
        except ImportError:
            logger.error("Kaggle package not installed. Run: pip install kaggle")
            raise
        except Exception as e:
            logger.error(f"Error loading Kaggle dataset: {e}")
            raise
    
    def load_local_file(self, filepath: str) -> pd.DataFrame:
        """
        Load data from local CSV file
        
        Args:
            filepath: Path to CSV file
            
        Returns:
            DataFrame with raw data
        """
        try:
            logger.info(f"Loading local file: {filepath}")
            df = pd.read_csv(filepath)
            logger.info(f"Loaded {len(df)} rows from {filepath}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading local file: {e}")
            raise
    
    def generate_synthetic_data(self, n_samples: int = 1000) -> pd.DataFrame:
        """
        Generate synthetic marketplace data for testing
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            DataFrame with synthetic data
        """
        logger.info(f"Generating {n_samples} synthetic samples")
        
        np.random.seed(42)
        
        # Product categories
        categories = [
            "Eletrônicos", "Moda", "Casa e Decoração", "Esportes",
            "Livros", "Beleza", "Alimentos", "Brinquedos"
        ]
        
        # Generate data
        data = {
            "product_id": [f"PROD_{i:05d}" for i in range(n_samples)],
            "product_name": [f"Produto {i}" for i in range(n_samples)],
            "category": np.random.choice(categories, n_samples),
            "price": np.random.uniform(10, 1000, n_samples).round(2),
            "stock_quantity": np.random.randint(0, 500, n_samples),
            "sales_last_30d": np.random.randint(0, 300, n_samples),
            "views_last_30d": np.random.randint(100, 10000, n_samples),
            "rating": np.random.uniform(1, 5, n_samples).round(1),
            "num_reviews": np.random.randint(0, 500, n_samples),
            "days_since_launch": np.random.randint(1, 730, n_samples),
            "discount_percentage": np.random.uniform(0, 50, n_samples).round(1),
            "return_rate": np.random.uniform(0, 0.3, n_samples).round(3),
            "supplier_rating": np.random.uniform(3, 5, n_samples).round(1),
            "shipping_time_days": np.random.randint(1, 30, n_samples),
            "is_promoted": np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
            "competitor_price": np.random.uniform(10, 1000, n_samples).round(2),
        }
        
        df = pd.DataFrame(data)
        
        # Add some business logic
        # Products with high sales should have lower stock
        high_sales_mask = df["sales_last_30d"] > df["sales_last_30d"].quantile(0.75)
        df.loc[high_sales_mask, "stock_quantity"] = (
            df.loc[high_sales_mask, "stock_quantity"] * 0.5
        ).astype(int)
        
        # Products with high ratings should have more reviews
        high_rating_mask = df["rating"] >= 4.5
        df.loc[high_rating_mask, "num_reviews"] = (
            df.loc[high_rating_mask, "num_reviews"] * 1.5
        ).astype(int)
        
        # Add timestamp
        df["created_at"] = datetime.now().isoformat()
        
        logger.info(f"Generated synthetic data with shape: {df.shape}")
        
        return df
    
    def load_data(
        self,
        source: str = "synthetic",
        kaggle_dataset: Optional[str] = None,
        local_path: Optional[str] = None,
        n_samples: int = 1000
    ) -> pd.DataFrame:
        """
        Load data from specified source
        
        Args:
            source: Data source ('kaggle', 'local', or 'synthetic')
            kaggle_dataset: Kaggle dataset name (if source='kaggle')
            local_path: Path to local file (if source='local')
            n_samples: Number of samples for synthetic data
            
        Returns:
            DataFrame with loaded data
        """
        if source == "kaggle":
            if not kaggle_dataset:
                kaggle_dataset = "aimlveera/counterfeit-product-detection-dataset"
            return self.load_kaggle_dataset(kaggle_dataset)
        
        elif source == "local":
            if not local_path:
                raise ValueError("local_path must be provided when source='local'")
            return self.load_local_file(local_path)
        
        elif source == "synthetic":
            return self.generate_synthetic_data(n_samples)
        
        else:
            raise ValueError(f"Unknown source: {source}. Use 'kaggle', 'local', or 'synthetic'")
    
    def save_processed_data(self, df: pd.DataFrame, filename: str = "processed_data.csv"):
        """Save processed data to disk"""
        filepath = self.processed_dir / filename
        df.to_csv(filepath, index=False)
        logger.info(f"Saved processed data to {filepath}")
        return filepath

#%%
if __name__ == "__main__":
    # Test the data loader
    loader = DataLoader()
    
    # Generate synthetic data
    df = loader.load_data(source="synthetic", n_samples=100)
    print("\nSynthetic Data Sample:")
    print(df.head())
    print(f"\nShape: {df.shape}")
    print(f"\nColumns: {df.columns.tolist()}")
    
    # Save
    loader.save_processed_data(df, "test_data.csv")