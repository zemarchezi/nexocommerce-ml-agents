#%%

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataProcessor:
    """Process and engineer features for product lifecycle prediction"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create engineered features for product lifecycle prediction
        
        Args:
            df: Raw dataframe
            
        Returns:
            DataFrame with engineered features
        """
        logger.info("Creating engineered features...")
        
        df = df.copy()
        
        # 1. Conversion Rate
        df["conversion_rate"] = np.where(
            df["views_last_30d"] > 0,
            df["sales_last_30d"] / df["views_last_30d"],
            0
        )
        
        # 2. Stock Coverage (days of stock remaining)
        df["stock_coverage_days"] = np.where(
            df["sales_last_30d"] > 0,
            (df["stock_quantity"] / df["sales_last_30d"]) * 30,
            999  # High value for products with no sales
        )
        
        # 3. Revenue Last 30 Days
        df["revenue_last_30d"] = df["sales_last_30d"] * df["price"]
        
        # 4. Review Engagement Rate
        df["review_engagement_rate"] = np.where(
            df["sales_last_30d"] > 0,
            df["num_reviews"] / df["sales_last_30d"],
            0
        )
        
        # 5. Price Competitiveness
        if "competitor_price" in df.columns:
            df["price_competitiveness"] = np.where(
                df["competitor_price"] > 0,
                df["price"] / df["competitor_price"],
                1.0
            )
        else:
            df["price_competitiveness"] = 1.0
        
        # 6. Product Age Category
        df["age_category"] = pd.cut(
            df["days_since_launch"],
            bins=[0, 30, 90, 180, 365, 999999],
            labels=["new", "recent", "established", "mature", "old"]
        )
        
        # 7. Performance Score (composite metric)
        df["performance_score"] = (
            (df["rating"] / 5.0) * 0.3 +
            (df["conversion_rate"] * 100) * 0.3 +
            (1 - df["return_rate"]) * 0.2 +
            (df["sales_last_30d"] / df["sales_last_30d"].max()) * 0.2
        )
        
        # 8. Stock Status
        df["stock_status"] = pd.cut(
            df["stock_quantity"],
            bins=[-1, 0, 10, 50, 999999],
            labels=["out_of_stock", "low", "medium", "high"]
        )
        
        # 9. Sales Velocity (sales per day since launch)
        df["sales_velocity"] = np.where(
            df["days_since_launch"] > 0,
            df["sales_last_30d"] / df["days_since_launch"],
            0
        )
        
        # 10. Discount Impact
        df["discount_impact"] = df["discount_percentage"] * df["sales_last_30d"]
        
        # 11. Rating Quality (rating weighted by number of reviews)
        df["rating_quality"] = df["rating"] * np.log1p(df["num_reviews"])
        
        # 12. Price Tier
        df["price_tier"] = pd.qcut(
            df["price"],
            q=4,
            labels=["budget", "mid", "premium", "luxury"],
            duplicates="drop"
        )
        
        logger.info(f"Created {len(df.columns)} total features")
        
        return df
    
    def create_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create target variable for product lifecycle prediction
        
        Target classes:
        - 0: DESCONTINUAR (discontinue)
        - 1: MANTER (maintain)
        - 2: PROMOVER (promote)
        
        Args:
            df: DataFrame with features
            
        Returns:
            DataFrame with target variable
        """
        logger.info("Creating target variable...")
        
        df = df.copy()
        
        # Initialize target
        df["lifecycle_action"] = 1  # Default: MANTER
        
        # DESCONTINUAR: Low performance products
        descontinuar_mask = (
            (df["sales_last_30d"] < df["sales_last_30d"].quantile(0.2)) &
            (df["rating"] < 3.0) &
            (df["return_rate"] > 0.15) &
            (df["days_since_launch"] > 90)
        ) | (
            (df["stock_quantity"] == 0) &
            (df["sales_last_30d"] < df["sales_last_30d"].quantile(0.3))
        )
        
        df.loc[descontinuar_mask, "lifecycle_action"] = 0
        
        # PROMOVER: High potential products
        promover_mask = (
            (df["sales_last_30d"] > df["sales_last_30d"].quantile(0.7)) &
            (df["rating"] >= 4.0) &
            (df["conversion_rate"] > df["conversion_rate"].quantile(0.6)) &
            (df["stock_quantity"] > 10)
        ) | (
            (df["performance_score"] > df["performance_score"].quantile(0.8)) &
            (df["stock_quantity"] > 20)
        )
        
        df.loc[promover_mask, "lifecycle_action"] = 2
        
        # Log distribution
        target_dist = df["lifecycle_action"].value_counts().sort_index()
        logger.info(f"Target distribution:\n{target_dist}")
        logger.info(f"Target percentages:\n{target_dist / len(df) * 100}")
        
        return df
    
    def prepare_features(
        self,
        df: pd.DataFrame,
        is_training: bool = True
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Prepare features for modeling
        
        Args:
            df: DataFrame with engineered features
            is_training: Whether this is training data (fit encoders) or test data (transform only)
            
        Returns:
            Tuple of (processed DataFrame, list of feature names)
        """
        logger.info("Preparing features for modeling...")
        
        df = df.copy()
        
        # Categorical features to encode
        categorical_features = ["category", "age_category", "stock_status", "price_tier"]
        
        # Encode categorical features
        for col in categorical_features:
            if col in df.columns:
                if is_training:
                    self.label_encoders[col] = LabelEncoder()
                    df[f"{col}_encoded"] = self.label_encoders[col].fit_transform(
                        df[col].astype(str)
                    )
                else:
                    if col in self.label_encoders:
                        # Handle unseen categories
                        df[f"{col}_encoded"] = df[col].astype(str).apply(
                            lambda x: self.label_encoders[col].transform([x])[0]
                            if x in self.label_encoders[col].classes_
                            else -1
                        )
        
        # Numerical features
        numerical_features = [
            "price", "stock_quantity", "sales_last_30d", "views_last_30d",
            "rating", "num_reviews", "days_since_launch", "discount_percentage",
            "return_rate", "conversion_rate", "stock_coverage_days",
            "revenue_last_30d", "review_engagement_rate", "price_competitiveness",
            "performance_score", "sales_velocity", "discount_impact", "rating_quality"
        ]
        
        # Add encoded categorical features
        encoded_features = [f"{col}_encoded" for col in categorical_features if col in df.columns]
        
        # Select features that exist in the dataframe
        self.feature_names = [f for f in numerical_features if f in df.columns] + encoded_features
        
        logger.info(f"Selected {len(self.feature_names)} features for modeling")
        
        return df, self.feature_names
    
    def scale_features(
        self,
        df: pd.DataFrame,
        feature_names: List[str],
        is_training: bool = True
    ) -> pd.DataFrame:
        """
        Scale numerical features
        
        Args:
            df: DataFrame with features
            feature_names: List of feature names to scale
            is_training: Whether to fit the scaler
            
        Returns:
            DataFrame with scaled features
        """
        df = df.copy()
        
        if is_training:
            df[feature_names] = self.scaler.fit_transform(df[feature_names])
            logger.info("Fitted and transformed features")
        else:
            df[feature_names] = self.scaler.transform(df[feature_names])
            logger.info("Transformed features using fitted scaler")
        
        return df
    
    def process_pipeline(
        self,
        df: pd.DataFrame,
        is_training: bool = True,
        create_target_var: bool = True
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Complete processing pipeline
        
        Args:
            df: Raw dataframe
            is_training: Whether this is training data
            create_target_var: Whether to create target variable
            
        Returns:
            Tuple of (processed DataFrame, feature names)
        """
        # Create features
        df = self.create_features(df)
        
        # Create target if needed
        if create_target_var:
            df = self.create_target(df)
        
        # Prepare features
        df, feature_names = self.prepare_features(df, is_training)
        
        # Scale features
        df = self.scale_features(df, feature_names, is_training)
        
        logger.info("Processing pipeline completed")
        
        return df, feature_names
#%%

if __name__ == "__main__":
    # Test the processor
    from data_loader import DataLoader
    
    loader = DataLoader()
    df = loader.load_data(source="synthetic", n_samples=100)
    
    processor = DataProcessor()
    processed_df, features = processor.process_pipeline(df, is_training=True)
    
    print("\nProcessed Data Sample:")
    print(processed_df[features + ["lifecycle_action"]].head())
    print(f"\nFeatures: {features}")
    print(f"\nTarget distribution:\n{processed_df['lifecycle_action'].value_counts()}")