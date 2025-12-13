import pandas as pd
from typing import Dict, List, Any
import streamlit as st


def validate_product_data(data: Dict[str, Any]) -> tuple[bool, str]:
    """
    Validate product data
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    required_fields = [
        'product_id', 'price', 'rating', 'num_reviews', 
        'stock_quantity', 'sales_last_30d', 'views_last_30d',
        'category', 'brand', 'days_since_launch'
    ]
    
    for field in required_fields:
        if field not in data:
            return False, f"Missing required field: {field}"
    
    # Validate numeric fields
    numeric_fields = ['price', 'rating', 'num_reviews', 'stock_quantity', 
                     'sales_last_30d', 'views_last_30d', 'days_since_launch']
    
    for field in numeric_fields:
        try:
            float(data[field])
        except (ValueError, TypeError):
            return False, f"Field '{field}' must be numeric"
    
    # Validate ranges
    if not (0 <= float(data['rating']) <= 5):
        return False, "Rating must be between 0 and 5"
    
    if float(data['price']) < 0:
        return False, "Price must be positive"
    
    return True, ""


def create_sample_product() -> Dict[str, Any]:
    """Create sample product data"""
    return {
        "product_id": "PROD_SAMPLE_001",
        "price": 99.99,
        "rating": 4.5,
        "num_reviews": 150,
        "stock_quantity": 50,
        "sales_last_30d": 25,
        "views_last_30d": 500,
        "category": "Electronics",
        "brand": "TechBrand",
        "days_since_launch": 180,
        "discount_percentage": 10.0,
        "return_rate": 0.05
    }


def csv_to_products(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Convert DataFrame to list of product dictionaries"""
    return df.to_dict(orient='records')


def products_to_dataframe(products: List[Dict[str, Any]]) -> pd.DataFrame:
    """Convert list of products to DataFrame"""
    return pd.DataFrame(products)


def format_currency(value: float) -> str:
    """Format value as currency"""
    return f"R$ {value:,.2f}"


def format_percentage(value: float) -> str:
    """Format value as percentage"""
    return f"{value*100:.1f}%"


def get_action_color(action: str) -> str:
    """Get color for lifecycle action"""
    colors = {
        "PROMOVER": "green",
        "MANTER": "blue",
        "DESCONTINUAR": "red"
    }
    return colors.get(action, "gray")


def get_action_emoji(action: str) -> str:
    """Get emoji for lifecycle action"""
    emojis = {
        "PROMOVER": "🚀",
        "MANTER": "✅",
        "DESCONTINUAR": "⛔"
    }
    return emojis.get(action, "❓")