# src/pipeline/data_processing.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_synthetic_data(n_products=1000):
    """Gera dados sintéticos de produtos para o marketplace"""
    np.random.seed(42)
    
    categories = ['Eletrônicos', 'Moda', 'Casa', 'Esportes', 'Livros']
    
    data = {
        'product_id': [f'PROD_{i:04d}' for i in range(n_products)],
        'category': np.random.choice(categories, n_products),
        'price': np.random.uniform(10, 1000, n_products),
        'stock_quantity': np.random.randint(0, 500, n_products),
        'sales_last_30d': np.random.randint(0, 200, n_products),
        'views_last_30d': np.random.randint(0, 5000, n_products),
        'rating': np.random.uniform(1, 5, n_products),
        'num_reviews': np.random.randint(0, 500, n_products),
        'days_since_launch': np.random.randint(1, 730, n_products),
        'discount_percentage': np.random.uniform(0, 50, n_products),
        'return_rate': np.random.uniform(0, 0.3, n_products),
    }
    
    df = pd.DataFrame(data)
    
    # Criar target: lifecycle_stage
    # 0: Descontinuar, 1: Manter, 2: Promover
    df['conversion_rate'] = df['sales_last_30d'] / (df['views_last_30d'] + 1)
    df['revenue'] = df['price'] * df['sales_last_30d']
    
    conditions = [
        (df['sales_last_30d'] < 10) & (df['stock_quantity'] > 100),
        (df['conversion_rate'] > 0.05) & (df['rating'] > 4.0),
    ]
    choices = [0, 2]
    df['lifecycle_stage'] = np.select(conditions, choices, default=1)
    
    return df