# src/agents/analyst_agent.py
import mlflow
import time
from datetime import datetime

class AnalystAgent:
    """Agente responsável por análise de dados e insights quantitativos"""
    
    def __init__(self, model_uri="models:/product_lifecycle_model/latest"):
        self.model = mlflow.sklearn.load_model(model_uri)
        self.metrics = {
            'execution_time': 0,
            'products_analyzed': 0,
            'timestamp': None
        }
    
    def analyze(self, df):
        """Analisa produtos e gera predições"""
        start_time = time.time()
        
        feature_cols = [
            'price', 'stock_quantity', 'sales_last_30d', 
            'views_last_30d', 'rating', 'num_reviews',
            'days_since_launch', 'discount_percentage', 'return_rate',
            'conversion_rate', 'revenue'
        ]
        
        X = df[feature_cols]
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        
        df['predicted_stage'] = predictions
        df['confidence'] = probabilities.max(axis=1)
        
        # Métricas do agente
        self.metrics['execution_time'] = time.time() - start_time
        self.metrics['products_analyzed'] = len(df)
        self.metrics['timestamp'] = datetime.now().isoformat()
        
        # Insights quantitativos
        insights = {
            'total_products': len(df),
            'to_discontinue': (predictions == 0).sum(),
            'to_maintain': (predictions == 1).sum(),
            'to_promote': (predictions == 2).sum(),
            'avg_confidence': probabilities.max(axis=1).mean(),
            'high_confidence_predictions': (probabilities.max(axis=1) > 0.8).sum()
        }
        
        return df, insights, self.metrics