# src/api/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import mlflow
from typing import List, Dict
import sys
sys.path.append('..')

from agents.analyst_agent import AnalystAgent
from agents.strategist_agent import StrategistAgent
from agents.reporter_agent import ReporterAgent

app = FastAPI(title="NexoCommerce AI API", version="1.0.0")

# Inicializar agentes
analyst = AnalystAgent()
strategist = StrategistAgent()
reporter = ReporterAgent()

class ProductData(BaseModel):
    product_id: str
    category: str
    price: float
    stock_quantity: int
    sales_last_30d: int
    views_last_30d: int
    rating: float
    num_reviews: int
    days_since_launch: int
    discount_percentage: float
    return_rate: float

@app.post("/analyze")
async def analyze_products(products: List[ProductData]):
    """Endpoint principal para análise de produtos"""
    try:
        # Converter para DataFrame
        df = pd.DataFrame([p.dict() for p in products])
        df['conversion_rate'] = df['sales_last_30d'] / (df['views_last_30d'] + 1)
        df['revenue'] = df['price'] * df['sales_last_30d']
        
        # Pipeline de agentes
        df_analyzed, insights, analyst_metrics = analyst.analyze(df)
        recommendations, strategist_metrics = strategist.generate_recommendations(df_analyzed, insights)
        
        agent_metrics = {
            'analyst': analyst_metrics,
            'strategist': strategist_metrics
        }
        
        report, reporter_metrics = reporter.generate_report(insights, recommendations, agent_metrics)
        report['observability']['reporter_agent'] = reporter_metrics
        
        # Log no MLflow
        with mlflow.start_run(run_name="api_inference"):
            mlflow.log_metrics({
                'products_analyzed': len(df),
                'total_processing_time': report['observability']['total_processing_time']
            })
        
        return report
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": analyst.model is not None}