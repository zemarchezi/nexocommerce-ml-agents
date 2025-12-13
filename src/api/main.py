#%%
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import pandas as pd
import uvicorn

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from models.product_model import ProductLifecycleModel
from agents.analyst_agent import AnalystAgent
from agents.strategist_agent import StrategistAgent
from agents.reporter_agent import ReporterAgent
from pipeline.data_processing import DataProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="NexoCommerce Multi-Agent API",
    description="AI-powered product lifecycle management system with multi-agent architecture",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and agents
model: Optional[ProductLifecycleModel] = None
data_processor: Optional[DataProcessor] = None
analyst_agent: Optional[AnalystAgent] = None
strategist_agent: Optional[StrategistAgent] = None
reporter_agent: Optional[ReporterAgent] = None

# ============================================================================
# Pydantic Models (Request/Response Schemas)
# ============================================================================

class ProductData(BaseModel):
    product_id: str
    price: float
    rating: float
    num_reviews: int
    stock_quantity: int
    sales_last_30d: int
    views_last_30d: int
    category: Optional[str] = "Unknown"
    brand: Optional[str] = "Unknown"

    # Add these optional fields with defaults
    days_since_launch: Optional[int] = 180  # Default 6 months
    discount_percentage: Optional[float] = 0.0
    return_rate: Optional[float] = 0.05  # 5% default
    conversion_rate: Optional[float] = None  # Will be calculated
    launch_date: Optional[str] = None
    last_updated: Optional[str] = None
    class Config:
        schema_extra = {
            "example": {
                "product_id": "PROD_001",
                "price": 99.99,
                "rating": 4.5,
                "num_reviews": 150,
                "stock_quantity": 50,
                "sales_last_30d": 25,
                "views_last_30d": 500,
                "category": "Electronics",
                "brand": "TechBrand"
            }
        }


class BatchProductData(BaseModel):
    """Batch of products for prediction"""
    products: List[ProductData] = Field(..., description="List of products")

    class Config:
        schema_extra = {
            "example": {
                "products": [
                    {
                        "product_id": "PROD_001",
                        "price": 99.99,
                        "rating": 4.5,
                        "num_reviews": 150,
                        "stock_quantity": 50,
                        "sales_last_30d": 25,
                        "views_last_30d": 500,
                        "category": "Electronics",
                        "brand": "TechBrand"
                    }
                ]
            }
        }


class PredictionResponse(BaseModel):
    """Prediction response"""
    product_id: str
    prediction: str
    confidence: float
    probabilities: Dict[str, float]
    timestamp: str


class BatchPredictionResponse(BaseModel):
    """Batch prediction response"""
    predictions: List[PredictionResponse]
    total_products: int
    timestamp: str


class AnalysisRequest(BaseModel):
    """Request for multi-agent analysis"""
    products: List[ProductData] = Field(..., description="Products to analyze")
    analysis_type: str = Field(
        "complete",
        description="Analysis type: 'complete', 'quick', 'detailed'"
    )
    include_recommendations: bool = Field(True, description="Include strategic recommendations")

    class Config:
        schema_extra = {
            "example": {
                "products": [
                    {
                        "product_id": "PROD_001",
                        "price": 99.99,
                        "rating": 4.5,
                        "num_reviews": 150,
                        "stock_quantity": 50,
                        "sales_last_30d": 25,
                        "views_last_30d": 500
                    }
                ],
                "analysis_type": "complete",
                "include_recommendations": True
            }
        }


class AnalysisResponse(BaseModel):
    """Multi-agent analysis response"""
    analysis_id: str
    timestamp: str
    products_analyzed: int
    analyst_insights: Dict[str, Any]
    strategist_recommendations: Optional[Dict[str, Any]]
    executive_summary: Dict[str, Any]
    status: str


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    agents_initialized: bool
    timestamp: str
    version: str


# ============================================================================
# Startup and Shutdown Events
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize model and agents on startup"""
    global model, data_processor, analyst_agent, strategist_agent, reporter_agent

    logger.info("Starting NexoCommerce API...")

    try:
        # Initialize data processor
        data_processor = DataProcessor()
        logger.info("✓ Data processor initialized")

        # Load model
        model_path = os.getenv("MODEL_PATH", "models/product_lifecycle_model.pkl")
        processor_path = model_path.replace('.pkl', '_processor.pkl')

        # Load processor state if exists
        if os.path.exists(processor_path):
            import pickle
            with open(processor_path, 'rb') as f:
                processor_data = pickle.load(f)
            data_processor.scaler = processor_data['scaler']
            data_processor.label_encoders = processor_data['label_encoders']
            data_processor.feature_names = processor_data['feature_names']
            logger.info("✓ Data processor loaded with fitted transformers")
        else:
            logger.warning("⚠ Processor file not found, using unfitted processor")

        # Initialize and load model (THIS WAS MISSING!)
        model = ProductLifecycleModel()
        model.load_model(model_path)
        logger.info(f"✓ Model loaded from {model_path}")
        logger.info(f"✓ Model is_trained: {model.is_trained}")

        # Initialize agents
        analyst_agent = AnalystAgent(model=model, processor=data_processor)
        strategist_agent = StrategistAgent()
        reporter_agent = ReporterAgent()
        logger.info("✓ Multi-agent system initialized")

        logger.info("="*80)
        logger.info("🚀 NexoCommerce API is ready!")
        logger.info("="*80)

    except Exception as e:
        logger.error(f"Failed to initialize API: {e}", exc_info=True)
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down NexoCommerce API...")


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "NexoCommerce Multi-Agent API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if model and model.is_trained else "degraded",
        model_loaded=model is not None and model.is_trained,
        agents_initialized=all([analyst_agent, strategist_agent, reporter_agent]),
        timestamp=datetime.now().isoformat(),
        version="1.0.0"
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_single(product: ProductData):
    """Predict lifecycle action for a single product"""
    try:
        # Convert to dict and add missing fields with defaults
        product_dict = product.dict()
        
        # Add calculated fields if missing
        if product_dict.get('conversion_rate') is None:
            views = product_dict.get('views_last_30d', 1)
            sales = product_dict.get('sales_last_30d', 0)
            product_dict['conversion_rate'] = sales / views if views > 0 else 0.0
        
        # Convert to DataFrame
        df = pd.DataFrame([product_dict])
        
        # Process and predict
        processed_df, features = data_processor.process_pipeline(
            df, 
            is_training=False
        )

        # Make prediction
        X = processed_df[features]
        prediction = model.predict(X)[0]
        probabilities = model.predict_proba(X)[0]

        # Map prediction to action
        action_map = {0: "DESCONTINUAR", 1: "MANTER", 2: "PROMOVER"}
        predicted_action = action_map[prediction]

        # Get probabilities for each class
        prob_dict = {
            "DESCONTINUAR": float(probabilities[0]),
            "MANTER": float(probabilities[1]),
            "PROMOVER": float(probabilities[2])
        }

        return PredictionResponse(
            product_id=product.product_id,
            prediction=predicted_action,
            confidence=float(max(probabilities)),
            probabilities=prob_dict,
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(batch: BatchProductData):
    """
    Predict lifecycle actions for multiple products

    Efficient batch prediction endpoint
    """
    if not model or not model.is_trained:
        raise HTTPException(status_code=503, detail="Model not loaded or not trained")

    try:
        # Convert to DataFrame
        products_data = [p.dict() for p in batch.products]
        products_df = pd.DataFrame(products_data)

        # Process features
        processed_df, features = data_processor.process_pipeline(
            products_df,
            is_training=False,
            create_target_var=False
        )

        # Make predictions
        X = processed_df[features]
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)

        # Map predictions to actions
        action_map = {0: "DESCONTINUAR", 1: "MANTER", 2: "PROMOVER"}

        # Build response
        prediction_results = []
        for i, product in enumerate(batch.products):
            predicted_action = action_map[predictions[i]]
            prob_dict = {
                "DESCONTINUAR": float(probabilities[i][0]),
                "MANTER": float(probabilities[i][1]),
                "PROMOVER": float(probabilities[i][2])
            }

            prediction_results.append(
                PredictionResponse(
                    product_id=product.product_id,
                    prediction=predicted_action,
                    confidence=float(max(probabilities[i])),
                    probabilities=prob_dict,
                    timestamp=datetime.now().isoformat()
                )
            )

        return BatchPredictionResponse(
            predictions=prediction_results,
            total_products=len(batch.products),
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        logger.error(f"Batch prediction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.post("/analyze", response_model=AnalysisResponse)
async def multi_agent_analysis(request: AnalysisRequest):
    """
    Perform complete multi-agent analysis

    Coordinates Analyst, Strategist, and Reporter agents to provide
    comprehensive product lifecycle insights and recommendations
    """
    if not all([analyst_agent, strategist_agent, reporter_agent]):
        raise HTTPException(status_code=503, detail="Agents not initialized")

    try:
        # Convert to DataFrame
        products_data = [p.dict() for p in request.products]
        products_df = pd.DataFrame(products_data)

        # 1. Analyst Agent - Data Analysis & Predictions
        logger.info("Running Analyst Agent...")
        analyst_results = analyst_agent.analyze(products_df)

        # 2. Strategist Agent - Strategic Recommendations
        strategist_results = None
        if request.include_recommendations:
            logger.info("Running Strategist Agent...")
            strategist_results = strategist_agent.generate_recommendations(analyst_results)

        # 3. Reporter Agent - Comprehensive Report
        logger.info("Running Reporter Agent...")
        report = reporter_agent.generate_report(
            analyst_results=analyst_results,
            strategist_results=strategist_results
        )

        # Generate analysis ID
        analysis_id = f"ANALYSIS_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        return AnalysisResponse(
            analysis_id=analysis_id,
            timestamp=datetime.now().isoformat(),
            products_analyzed=len(request.products),
            analyst_insights=analyst_results,
            strategist_recommendations=strategist_results,
            executive_summary=report.get("executive_summary", {}),
            status="completed"
        )

    except Exception as e:
        logger.error(f"Multi-agent analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.get("/analyze/{analysis_id}")
async def get_analysis(analysis_id: str):
    """
    Retrieve a previous analysis by ID

    Note: This is a placeholder. In production, you would store
    analyses in a database and retrieve them here.
    """
    raise HTTPException(
        status_code=501,
        detail="Analysis retrieval not implemented. Store analyses in a database for this feature."
    )


@app.post("/report/export")
async def export_report(
    analysis_id: str = Query(..., description="Analysis ID to export"),
    format: str = Query("json", description="Export format: json, markdown, html")
):
    """
    Export analysis report in different formats

    Supports JSON, Markdown, and HTML formats
    """
    raise HTTPException(
        status_code=501,
        detail="Report export not implemented. Implement based on your storage solution."
    )

@app.get("/model/info")
async def model_info():
    """Get information about the loaded model"""
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Get feature importance safely
        feature_importance = []
        if model.is_trained:
            try:
                raw_importance = model.get_feature_importance()
                
                # Handle different return formats
                if isinstance(raw_importance, dict):
                    # If it's a dictionary: {feature: importance}
                    feature_importance = [
                        {"feature": str(feat), "importance": float(imp)}
                        for feat, imp in list(raw_importance.items())[:10]
                    ]
                elif isinstance(raw_importance, list):
                    # If it's a list of tuples/lists
                    feature_importance = []
                    for item in raw_importance[:10]:
                        if isinstance(item, (list, tuple)) and len(item) >= 2:
                            feature_importance.append({
                                "feature": str(item[0]),
                                "importance": float(item[1])
                            })
                        elif isinstance(item, dict):
                            feature_importance.append({
                                "feature": str(item.get("feature", "unknown")),
                                "importance": float(item.get("importance", 0.0))
                            })
                else:
                    logger.warning(f"Unexpected feature_importance format: {type(raw_importance)}")
                    feature_importance = []
            except Exception as e:
                logger.error(f"Error getting feature importance: {e}")
                feature_importance = []

        return {
            "model_type": model.model_type,
            "is_trained": model.is_trained,
            "feature_count": len(model.feature_names) if model.feature_names else 0,
            "top_features": feature_importance,
            "classes": ["DESCONTINUAR", "MANTER", "PROMOVER"],
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get model info: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/agents/status")
async def agents_status():
    """Get status of all agents"""
    return {
        "analyst_agent": {
            "initialized": analyst_agent is not None,
            "has_model": analyst_agent.model is not None if analyst_agent else False
        },
        "strategist_agent": {
            "initialized": strategist_agent is not None
        },
        "reporter_agent": {
            "initialized": reporter_agent is not None
        },
        "timestamp": datetime.now().isoformat()
    }


# ============================================================================
# Error Handlers
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "timestamp": datetime.now().isoformat()
        }
    )


#%%
# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Run the API server"""
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")

    logger.info(f"Starting server on {host}:{port}")

    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )


if __name__ == "__main__":
    main()