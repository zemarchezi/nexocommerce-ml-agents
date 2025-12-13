#%%
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AnalystAgent:
    """
    Analyst Agent - Performs quantitative analysis and ML predictions
    
    Responsibilities:
    - Load and validate product data
    - Generate ML predictions using trained model
    - Calculate confidence scores
    - Provide statistical insights
    - Track performance metrics
    """
    
    def __init__(self, model, processor):
        """
        Initialize Analyst Agent
        
        Args:
            model: Trained ProductLifecycleModel
            processor: DataProcessor instance
        """
        self.model = model
        self.processor = processor
        self.name = "Analyst Agent"
        self.version = "1.0.0"
        self.metrics = {
            "total_analyses": 0,
            "total_products_analyzed": 0,
            "average_confidence": 0.0,
            "execution_times": []
        }
    
    def analyze(self, products_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze products and generate predictions
        
        Args:
            products_df: DataFrame with product data
            
        Returns:
            Dictionary with analysis results
        """
        start_time = time.time()
        
        logger.info(f"{self.name}: Starting analysis of {len(products_df)} products")
        
        try:
            # Validate input
            self._validate_input(products_df)
            
            # Process data
            processed_df, features = self.processor.process_pipeline(
                products_df,
                is_training=False,
                create_target_var=False
            )
            
            # Generate predictions
            predictions_df = self.model.predict_with_confidence(processed_df[features])
            
            # Combine with original data
            results_df = pd.concat([
                products_df.reset_index(drop=True),
                predictions_df.reset_index(drop=True)
            ], axis=1)
            
            # Generate insights
            insights = self._generate_insights(results_df)
            
            # Calculate statistics
            statistics = self._calculate_statistics(results_df)
            
            # Update metrics
            execution_time = time.time() - start_time
            self._update_metrics(len(products_df), predictions_df["confidence"].mean(), execution_time)
            
            # Prepare response
            response = {
                "agent": self.name,
                "version": self.version,
                "timestamp": datetime.now().isoformat(),
                "products_analyzed": len(products_df),
                "predictions": results_df.to_dict(orient="records"),
                "insights": insights,
                "statistics": statistics,
                "execution_time": execution_time,
                "agent_metrics": self.get_metrics()
            }
            
            logger.info(f"{self.name}: Analysis completed in {execution_time:.2f}s")
            
            return response
            
        except Exception as e:
            logger.error(f"{self.name}: Error during analysis: {e}")
            raise
    
    def _validate_input(self, df: pd.DataFrame):
        """Validate input dataframe"""
        required_columns = [
            "product_id", "category", "price", "stock_quantity",
            "sales_last_30d", "views_last_30d", "rating"
        ]
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        if len(df) == 0:
            raise ValueError("Empty dataframe provided")
        
        logger.info(f"{self.name}: Input validation passed")
    
    def _generate_insights(self, results_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Generate business insights from predictions
        
        Args:
            results_df: DataFrame with predictions
            
        Returns:
            List of insights
        """
        insights = []
        
        # 1. Products to promote
        promote_products = results_df[results_df["prediction_label"] == "PROMOVER"]
        if len(promote_products) > 0:
            avg_confidence = promote_products["confidence"].mean()
            top_product = promote_products.nlargest(1, "confidence").iloc[0]
            
            insights.append({
                "type": "PROMOTE",
                "title": "Produtos com Alto Potencial",
                "description": f"Identificados {len(promote_products)} produtos recomendados para promoção",
                "count": len(promote_products),
                "average_confidence": float(avg_confidence),
                "top_product": {
                    "id": top_product["product_id"],
                    "name": top_product.get("product_name", "N/A"),
                    "confidence": float(top_product["confidence"]),
                    "category": top_product["category"]
                },
                "key_metrics": {
                    "avg_rating": float(promote_products["rating"].mean()),
                    "avg_sales": float(promote_products["sales_last_30d"].mean()),
                    "total_revenue_potential": float(
                        (promote_products["price"] * promote_products["sales_last_30d"]).sum()
                    )
                }
            })
        
        # 2. Products to discontinue
        discontinue_products = results_df[results_df["prediction_label"] == "DESCONTINUAR"]
        if len(discontinue_products) > 0:
            avg_confidence = discontinue_products["confidence"].mean()
            
            insights.append({
                "type": "DISCONTINUE",
                "title": "Produtos para Descontinuação",
                "description": f"Identificados {len(discontinue_products)} produtos com baixo desempenho",
                "count": len(discontinue_products),
                "average_confidence": float(avg_confidence),
                "key_metrics": {
                    "avg_rating": float(discontinue_products["rating"].mean()),
                    "avg_sales": float(discontinue_products["sales_last_30d"].mean()),
                    "avg_return_rate": float(discontinue_products.get("return_rate", pd.Series([0])).mean())
                },
                "potential_savings": "Redução de custos de estoque e operação"
            })
        
        # 3. Products to maintain
        maintain_products = results_df[results_df["prediction_label"] == "MANTER"]
        if len(maintain_products) > 0:
            insights.append({
                "type": "MAINTAIN",
                "title": "Produtos Estáveis",
                "description": f"{len(maintain_products)} produtos com desempenho estável",
                "count": len(maintain_products),
                "recommendation": "Manter estratégia atual e monitorar performance"
            })
        
        # 4. Low stock alert
        low_stock = results_df[
            (results_df["stock_quantity"] < 10) &
            (results_df["prediction_label"] == "PROMOVER")
        ]
        if len(low_stock) > 0:
            insights.append({
                "type": "ALERT",
                "title": "Alerta de Estoque Baixo",
                "description": f"{len(low_stock)} produtos de alto potencial com estoque baixo",
                "count": len(low_stock),
                "priority": "HIGH",
                "action_required": "Reabastecer estoque urgentemente",
                "products": low_stock["product_id"].tolist()[:5]
            })
        
        # 5. High confidence predictions
        high_confidence = results_df[results_df["confidence"] > 0.9]
        if len(high_confidence) > 0:
            insights.append({
                "type": "CONFIDENCE",
                "title": "Predições de Alta Confiança",
                "description": f"{len(high_confidence)} produtos com confiança > 90%",
                "count": len(high_confidence),
                "average_confidence": float(high_confidence["confidence"].mean()),
                "recommendation": "Priorizar ações nestes produtos"
            })
        
        # 6. Category analysis
        category_performance = results_df.groupby("category").agg({
            "prediction_label": lambda x: x.value_counts().to_dict(),
            "confidence": "mean",
            "sales_last_30d": "sum"
        }).to_dict(orient="index")
        
        best_category = max(
            category_performance.items(),
            key=lambda x: x[1]["sales_last_30d"]
        )
        
        insights.append({
            "type": "CATEGORY",
            "title": "Análise por Categoria",
            "description": f"Categoria com melhor desempenho: {best_category[0]}",
            "best_category": best_category[0],
            "total_sales": float(best_category[1]["sales_last_30d"]),
            "all_categories": {
                k: {
                    "avg_confidence": float(v["confidence"]),
                    "total_sales": float(v["sales_last_30d"])
                }
                for k, v in category_performance.items()
            }
        })
        
        return insights
    
    def _calculate_statistics(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate statistical metrics
        
        Args:
            results_df: DataFrame with predictions
            
        Returns:
            Dictionary with statistics
        """
        stats = {
            "total_products": len(results_df),
            "predictions_distribution": results_df["prediction_label"].value_counts().to_dict(),
            "confidence_metrics": {
                "mean": float(results_df["confidence"].mean()),
                "median": float(results_df["confidence"].median()),
                "min": float(results_df["confidence"].min()),
                "max": float(results_df["confidence"].max()),
                "std": float(results_df["confidence"].std())
            },
            "business_metrics": {
                "total_revenue_last_30d": float((results_df["price"] * results_df["sales_last_30d"]).sum()),
                "average_rating": float(results_df["rating"].mean()),
                "total_sales_last_30d": int(results_df["sales_last_30d"].sum()),
                "total_stock_value": float((results_df["price"] * results_df["stock_quantity"]).sum()),
                "products_out_of_stock": int((results_df["stock_quantity"] == 0).sum())
            },
            "category_breakdown": results_df.groupby("category").size().to_dict()
        }
        
        return stats
    
    def _update_metrics(self, n_products: int, avg_confidence: float, execution_time: float):
        """Update agent metrics"""
        self.metrics["total_analyses"] += 1
        self.metrics["total_products_analyzed"] += n_products
        self.metrics["execution_times"].append(execution_time)
        
        # Update average confidence (running average)
        total = self.metrics["total_analyses"]
        current_avg = self.metrics["average_confidence"]
        self.metrics["average_confidence"] = (
            (current_avg * (total - 1) + avg_confidence) / total
        )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get agent performance metrics"""
        execution_times = self.metrics["execution_times"]
        
        return {
            "total_analyses": self.metrics["total_analyses"],
            "total_products_analyzed": self.metrics["total_products_analyzed"],
            "average_confidence": self.metrics["average_confidence"],
            "performance": {
                "avg_execution_time": np.mean(execution_times) if execution_times else 0,
                "min_execution_time": np.min(execution_times) if execution_times else 0,
                "max_execution_time": np.max(execution_times) if execution_times else 0
            }
        }
    
    def reset_metrics(self):
        """Reset agent metrics"""
        self.metrics = {
            "total_analyses": 0,
            "total_products_analyzed": 0,
            "average_confidence": 0.0,
            "execution_times": []
        }
        logger.info(f"{self.name}: Metrics reset")

#%%
if __name__ == "__main__":

    import sys
    sys.path.append("/Users/jose/interviews_projects/nexocommerce-ml-agents/")
    from src.pipeline.data_loader import DataLoader
    from src.pipeline.data_processing import DataProcessor
    from src.models.product_model import ProductLifecycleModel
    #%%
    # Load and train model
    loader = DataLoader()
    df = loader.load_data(source="synthetic", n_samples=1000)
    #%%
    processor = DataProcessor()
    processed_df, features = processor.process_pipeline(df, is_training=True)
    
    X = processed_df[features]
    y = processed_df["lifecycle_action"]
    #%%
    model = ProductLifecycleModel(model_type="random_forest")
    model.train(X, y, log_mlflow=False)
    #%%
    # Create agent
    agent = AnalystAgent(model=model, processor=processor)
    #%%
    # Test analysis
    test_products = df.head(10)
    results = agent.analyze(test_products)
    
    print("\n" + "="*80)
    print(f"ANALYST AGENT RESULTS")
    print("="*80)
    print(f"\nProducts Analyzed: {results['products_analyzed']}")
    print(f"Execution Time: {results['execution_time']:.2f}s")
    print(f"\nInsights Generated: {len(results['insights'])}")
    for insight in results['insights']:
        print(f"\n- {insight['title']}: {insight['description']}")
# %%
