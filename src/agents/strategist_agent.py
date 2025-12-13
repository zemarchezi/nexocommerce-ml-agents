#%%

import logging
from typing import Dict, Any, List
from datetime import datetime

logger = logging.getLogger(__name__)


class StrategistAgent:
    """
    Strategist Agent - Business Strategy & Recommendations
    
    Responsibilities:
    - Generate actionable business recommendations
    - Prioritize actions based on impact and urgency
    - Create strategic plans for product lifecycle management
    - Estimate business impact of recommendations
    """
    
    def __init__(self):
        """Initialize Strategist Agent"""
        self.agent_name = "Strategist Agent"
        logger.info(f"{self.agent_name} initialized")
    
    def generate_recommendations(self, analyst_insights: Any) -> Dict[str, Any]:
        """
        Generate strategic recommendations based on analyst insights
        
        Args:
            analyst_insights: Results from AnalystAgent (can be dict or list)
            
        Returns:
            Strategic recommendations with priorities and action plans
        """
        logger.info(f"{self.agent_name}: Generating recommendations...")
        start_time = datetime.now()
        
        try:
            # Handle both dict and list inputs
            if isinstance(analyst_insights, list):
                # If it's a list of predictions, convert to expected format
                insights = {
                    "predictions": analyst_insights,
                    "total_products": len(analyst_insights),
                    "statistics": self._extract_stats_from_list(analyst_insights)
                }
            else:
                insights = analyst_insights
            
            recommendations = {
                "strategic_priorities": self._identify_priorities(insights),
                "action_plans": self._create_action_plans(insights),
                "risk_mitigation": self._assess_risks(insights),
                "expected_impact": self._estimate_impact(insights),
                "timeline": self._create_timeline(insights),
                "resource_allocation": self._recommend_resources(insights)
            }
            
            elapsed = (datetime.now() - start_time).total_seconds()
            logger.info(f"{self.agent_name}: Recommendations generated in {elapsed:.2f}s")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"{self.agent_name}: Failed to generate recommendations - {e}")
            raise
    
    def _extract_stats_from_list(self, predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract statistics from list of predictions"""
        if not predictions:
            return {}
        
        import pandas as pd
        df = pd.DataFrame(predictions)
        
        stats = {}
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        for col in numeric_cols:
            stats[col] = {
                "mean": float(df[col].mean()),
                "std": float(df[col].std()),
                "min": float(df[col].min()),
                "max": float(df[col].max())
            }
        
        return stats
    
    def _identify_priorities(self, insights: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify strategic priorities"""
        priorities = []
        
        predictions = insights.get("predictions", [])
        stats = insights.get("statistics", {})
        
        # Count actions from predictions list
        if isinstance(predictions, list):
            action_counts = {}
            for pred in predictions:
                action = pred.get("prediction", pred.get("lifecycle_action", "MANTER"))
                action_counts[action] = action_counts.get(action, 0) + 1
        else:
            action_counts = predictions.get("action_distribution", {})
        
        # Priority 1: Products to discontinue
        descontinuar_count = action_counts.get("DESCONTINUAR", 0)
        if descontinuar_count > 0:
            priorities.append({
                "priority": "HIGH",
                "category": "Cost Reduction",
                "action": "Discontinue underperforming products",
                "affected_products": descontinuar_count,
                "rationale": "Reduce inventory costs and focus resources on profitable items"
            })
        
        # Priority 2: Products to promote
        promover_count = action_counts.get("PROMOVER", 0)
        if promover_count > 0:
            priorities.append({
                "priority": "HIGH",
                "category": "Revenue Growth",
                "action": "Promote high-potential products",
                "affected_products": promover_count,
                "rationale": "Maximize revenue from products with strong performance indicators"
            })
        
        # Priority 3: Inventory optimization
        avg_stock = stats.get("stock_quantity", {}).get("mean", 0)
        if avg_stock > 100:
            priorities.append({
                "priority": "MEDIUM",
                "category": "Inventory Management",
                "action": "Optimize stock levels",
                "affected_products": insights.get("total_products", 0),
                "rationale": "Reduce holding costs while maintaining service levels"
            })
        
        return priorities
    
    def _create_action_plans(self, insights: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create detailed action plans"""
        action_plans = []
        
        predictions = insights.get("predictions", [])
        
        # Count actions from predictions list
        if isinstance(predictions, list):
            action_counts = {}
            for pred in predictions:
                action = pred.get("prediction", pred.get("lifecycle_action", "MANTER"))
                action_counts[action] = action_counts.get(action, 0) + 1
        else:
            action_counts = predictions.get("action_distribution", {})
        
        # Action plan for each lifecycle action
        for action, count in action_counts.items():
            if count > 0:
                plan = {
                    "action": action,
                    "products_count": count,
                    "steps": self._get_action_steps(action),
                    "estimated_duration": self._estimate_duration(action),
                    "required_resources": self._get_required_resources(action)
                }
                action_plans.append(plan)
        
        return action_plans
    
    def _get_action_steps(self, action: str) -> List[str]:
        """Get specific steps for each action"""
        steps_map = {
            "DESCONTINUAR": [
                "1. Notify stakeholders and customers",
                "2. Plan clearance sales or liquidation",
                "3. Update inventory systems",
                "4. Redirect marketing budget to other products",
                "5. Archive product data for analysis"
            ],
            "MANTER": [
                "1. Monitor performance metrics weekly",
                "2. Maintain current stock levels",
                "3. Continue standard marketing activities",
                "4. Review pricing strategy quarterly",
                "5. Gather customer feedback"
            ],
            "PROMOVER": [
                "1. Increase marketing budget allocation",
                "2. Launch targeted promotional campaigns",
                "3. Optimize product placement and visibility",
                "4. Increase stock levels to meet demand",
                "5. Consider bundle offers and cross-selling"
            ]
        }
        return steps_map.get(action, [])
    
    def _estimate_duration(self, action: str) -> str:
        """Estimate duration for action implementation"""
        duration_map = {
            "DESCONTINUAR": "4-6 weeks",
            "MANTER": "Ongoing",
            "PROMOVER": "2-4 weeks"
        }
        return duration_map.get(action, "Unknown")
    
    def _get_required_resources(self, action: str) -> List[str]:
        """Get required resources for action"""
        resources_map = {
            "DESCONTINUAR": [
                "Inventory management team",
                "Marketing team for clearance",
                "Finance team for write-offs"
            ],
            "MANTER": [
                "Product management team",
                "Customer service team"
            ],
            "PROMOVER": [
                "Marketing team",
                "Sales team",
                "Supply chain team",
                "Additional marketing budget"
            ]
        }
        return resources_map.get(action, [])
    
    def _assess_risks(self, insights: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Assess potential risks"""
        risks = []
        
        stats = insights.get("statistics", {})
        
        # High return rate risk
        avg_return_rate = stats.get("return_rate", {}).get("mean", 0)
        if avg_return_rate > 0.1:
            risks.append({
                "risk": "High Return Rate",
                "severity": "HIGH",
                "description": f"Average return rate of {avg_return_rate:.1%} may indicate quality issues",
                "mitigation": "Conduct quality audit and improve product descriptions"
            })
        
        # Low rating risk
        avg_rating = stats.get("rating", {}).get("mean", 0)
        if avg_rating < 3.5:
            risks.append({
                "risk": "Low Customer Satisfaction",
                "severity": "MEDIUM",
                "description": f"Average rating of {avg_rating:.1f} may hurt sales",
                "mitigation": "Improve product quality and customer service"
            })
        
        # Stock risk
        avg_stock = stats.get("stock_quantity", {}).get("mean", 0)
        if avg_stock < 20:
            risks.append({
                "risk": "Stock Shortage",
                "severity": "MEDIUM",
                "description": "Low stock levels may lead to stockouts",
                "mitigation": "Increase safety stock and improve forecasting"
            })
        
        return risks
    
    def _estimate_impact(self, insights: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate business impact of recommendations"""
        predictions = insights.get("predictions", [])
        stats = insights.get("statistics", {})
        
        total_products = insights.get("total_products", 0)
        avg_price = stats.get("price", {}).get("mean", 50)
        avg_sales = stats.get("sales_last_30d", {}).get("mean", 10)
        
        # Count actions
        if isinstance(predictions, list):
            action_counts = {}
            for pred in predictions:
                action = pred.get("prediction", pred.get("lifecycle_action", "MANTER"))
                action_counts[action] = action_counts.get(action, 0) + 1
        else:
            action_counts = predictions.get("action_distribution", {})
        
        # Estimate revenue impact
        promover_count = action_counts.get("PROMOVER", 0)
        estimated_revenue_increase = promover_count * avg_price * avg_sales * 0.2  # 20% increase
        
        # Estimate cost savings
        descontinuar_count = action_counts.get("DESCONTINUAR", 0)
        estimated_cost_savings = descontinuar_count * avg_price * 0.3  # 30% of product value
        
        return {
            "estimated_revenue_increase": round(estimated_revenue_increase, 2),
            "estimated_cost_savings": round(estimated_cost_savings, 2),
            "total_estimated_impact": round(estimated_revenue_increase + estimated_cost_savings, 2),
            "confidence_level": "MEDIUM",
            "timeframe": "3-6 months"
        }
    
    def _create_timeline(self, insights: Dict[str, Any]) -> Dict[str, List[str]]:
        """Create implementation timeline"""
        return {
            "immediate (0-2 weeks)": [
                "Analyze detailed product performance data",
                "Identify quick wins for promotion",
                "Start discontinuation process for worst performers"
            ],
            "short_term (2-8 weeks)": [
                "Launch promotional campaigns",
                "Complete product discontinuations",
                "Optimize inventory levels"
            ],
            "medium_term (2-6 months)": [
                "Monitor impact of changes",
                "Adjust strategies based on results",
                "Scale successful initiatives"
            ]
        }
    
    def _recommend_resources(self, insights: Dict[str, Any]) -> Dict[str, Any]:
        """Recommend resource allocation"""
        predictions = insights.get("predictions", [])
        
        # Count actions
        if isinstance(predictions, list):
            action_counts = {}
            for pred in predictions:
                action = pred.get("prediction", pred.get("lifecycle_action", "MANTER"))
                action_counts[action] = action_counts.get(action, 0) + 1
        else:
            action_counts = predictions.get("action_distribution", {})
        
        promover_count = action_counts.get("PROMOVER", 0)
        total_products = insights.get("total_products", 1)
        
        marketing_budget_increase = (promover_count / total_products) * 100 if total_products > 0 else 0
        
        return {
            "marketing_budget": {
                "recommendation": f"Increase by {marketing_budget_increase:.0f}%",
                "allocation": "Focus on products marked for promotion"
            },
            "inventory_investment": {
                "recommendation": "Increase for high-potential products",
                "allocation": "Reduce for products to discontinue"
            },
            "team_focus": {
                "recommendation": "Prioritize high-impact actions",
                "allocation": "Assign dedicated resources to promotional campaigns"
            }
        }
#%%

if __name__ == "__main__":
    # Test the Strategist Agent
    print("\n" + "="*80)
    print("STRATEGIST AGENT TEST")
    print("="*80)
    

    mock_results = {
        "predictions": [
            {
                "product_id": "PROD_001",
                "category": "Eletrônicos",
                "price": 299.90,
                "stock_quantity": 5,
                "sales_last_30d": 120,
                "prediction_label": "PROMOVER",
                "confidence": 0.92
            }
        ],
        "insights": [],
        "statistics": {}
    }
    #%%
    agent = StrategistAgent()
    results = agent.strategize(mock_results)
    
    print(f"\nRecommendations Generated: {len(results['recommendations'])}")
    print(f"Execution Time: {results['execution_time']:.2f}s")
# %%
