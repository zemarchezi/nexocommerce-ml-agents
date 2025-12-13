#%%

import pandas as pd
from typing import Dict, List, Any
from datetime import datetime
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StrategistAgent:
    """
    Strategist Agent - Generates business strategies and recommendations
    
    Responsibilities:
    - Analyze insights from Analyst Agent
    - Generate actionable recommendations
    - Prioritize actions based on business impact
    - Estimate financial impact
    - Create strategic action plans
    """
    
    def __init__(self):
        self.name = "Strategist Agent"
        self.version = "1.0.0"
        self.metrics = {
            "total_strategies": 0,
            "total_recommendations": 0,
            "execution_times": []
        }
    
    def strategize(self, analyst_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate strategic recommendations based on analyst insights
        
        Args:
            analyst_results: Results from Analyst Agent
            
        Returns:
            Dictionary with strategic recommendations
        """
        start_time = time.time()
        
        logger.info(f"{self.name}: Generating strategic recommendations")
        
        try:
            predictions = pd.DataFrame(analyst_results["predictions"])
            insights = analyst_results["insights"]
            statistics = analyst_results["statistics"]
            
            # Generate recommendations
            recommendations = self._generate_recommendations(predictions, insights, statistics)
            
            # Prioritize actions
            priority_actions = self._prioritize_actions(recommendations)
            
            # Estimate impact
            impact_analysis = self._estimate_impact(predictions, recommendations)
            
            # Generate action plan
            action_plan = self._create_action_plan(recommendations, priority_actions)
            
            # Update metrics
            execution_time = time.time() - start_time
            self._update_metrics(len(recommendations), execution_time)
            
            # Prepare response
            response = {
                "agent": self.name,
                "version": self.version,
                "timestamp": datetime.now().isoformat(),
                "recommendations": recommendations,
                "priority_actions": priority_actions,
                "impact_analysis": impact_analysis,
                "action_plan": action_plan,
                "execution_time": execution_time,
                "agent_metrics": self.get_metrics()
            }
            
            logger.info(f"{self.name}: Strategy generation completed in {execution_time:.2f}s")
            
            return response
            
        except Exception as e:
            logger.error(f"{self.name}: Error during strategy generation: {e}")
            raise
    
    def _generate_recommendations(
        self,
        predictions: pd.DataFrame,
        insights: List[Dict],
        statistics: Dict
    ) -> List[Dict[str, Any]]:
        """Generate business recommendations"""
        recommendations = []
        
        # 1. PROMOTE recommendations
        promote_products = predictions[predictions["prediction_label"] == "PROMOVER"]
        if len(promote_products) > 0:
            high_confidence_promote = promote_products[promote_products["confidence"] > 0.85]
            
            recommendations.append({
                "action": "PROMOVER",
                "priority": self._calculate_priority(len(high_confidence_promote), "promote"),
                "products": promote_products["product_id"].tolist(),
                "high_confidence_products": high_confidence_promote["product_id"].tolist(),
                "count": len(promote_products),
                "reason": f"Identificados {len(promote_products)} produtos com alto potencial de conversão e vendas",
                "expected_impact": "Aumento de 20-30% nas vendas destes produtos",
                "confidence_level": float(promote_products["confidence"].mean()),
                "suggested_actions": [
                    "Aumentar investimento em marketing digital (Google Ads, Facebook Ads)",
                    "Destacar produtos na página principal e categorias",
                    "Criar campanhas de email marketing segmentadas",
                    "Oferecer descontos estratégicos (5-15%) para impulsionar vendas",
                    "Aumentar estoque preventivamente",
                    "Criar bundles com produtos complementares"
                ],
                "kpis_to_monitor": [
                    "Taxa de conversão",
                    "ROI de marketing",
                    "Ticket médio",
                    "Velocidade de vendas"
                ],
                "estimated_investment": self._estimate_investment(promote_products, "promote"),
                "estimated_revenue_increase": self._estimate_revenue_increase(promote_products)
            })
        
        # 2. DISCONTINUE recommendations
        discontinue_products = predictions[predictions["prediction_label"] == "DESCONTINUAR"]
        if len(discontinue_products) > 0:
            recommendations.append({
                "action": "DESCONTINUAR",
                "priority": self._calculate_priority(len(discontinue_products), "discontinue"),
                "products": discontinue_products["product_id"].tolist(),
                "count": len(discontinue_products),
                "reason": f"Identificados {len(discontinue_products)} produtos com baixo desempenho e alta taxa de retorno",
                "expected_impact": "Redução de custos operacionais e de estoque em 15-25%",
                "confidence_level": float(discontinue_products["confidence"].mean()),
                "suggested_actions": [
                    "Realizar liquidação com descontos agressivos (30-50%)",
                    "Interromper reabastecimento de estoque",
                    "Remover produtos das campanhas de marketing",
                    "Analisar feedback de clientes para entender problemas",
                    "Considerar substituição por produtos similares de melhor qualidade",
                    "Negociar devolução com fornecedores se possível"
                ],
                "kpis_to_monitor": [
                    "Taxa de liquidação",
                    "Redução de custos de armazenagem",
                    "Feedback de clientes",
                    "Tempo até estoque zero"
                ],
                "estimated_cost_savings": self._estimate_cost_savings(discontinue_products),
                "liquidation_strategy": {
                    "phase_1": "Desconto de 30% por 2 semanas",
                    "phase_2": "Desconto de 50% por 2 semanas",
                    "phase_3": "Desconto de 70% até liquidação total"
                }
            })
        
        # 3. MAINTAIN recommendations
        maintain_products = predictions[predictions["prediction_label"] == "MANTER"]
        if len(maintain_products) > 0:
            recommendations.append({
                "action": "MANTER",
                "priority": "MÉDIA",
                "products": maintain_products["product_id"].tolist(),
                "count": len(maintain_products),
                "reason": f"{len(maintain_products)} produtos com desempenho estável e consistente",
                "expected_impact": "Manutenção da receita atual com otimizações incrementais",
                "confidence_level": float(maintain_products["confidence"].mean()),
                "suggested_actions": [
                    "Manter estratégia de marketing atual",
                    "Monitorar performance semanalmente",
                    "Otimizar descrições e imagens dos produtos",
                    "Coletar mais reviews de clientes",
                    "Testar pequenas variações de preço (A/B testing)",
                    "Garantir disponibilidade de estoque"
                ],
                "kpis_to_monitor": [
                    "Estabilidade de vendas",
                    "Satisfação do cliente",
                    "Margem de lucro",
                    "Giro de estoque"
                ],
                "optimization_opportunities": self._identify_optimization_opportunities(maintain_products)
            })
        
        # 4. URGENT ACTIONS (low stock + high potential)
        urgent_products = predictions[
            (predictions["stock_quantity"] < 10) &
            (predictions["prediction_label"] == "PROMOVER") &
            (predictions["confidence"] > 0.8)
        ]
        
        if len(urgent_products) > 0:
            recommendations.append({
                "action": "AÇÃO_URGENTE",
                "priority": "URGENTE",
                "products": urgent_products["product_id"].tolist(),
                "count": len(urgent_products),
                "reason": f"{len(urgent_products)} produtos de alto potencial com estoque crítico",
                "expected_impact": "Evitar perda de vendas por falta de estoque (potencial perda de R$ 50k-100k)",
                "confidence_level": float(urgent_products["confidence"].mean()),
                "suggested_actions": [
                    "🚨 URGENTE: Reabastecer estoque imediatamente",
                    "Contatar fornecedores para entrega expressa",
                    "Considerar fornecedores alternativos",
                    "Ativar notificação de 'volta ao estoque' para clientes",
                    "Pausar campanhas de marketing até reabastecimento"
                ],
                "deadline": "24-48 horas",
                "risk_level": "ALTO",
                "potential_revenue_loss": self._calculate_revenue_loss(urgent_products)
            })
        
        # 5. CATEGORY-SPECIFIC recommendations
        category_recommendations = self._generate_category_recommendations(predictions)
        if category_recommendations:
            recommendations.extend(category_recommendations)
        
        return recommendations
    
    def _calculate_priority(self, count: int, action_type: str) -> str:
        """Calculate priority level"""
        if action_type == "promote":
            if count > 20:
                return "ALTA"
            elif count > 10:
                return "MÉDIA"
            else:
                return "BAIXA"
        elif action_type == "discontinue":
            if count > 30:
                return "ALTA"
            elif count > 15:
                return "MÉDIA"
            else:
                return "BAIXA"
        return "MÉDIA"
    
    def _estimate_investment(self, products: pd.DataFrame, action: str) -> Dict[str, Any]:
        """Estimate required investment"""
        if action == "promote":
            n_products = len(products)
            marketing_budget = n_products * 500  # R$ 500 per product
            inventory_investment = (products["price"] * 50).sum()  # 50 units per product
            
            return {
                "marketing_budget": f"R$ {marketing_budget:,.2f}",
                "inventory_investment": f"R$ {inventory_investment:,.2f}",
                "total_investment": f"R$ {marketing_budget + inventory_investment:,.2f}",
                "investment_per_product": f"R$ {(marketing_budget + inventory_investment) / n_products:,.2f}"
            }
        return {}
    
    def _estimate_revenue_increase(self, products: pd.DataFrame) -> Dict[str, Any]:
        """Estimate potential revenue increase"""
        current_revenue = (products["price"] * products["sales_last_30d"]).sum()
        
        # Conservative estimate: 20% increase
        conservative = current_revenue * 1.20
        # Optimistic estimate: 30% increase
        optimistic = current_revenue * 1.30
        
        return {
            "current_monthly_revenue": f"R$ {current_revenue:,.2f}",
            "conservative_estimate": f"R$ {conservative:,.2f} (+20%)",
            "optimistic_estimate": f"R$ {optimistic:,.2f} (+30%)",
            "potential_gain": f"R$ {conservative - current_revenue:,.2f} - R$ {optimistic - current_revenue:,.2f}"
        }
    
    def _estimate_cost_savings(self, products: pd.DataFrame) -> Dict[str, Any]:
        """Estimate cost savings from discontinuation"""
        inventory_value = (products["price"] * products["stock_quantity"]).sum()
        monthly_holding_cost = inventory_value * 0.02  # 2% monthly holding cost
        
        return {
            "inventory_value_to_liquidate": f"R$ {inventory_value:,.2f}",
            "monthly_holding_cost_savings": f"R$ {monthly_holding_cost:,.2f}",
            "annual_savings": f"R$ {monthly_holding_cost * 12:,.2f}"
        }
    
    def _calculate_revenue_loss(self, products: pd.DataFrame) -> str:
        """Calculate potential revenue loss"""
        daily_sales = products["sales_last_30d"] / 30
        daily_revenue = (daily_sales * products["price"]).sum()
        weekly_loss = daily_revenue * 7
        
        return f"R$ {weekly_loss:,.2f} por semana"
    
    def _identify_optimization_opportunities(self, products: pd.DataFrame) -> List[str]:
        """Identify optimization opportunities for maintain products"""
        opportunities = []
        
        # Low rating products
        low_rating = products[products["rating"] < 4.0]
        if len(low_rating) > 0:
            opportunities.append(
                f"{len(low_rating)} produtos com rating < 4.0 - melhorar qualidade/descrição"
            )
        
        # High return rate
        if "return_rate" in products.columns:
            high_return = products[products["return_rate"] > 0.1]
            if len(high_return) > 0:
                opportunities.append(
                    f"{len(high_return)} produtos com alta taxa de retorno - investigar causas"
                )
        
        # Low conversion
        if "conversion_rate" in products.columns:
            low_conversion = products[products["conversion_rate"] < 0.02]
            if len(low_conversion) > 0:
                opportunities.append(
                    f"{len(low_conversion)} produtos com baixa conversão - otimizar página do produto"
                )
        
        return opportunities if opportunities else ["Nenhuma oportunidade crítica identificada"]
    
    def _generate_category_recommendations(self, predictions: pd.DataFrame) -> List[Dict]:
        """Generate category-specific recommendations"""
        recommendations = []
        
        category_analysis = predictions.groupby("category").agg({
            "prediction_label": lambda x: (x == "PROMOVER").sum(),
            "sales_last_30d": "sum",
            "confidence": "mean"
        }).reset_index()
        
        category_analysis.columns = ["category", "promote_count", "total_sales", "avg_confidence"]
        
        # Top performing category
        top_category = category_analysis.nlargest(1, "total_sales").iloc[0]
        
        if top_category["promote_count"] > 0:
            recommendations.append({
                "action": "FOCO_CATEGORIA",
                "priority": "ALTA",
                "category": top_category["category"],
                "reason": f"Categoria '{top_category['category']}' apresenta melhor desempenho",
                "suggested_actions": [
                    f"Expandir portfólio na categoria {top_category['category']}",
                    "Criar landing page dedicada para a categoria",
                    "Investir em SEO para palavras-chave da categoria",
                    "Negociar melhores condições com fornecedores"
                ],
                "metrics": {
                    "total_sales": int(top_category["total_sales"]),
                    "products_to_promote": int(top_category["promote_count"]),
                    "avg_confidence": float(top_category["avg_confidence"])
                }
            })
        
        return recommendations
    
    def _prioritize_actions(self, recommendations: List[Dict]) -> List[Dict]:
        """Prioritize actions based on urgency and impact"""
        priority_order = {"URGENTE": 0, "ALTA": 1, "MÉDIA": 2, "BAIXA": 3}
        
        prioritized = sorted(
            recommendations,
            key=lambda x: priority_order.get(x.get("priority", "MÉDIA"), 2)
        )
        
        return [
            {
                "rank": i + 1,
                "action": rec["action"],
                "priority": rec.get("priority", "MÉDIA"),
                "products_affected": rec.get("count", 0),
                "expected_impact": rec.get("expected_impact", "N/A")
            }
            for i, rec in enumerate(prioritized[:5])  # Top 5
        ]
    
    def _estimate_impact(
        self,
        predictions: pd.DataFrame,
        recommendations: List[Dict]
    ) -> Dict[str, Any]:
        """Estimate overall business impact"""
        
        total_revenue = (predictions["price"] * predictions["sales_last_30d"]).sum()
        
        # Calculate potential impact
        promote_products = predictions[predictions["prediction_label"] == "PROMOVER"]
        promote_revenue = (promote_products["price"] * promote_products["sales_last_30d"]).sum()
        potential_increase = promote_revenue * 0.25  # 25% increase
        
        return {
            "current_monthly_revenue": f"R$ {total_revenue:,.2f}",
            "potential_revenue_increase": f"R$ {potential_increase:,.2f}",
            "estimated_new_revenue": f"R$ {total_revenue + potential_increase:,.2f}",
            "roi_estimate": "150-200% em 3 meses",
            "confidence": "Alta (baseado em {:.1f}% confiança média)".format(
                predictions["confidence"].mean() * 100
            ),
            "timeline": {
                "short_term": "Resultados visíveis em 2-4 semanas",
                "medium_term": "ROI positivo em 2-3 meses",
                "long_term": "Otimização contínua do portfólio"
            }
        }
    
    def _create_action_plan(
        self,
        recommendations: List[Dict],
        priority_actions: List[Dict]
    ) -> Dict[str, Any]:
        """Create detailed action plan"""
        
        return {
            "immediate_actions": [
                action for action in priority_actions
                if action["priority"] in ["URGENTE", "ALTA"]
            ],
            "short_term_actions": [
                {
                    "timeframe": "Próximas 2 semanas",
                    "actions": [
                        "Implementar campanhas de marketing para produtos PROMOVER",
                        "Iniciar liquidação de produtos DESCONTINUAR",
                        "Reabastecer produtos com estoque crítico"
                    ]
                }
            ],
            "medium_term_actions": [
                {
                    "timeframe": "Próximo mês",
                    "actions": [
                        "Avaliar resultados das campanhas",
                        "Ajustar estratégias baseado em performance",
                        "Expandir ações para produtos MANTER"
                    ]
                }
            ],
            "monitoring_plan": {
                "frequency": "Semanal",
                "key_metrics": [
                    "Vendas por produto",
                    "Taxa de conversão",
                    "ROI de marketing",
                    "Nível de estoque",
                    "Satisfação do cliente"
                ],
                "review_meetings": "Reunião quinzenal com stakeholders"
            }
        }
    
    def _update_metrics(self, n_recommendations: int, execution_time: float):
        """Update agent metrics"""
        self.metrics["total_strategies"] += 1
        self.metrics["total_recommendations"] += n_recommendations
        self.metrics["execution_times"].append(execution_time)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get agent performance metrics"""
        execution_times = self.metrics["execution_times"]
        
        return {
            "total_strategies": self.metrics["total_strategies"],
            "total_recommendations": self.metrics["total_recommendations"],
            "avg_execution_time": sum(execution_times) / len(execution_times) if execution_times else 0
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
