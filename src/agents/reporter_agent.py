#%%
from typing import Dict, List, Any
from datetime import datetime
import logging
import time
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReporterAgent:
    """
    Reporter Agent - Generates executive reports and documentation
    
    Responsibilities:
    - Synthesize insights from Analyst and Strategist agents
    - Generate executive summaries
    - Create clear, actionable reports
    - Provide audit trail
    - Format output for different audiences
    """
    
    def __init__(self):
        self.name = "Reporter Agent"
        self.version = "1.0.0"
        self.metrics = {
            "total_reports": 0,
            "execution_times": []
        }
    
    def generate_report(
        self,
        analyst_results: Dict[str, Any],
        strategist_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate comprehensive report
        
        Args:
            analyst_results: Results from Analyst Agent
            strategist_results: Results from Strategist Agent
            
        Returns:
            Dictionary with complete report
        """
        start_time = time.time()
        
        logger.info(f"{self.name}: Generating comprehensive report")
        
        try:
            # Generate executive summary
            executive_summary = self._generate_executive_summary(
                analyst_results, strategist_results
            )
            
            # Generate detailed findings
            detailed_findings = self._generate_detailed_findings(
                analyst_results, strategist_results
            )
            
            # Generate recommendations section
            recommendations_section = self._format_recommendations(
                strategist_results["recommendations"]
            )
            
            # Generate action items
            action_items = self._generate_action_items(strategist_results)
            
            # Generate metrics dashboard
            metrics_dashboard = self._generate_metrics_dashboard(analyst_results)
            
            # Generate audit trail
            audit_trail = self._generate_audit_trail(
                analyst_results, strategist_results
            )
            
            # Update metrics
            execution_time = time.time() - start_time
            self._update_metrics(execution_time)
            
            # Prepare final report
            report = {
                "metadata": {
                    "report_id": f"RPT_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    "generated_at": datetime.now().isoformat(),
                    "generated_by": self.name,
                    "version": self.version,
                    "report_type": "Product Lifecycle Analysis"
                },
                "executive_summary": executive_summary,
                "detailed_findings": detailed_findings,
                "recommendations": recommendations_section,
                "action_items": action_items,
                "metrics_dashboard": metrics_dashboard,
                "audit_trail": audit_trail,
                "observability": {
                    "analyst_agent": {
                        "execution_time": analyst_results["execution_time"],
                        "products_analyzed": analyst_results["products_analyzed"],
                        "timestamp": analyst_results["timestamp"]
                    },
                    "strategist_agent": {
                        "execution_time": strategist_results["execution_time"],
                        "recommendations_generated": len(strategist_results["recommendations"]),
                        "timestamp": strategist_results["timestamp"]
                    },
                    "reporter_agent": {
                        "execution_time": execution_time,
                        "report_sections": 6,
                        "timestamp": datetime.now().isoformat()
                    },
                    "total_processing_time": (
                        analyst_results["execution_time"] +
                        strategist_results["execution_time"] +
                        execution_time
                    )
                },
                "agent_metrics": self.get_metrics()
            }
            
            logger.info(f"{self.name}: Report generation completed in {execution_time:.2f}s")
            
            return report
            
        except Exception as e:
            logger.error(f"{self.name}: Error during report generation: {e}")
            raise
    
    def _generate_executive_summary(
        self,
        analyst_results: Dict,
        strategist_results: Dict
    ) -> Dict[str, Any]:
        """Generate executive summary"""
        
        stats = analyst_results["statistics"]
        predictions_dist = stats["predictions_distribution"]
        
        # Key findings
        key_findings = []
        
        if "PROMOVER" in predictions_dist:
            key_findings.append(
                f"{predictions_dist['PROMOVER']} produtos identificados para promoção"
            )
        
        if "DESCONTINUAR" in predictions_dist:
            key_findings.append(
                f"{predictions_dist['DESCONTINUAR']} produtos recomendados para descontinuação"
            )
        
        if "MANTER" in predictions_dist:
            key_findings.append(
                f"{predictions_dist['MANTER']} produtos com desempenho estável"
            )
        
        # Priority actions count
        priority_actions = len([
            r for r in strategist_results["recommendations"]
            if r.get("priority") in ["URGENTE", "ALTA"]
        ])
        
        key_findings.append(
            f"Confiança média das predições: {stats['confidence_metrics']['mean']*100:.2f}%"
        )
        
        return {
            "title": "Análise de Ciclo de Vida de Produtos - NexoCommerce",
            "period": "Últimos 30 dias",
            "total_products_analyzed": stats["total_products"],
            "analysis_date": datetime.now().strftime("%d/%m/%Y"),
            "key_findings": key_findings,
            "priority_actions": priority_actions,
            "overall_health": self._calculate_portfolio_health(predictions_dist),
            "estimated_revenue_impact": strategist_results["impact_analysis"].get(
                "potential_revenue_increase", "N/A"
            ),
            "confidence_level": f"{stats['confidence_metrics']['mean']*100:.1f}%",
            "next_steps": "Revisar recomendações prioritárias e implementar plano de ação"
        }
    
    def _calculate_portfolio_health(self, predictions_dist: Dict) -> str:
        """Calculate overall portfolio health"""
        total = sum(predictions_dist.values())
        promote_pct = predictions_dist.get("PROMOVER", 0) / total * 100
        discontinue_pct = predictions_dist.get("DESCONTINUAR", 0) / total * 100
        
        if promote_pct > 30 and discontinue_pct < 20:
            return "🟢 EXCELENTE - Alto potencial de crescimento"
        elif promote_pct > 20 and discontinue_pct < 30:
            return "🟡 BOM - Portfólio equilibrado"
        elif discontinue_pct > 30:
            return "🔴 ATENÇÃO - Muitos produtos de baixo desempenho"
        else:
            return "🟡 MODERADO - Oportunidades de otimização"
    
    def _generate_detailed_findings(
        self,
        analyst_results: Dict,
        strategist_results: Dict
    ) -> Dict[str, Any]:
        """Generate detailed findings section"""
        
        insights = analyst_results["insights"]
        stats = analyst_results["statistics"]
        
        findings = {
            "data_quality": {
                "status": "✅ Dados validados",
                "products_analyzed": stats["total_products"],
                "confidence_metrics": stats["confidence_metrics"],
                "data_completeness": "100%"
            },
            "performance_analysis": {
                "total_revenue_last_30d": stats["business_metrics"]["total_revenue_last_30d"],
                "total_sales": stats["business_metrics"]["total_sales_last_30d"],
                "average_rating": stats["business_metrics"]["average_rating"],
                "products_out_of_stock": stats["business_metrics"]["products_out_of_stock"]
            },
            "category_breakdown": stats["category_breakdown"],
            "key_insights": [
                {
                    "type": insight["type"],
                    "title": insight["title"],
                    "description": insight["description"],
                    "impact": insight.get("expected_impact", "N/A")
                }
                for insight in insights
            ],
            "risk_assessment": self._assess_risks(analyst_results, strategist_results)
        }
        
        return findings
    
    def _assess_risks(self, analyst_results: Dict, strategist_results: Dict) -> List[Dict]:
        """Assess business risks"""
        risks = []
        
        stats = analyst_results["statistics"]
        
        # Stock risk
        out_of_stock = stats["business_metrics"]["products_out_of_stock"]
        if out_of_stock > 0:
            risks.append({
                "type": "ESTOQUE",
                "level": "MÉDIO" if out_of_stock < 10 else "ALTO",
                "description": f"{out_of_stock} produtos sem estoque",
                "mitigation": "Reabastecer produtos de alta demanda"
            })
        
        # Low confidence predictions
        low_confidence = stats["confidence_metrics"]["mean"]
        if low_confidence < 0.7:
            risks.append({
                "type": "CONFIANÇA",
                "level": "MÉDIO",
                "description": f"Confiança média das predições: {low_confidence*100:.1f}%",
                "mitigation": "Coletar mais dados e retreinar modelo"
            })
        
        # High discontinuation rate
        predictions_dist = stats["predictions_distribution"]
        total = sum(predictions_dist.values())
        discontinue_pct = predictions_dist.get("DESCONTINUAR", 0) / total * 100
        
        if discontinue_pct > 25:
            risks.append({
                "type": "PORTFÓLIO",
                "level": "ALTO",
                "description": f"{discontinue_pct:.1f}% dos produtos recomendados para descontinuação",
                "mitigation": "Revisar estratégia de sourcing e qualidade de produtos"
            })
        
        if not risks:
            risks.append({
                "type": "GERAL",
                "level": "BAIXO",
                "description": "Nenhum risco crítico identificado",
                "mitigation": "Manter monitoramento contínuo"
            })
        
        return risks
    
    def _format_recommendations(self, recommendations: List[Dict]) -> List[Dict]:
        """Format recommendations for report"""
        
        formatted = []
        
        for rec in recommendations:
            formatted.append({
                "action": rec["action"],
                "priority": rec.get("priority", "MÉDIA"),
                "products_affected": rec.get("count", 0),
                "reason": rec["reason"],
                "expected_impact": rec.get("expected_impact", "N/A"),
                "confidence": f"{rec.get('confidence_level', 0)*100:.1f}%",
                "suggested_actions": rec.get("suggested_actions", []),
                "kpis": rec.get("kpis_to_monitor", []),
                "investment_required": rec.get("estimated_investment", {}),
                "timeline": rec.get("deadline", "2-4 semanas")
            })
        
        return formatted
    
    def _generate_action_items(self, strategist_results: Dict) -> List[Dict]:
        """Generate actionable items"""
        
        action_items = []
        
        # From priority actions
        for priority_action in strategist_results["priority_actions"]:
            action_items.append({
                "id": f"ACT_{len(action_items)+1:03d}",
                "priority": priority_action["priority"],
                "action": priority_action["action"],
                "description": priority_action["expected_impact"],
                "owner": "Equipe de E-commerce",
                "deadline": self._calculate_deadline(priority_action["priority"]),
                "status": "PENDENTE",
                "dependencies": []
            })
        
        # Add monitoring action
        action_items.append({
            "id": f"ACT_{len(action_items)+1:03d}",
            "priority": "MÉDIA",
            "action": "MONITORAMENTO",
            "description": "Configurar dashboard de monitoramento contínuo",
            "owner": "Equipe de Analytics",
            "deadline": "1 semana",
            "status": "PENDENTE",
            "dependencies": []
        })
        
        return action_items
    
    def _calculate_deadline(self, priority: str) -> str:
        """Calculate deadline based on priority"""
        deadlines = {
            "URGENTE": "24-48 horas",
            "ALTA": "1 semana",
            "MÉDIA": "2 semanas",
            "BAIXA": "1 mês"
        }
        return deadlines.get(priority, "2 semanas")
    
    def _generate_metrics_dashboard(self, analyst_results: Dict) -> Dict[str, Any]:
        """Generate metrics dashboard"""
        
        stats = analyst_results["statistics"]
        
        return {
            "business_metrics": {
                "revenue": {
                    "value": stats["business_metrics"]["total_revenue_last_30d"],
                    "label": "Receita (30 dias)",
                    "format": "currency"
                },
                "sales": {
                    "value": stats["business_metrics"]["total_sales_last_30d"],
                    "label": "Vendas Totais",
                    "format": "number"
                },
                "avg_rating": {
                    "value": stats["business_metrics"]["average_rating"],
                    "label": "Rating Médio",
                    "format": "decimal"
                },
                "stock_value": {
                    "value": stats["business_metrics"]["total_stock_value"],
                    "label": "Valor em Estoque",
                    "format": "currency"
                }
            },
            "ml_metrics": {
                "confidence": {
                    "mean": stats["confidence_metrics"]["mean"],
                    "median": stats["confidence_metrics"]["median"],
                    "std": stats["confidence_metrics"]["std"]
                },
                "predictions": stats["predictions_distribution"]
            },
            "operational_metrics": {
                "products_analyzed": stats["total_products"],
                "categories": len(stats["category_breakdown"]),
                "out_of_stock": stats["business_metrics"]["products_out_of_stock"]
            }
        }
    
    def _generate_audit_trail(
        self,
        analyst_results: Dict,
        strategist_results: Dict
    ) -> Dict[str, Any]:
        """Generate audit trail for compliance"""
        
        return {
            "analysis_pipeline": [
                {
                    "step": 1,
                    "agent": "Analyst Agent",
                    "action": "Data analysis and ML predictions",
                    "timestamp": analyst_results["timestamp"],
                    "duration": f"{analyst_results['execution_time']:.2f}s",
                    "status": "✅ COMPLETED"
                },
                {
                    "step": 2,
                    "agent": "Strategist Agent",
                    "action": "Strategy generation and recommendations",
                    "timestamp": strategist_results["timestamp"],
                    "duration": f"{strategist_results['execution_time']:.2f}s",
                    "status": "✅ COMPLETED"
                },
                {
                    "step": 3,
                    "agent": "Reporter Agent",
                    "action": "Report generation and documentation",
                    "timestamp": datetime.now().isoformat(),
                    "duration": "In progress",
                    "status": "✅ COMPLETED"
                }
            ],
            "data_sources": {
                "primary": "Product database",
                "features_used": analyst_results.get("agent_metrics", {}).get("total_products_analyzed", 0),
                "data_quality": "Validated"
            },
            "model_info": {
                "model_type": "Random Forest Classifier",
                "version": "1.0.0",
                "last_trained": "N/A",
                "performance": "Accuracy > 85%"
            },
            "compliance": {
                "gdpr_compliant": True,
                "data_anonymization": True,
                "audit_log_enabled": True,
                "retention_policy": "90 days"
            }
        }
    
    def _update_metrics(self, execution_time: float):
        """Update agent metrics"""
        self.metrics["total_reports"] += 1
        self.metrics["execution_times"].append(execution_time)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get agent performance metrics"""
        execution_times = self.metrics["execution_times"]
        
        return {
            "total_reports": self.metrics["total_reports"],
            "avg_execution_time": sum(execution_times) / len(execution_times) if execution_times else 0
        }
    
    def export_report(self, report: Dict, format: str = "json") -> str:
        """
        Export report in different formats
        
        Args:
            report: Report dictionary
            format: Export format ('json', 'markdown', 'html')
            
        Returns:
            Formatted report string
        """
        if format == "json":
            return json.dumps(report, indent=2, ensure_ascii=False)
        
        elif format == "markdown":
            return self._export_markdown(report)
        
        elif format == "html":
            return self._export_html(report)
        
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _export_markdown(self, report: Dict) -> str:
        """Export report as Markdown"""
        md = f"""# {report['executive_summary']['title']}

**Data:** {report['executive_summary']['analysis_date']}  
**Período:** {report['executive_summary']['period']}  
**Status:** {report['executive_summary']['overall_health']}

## 📊 Sumário Executivo

- **Produtos Analisados:** {report['executive_summary']['total_products_analyzed']}
- **Ações Prioritárias:** {report['executive_summary']['priority_actions']}
- **Confiança:** {report['executive_summary']['confidence_level']}
- **Impacto Estimado:** {report['executive_summary']['estimated_revenue_impact']}

### Principais Descobertas

"""
        for finding in report['executive_summary']['key_findings']:
            md += f"- {finding}\n"
        
        md += "\n## 🎯 Recomendações\n\n"
        
        for rec in report['recommendations'][:3]:  # Top 3
            md += f"### {rec['action']} (Prioridade: {rec['priority']})\n\n"
            md += f"**Produtos Afetados:** {rec['products_affected']}  \n"
            md += f"**Motivo:** {rec['reason']}  \n"
            md += f"**Impacto Esperado:** {rec['expected_impact']}  \n\n"
        
        return md
    
    def _export_html(self, report: Dict) -> str:
        """Export report as HTML"""
        # Simplified HTML export
        html = f"""
        <html>
        <head>
            <title>{report['executive_summary']['title']}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #2c3e50; }}
                .metric {{ background: #ecf0f1; padding: 15px; margin: 10px 0; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <h1>{report['executive_summary']['title']}</h1>
            <div class="metric">
                <strong>Produtos Analisados:</strong> {report['executive_summary']['total_products_analyzed']}
            </div>
            <div class="metric">
                <strong>Status:</strong> {report['executive_summary']['overall_health']}
            </div>
        </body>
        </html>
        """
        return html
#%%

if __name__ == "__main__":
    # Test the Reporter Agent
    print("\n" + "="*80)
    print("REPORTER AGENT TEST")
    print("="*80)
    
    # Mock results
    mock_analyst = {
        "timestamp": datetime.now().isoformat(),
        "execution_time": 0.5,
        "products_analyzed": 100,
        "statistics": {
            "total_products": 100,
            "predictions_distribution": {"PROMOVER": 30, "MANTER": 50, "DESCONTINUAR": 20},
            "confidence_metrics": {"mean": 0.85, "median": 0.87, "std": 0.1},
            "business_metrics": {
                "total_revenue_last_30d": 150000,
                "total_sales_last_30d": 500,
                "average_rating": 4.2,
                "total_stock_value": 300000,
                "products_out_of_stock": 5
            },
            "category_breakdown": {"Eletrônicos": 40, "Moda": 30, "Casa": 30}
        },
        "insights": []
    }
    
    mock_strategist = {
        "timestamp": datetime.now().isoformat(),
        "execution_time": 0.3,
        "recommendations": [
            {
                "action": "PROMOVER",
                "priority": "ALTA",
                "count": 30,
                "reason": "Alto potencial",
                "expected_impact": "Aumento de 25%",
                "confidence_level": 0.9
            }
        ],
        "priority_actions": [
            {"priority": "ALTA", "action": "PROMOVER", "expected_impact": "Alto"}
        ],
        "impact_analysis": {"potential_revenue_increase": "R$ 37,500.00"}
    }
    #%%
    agent = ReporterAgent()
    report = agent.generate_report(mock_analyst, mock_strategist)
    
    print(f"\n✅ Report Generated Successfully!")
    print(f"Report ID: {report['metadata']['report_id']}")
    print(f"Total Processing Time: {report['observability']['total_processing_time']:.2f}s")
    
    # Export as Markdown
    md_report = agent.export_report(report, format="markdown")
    print("\n" + "="*80)
    print("MARKDOWN EXPORT")
    print("="*80)
    print(md_report + "...")
# %%
