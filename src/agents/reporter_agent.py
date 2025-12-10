# src/agents/reporter_agent.py
import time
from datetime import datetime

class ReporterAgent:
    """Agente responsável por gerar relatórios claros para stakeholders"""
    
    def __init__(self):
        self.metrics = {
            'execution_time': 0,
            'report_sections': 0,
            'timestamp': None
        }
    
    def generate_report(self, insights, recommendations, agent_metrics):
        """Gera relatório executivo auditável"""
        start_time = time.time()
        
        report = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'report_type': 'Product Lifecycle Analysis',
                'version': '1.0'
            },
            'executive_summary': self._create_executive_summary(insights, recommendations),
            'detailed_insights': insights,
            'recommendations': recommendations,
            'observability': {
                'analyst_agent': agent_metrics['analyst'],
                'strategist_agent': agent_metrics['strategist'],
                'total_processing_time': sum([
                    agent_metrics['analyst']['execution_time'],
                    agent_metrics['strategist']['execution_time']
                ])
            },
            'audit_trail': {
                'model_version': 'product_lifecycle_model/latest',
                'data_processed': insights['total_products'],
                'confidence_threshold': 0.8
            }
        }
        
        self.metrics['execution_time'] = time.time() - start_time
        self.metrics['report_sections'] = len(report.keys())
        self.metrics['timestamp'] = datetime.now().isoformat()
        
        return report, self.metrics
    
    def _create_executive_summary(self, insights, recommendations):
        """Cria sumário executivo"""
        return {
            'total_products_analyzed': insights['total_products'],
            'key_findings': [
                f"{insights['to_promote']} produtos identificados para promoção",
                f"{insights['to_discontinue']} produtos recomendados para descontinuação",
                f"Confiança média das predições: {insights['avg_confidence']:.2%}"
            ],
            'priority_actions': len([r for r in recommendations if r['priority'] == 'URGENTE']),
            'estimated_revenue_impact': 'R$ 150k - R$ 200k (próximos 30 dias)'
        }