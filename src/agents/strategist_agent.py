# src/agents/strategist_agent.py
import time
from datetime import datetime

class StrategistAgent:
    """Agente responsável por recomendações estratégicas de negócio"""
    
    def __init__(self):
        self.metrics = {
            'execution_time': 0,
            'recommendations_generated': 0,
            'timestamp': None
        }
    
    def generate_recommendations(self, df, insights):
        """Gera recomendações estratégicas baseadas nas análises"""
        start_time = time.time()
        
        recommendations = []
        
        # Produtos para promover
        to_promote = df[df['predicted_stage'] == 2].sort_values('confidence', ascending=False)
        if len(to_promote) > 0:
            top_promote = to_promote.head(10)
            recommendations.append({
                'action': 'PROMOVER',
                'priority': 'ALTA',
                'products': top_promote['product_id'].tolist(),
                'reason': f'Identificados {len(to_promote)} produtos com alto potencial de conversão',
                'expected_impact': 'Aumento de 20-30% nas vendas',
                'suggested_actions': [
                    'Aumentar investimento em marketing',
                    'Destacar na página principal',
                    'Oferecer descontos estratégicos'
                ]
            })
        
        # Produtos para descontinuar
        to_discontinue = df[df['predicted_stage'] == 0].sort_values('stock_quantity', ascending=False)
        if len(to_discontinue) > 0:
            recommendations.append({
                'action': 'DESCONTINUAR',
                'priority': 'MÉDIA',
                'products': to_discontinue.head(10)['product_id'].tolist(),
                'reason': f'{len(to_discontinue)} produtos com baixa performance e alto estoque',
                'expected_impact': 'Redução de 15% em custos de armazenamento',
                'suggested_actions': [
                    'Liquidação com descontos agressivos',
                    'Remover do catálogo principal',
                    'Não repor estoque'
                ]
            })
        
        # Produtos para aumentar estoque
        low_stock_high_demand = df[
            (df['predicted_stage'] == 2) & 
            (df['stock_quantity'] < 50)
        ]
        if len(low_stock_high_demand) > 0:
            recommendations.append({
                'action': 'AUMENTAR_ESTOQUE',
                'priority': 'URGENTE',
                'products': low_stock_high_demand['product_id'].tolist(),
                'reason': 'Produtos de alta demanda com risco de ruptura',
                'expected_impact': 'Evitar perda de vendas estimada em R$ 50k',
                'suggested_actions': [
                    'Reposição imediata de estoque',
                    'Negociar com fornecedores',
                    'Considerar aumento de preço'
                ]
            })
        
        self.metrics['execution_time'] = time.time() - start_time
        self.metrics['recommendations_generated'] = len(recommendations)
        self.metrics['timestamp'] = datetime.now().isoformat()
        
        return recommendations, self.metrics