import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from typing import Dict, List, Any


def create_prediction_pie_chart(predictions: Dict[str, int]) -> go.Figure:
    """Create pie chart for prediction distribution"""
    
    colors = {
        "PROMOVER": "#2ecc71",
        "MANTER": "#3498db",
        "DESCONTINUAR": "#e74c3c"
    }
    
    fig = go.Figure(data=[go.Pie(
        labels=list(predictions.keys()),
        values=list(predictions.values()),
        marker=dict(colors=[colors.get(k, "#95a5a6") for k in predictions.keys()]),
        hole=0.4,
        textinfo='label+percent+value',
        textfont_size=14
    )])
    
    fig.update_layout(
        title="Distribuição de Ações Recomendadas",
        showlegend=True,
        height=400
    )
    
    return fig


def create_confidence_histogram(confidences: List[float]) -> go.Figure:
    """Create histogram of confidence scores"""
    
    fig = go.Figure(data=[go.Histogram(
        x=confidences,
        nbinsx=20,
        marker_color='#3498db',
        opacity=0.7
    )])
    
    fig.update_layout(
        title="Distribuição de Confiança das Predições",
        xaxis_title="Confiança",
        yaxis_title="Frequência",
        showlegend=False,
        height=400
    )
    
    return fig


def create_category_bar_chart(category_data: Dict[str, int]) -> go.Figure:
    """Create bar chart for category breakdown"""
    
    fig = go.Figure(data=[go.Bar(
        x=list(category_data.keys()),
        y=list(category_data.values()),
        marker_color='#9b59b6',
        text=list(category_data.values()),
        textposition='auto'
    )])
    
    fig.update_layout(
        title="Produtos por Categoria",
        xaxis_title="Categoria",
        yaxis_title="Número de Produtos",
        showlegend=False,
        height=400
    )
    
    return fig


def create_metrics_gauge(value: float, title: str, max_value: float = 1.0) -> go.Figure:
    """Create gauge chart for metrics"""
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title},
        gauge={
            'axis': {'range': [None, max_value]},
            'bar': {'color': "#3498db"},
            'steps': [
                {'range': [0, max_value*0.33], 'color': "#e74c3c"},
                {'range': [max_value*0.33, max_value*0.66], 'color': "#f39c12"},
                {'range': [max_value*0.66, max_value], 'color': "#2ecc71"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': max_value * 0.9
            }
        }
    ))
    
    fig.update_layout(height=300)
    
    return fig


def create_revenue_impact_chart(impact_data: Dict[str, float]) -> go.Figure:
    """Create chart for revenue impact"""
    
    fig = go.Figure(data=[go.Bar(
        x=list(impact_data.keys()),
        y=list(impact_data.values()),
        marker_color=['#2ecc71', '#3498db', '#9b59b6'],
        text=[f"R$ {v:,.2f}" for v in impact_data.values()],
        textposition='auto'
    )])
    
    fig.update_layout(
        title="Impacto Financeiro Estimado",
        xaxis_title="Métrica",
        yaxis_title="Valor (R$)",
        showlegend=False,
        height=400
    )
    
    return fig


def create_product_comparison_radar(products_df: pd.DataFrame) -> go.Figure:
    """Create radar chart for product comparison"""
    
    categories = ['Rating', 'Sales', 'Stock', 'Views', 'Confidence']
    
    fig = go.Figure()
    
    for idx, row in products_df.head(5).iterrows():
        fig.add_trace(go.Scatterpolar(
            r=[
                row.get('rating', 0) / 5,
                min(row.get('sales_last_30d', 0) / 100, 1),
                min(row.get('stock_quantity', 0) / 100, 1),
                min(row.get('views_last_30d', 0) / 1000, 1),
                row.get('confidence', 0)
            ],
            theta=categories,
            fill='toself',
            name=row.get('product_id', f'Product {idx}')
        ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        title="Comparação de Produtos (Top 5)",
        height=500
    )
    
    return fig


def create_sales_trend_chart(df: pd.DataFrame) -> go.Figure:
    """Create sales trend chart"""
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['sales_last_30d'],
        mode='lines+markers',
        name='Sales',
        line=dict(color='#3498db', width=2),
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        title="Tendência de Vendas (Últimos 30 dias)",
        xaxis_title="Produto",
        yaxis_title="Vendas",
        height=400,
        hovermode='x unified'
    )
    
    return fig