# app/streamlit_app.py
import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go
import sys
sys.path.append('..')

from src.pipeline.data_processing import generate_synthetic_data

st.set_page_config(page_title="NexoCommerce AI", layout="wide")

st.title("🛒 NexoCommerce - Sistema de Recomendação Inteligente")
st.markdown("### Análise de Ciclo de Vida de Produtos com Multi-Agentes")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configurações")
    n_products = st.slider("Número de produtos", 100, 1000, 500)
    
    if st.button("🔄 Gerar Dados Sintéticos"):
        st.session_state['data'] = generate_synthetic_data(n_products)
        st.success(f"✅ {n_products} produtos gerados!")

# Main content
if 'data' in st.session_state:
    df = st.session_state['data']
    
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🤖 Análise AI", "📈 Métricas", "🔍 Observabilidade"])
    
    with tab1:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total de Produtos", len(df))
        col2.metric("Receita Total", f"R$ {df['revenue'].sum():,.0f}")
        col3.metric("Estoque Total", f"{df['stock_quantity'].sum():,}")
        col4.metric("Rating Médio", f"{df['rating'].mean():.2f}⭐")
        
        # Gráficos
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(df, x='lifecycle_stage', 
                             title='Distribuição por Estágio',
                             labels={'lifecycle_stage': 'Estágio'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.scatter(df, x='views_last_30d', y='sales_last_30d',
                           color='category', size='revenue',
                           title='Visualizações vs Vendas')
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.header("🤖 Análise com Multi-Agentes")
        
        if st.button("▶️ Executar Análise"):
            with st.spinner("Processando com agentes..."):
                # Aqui você chamaria a API
                st.info("💡 Em produção, isso chamaria o endpoint /analyze da API")
                
                # Simulação
                st.success("✅ Análise concluída!")
                
                st.subheader("📋 Recomendações Estratégicas")
                
                st.markdown("#### 🚀 PROMOVER (Prioridade ALTA)")
                st.write("10 produtos identificados com alto potencial")
                
                st.markdown("#### ⚠️ DESCONTINUAR (Prioridade MÉDIA)")
                st.write("15 produtos com baixa performance")
                
                st.markdown("#### 📦 AUMENTAR ESTOQUE (Prioridade URGENTE)")
                st.write("5 produtos em risco de ruptura")
    
    with tab3:
        st.header("📈 Métricas do Modelo")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Acurácia", "0.87")
        col2.metric("ROC-AUC", "0.92")
        col3.metric("F1-Score", "0.85")
        
        st.subheader("Feature Importance")
        # Aqui você mostraria o gráfico real de feature importance
        
    with tab4:
        st.header("🔍 Observabilidade dos Agentes")
        
        metrics_data = {
            'Agente': ['Analyst', 'Strategist', 'Reporter'],
            'Tempo (s)': [0.45, 0.23, 0.12],
            'Itens Processados': [500, 25, 3]
        }
        
        st.dataframe(pd.DataFrame(metrics_data))

else:
    st.info("👈 Use o painel lateral para gerar dados sintéticos")