import streamlit as st
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Page configuration
st.set_page_config(
    page_title="NexoCommerce Multi-Agent System",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/zemarchezi/nexocommerce-ml-agents',
        'Report a bug': "https://github.com/zemarchezi/nexocommerce-ml-agents/issues",
        'About': "# NexoCommerce Multi-Agent System\nPowered by ML & AI Agents"
    }
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #666;
        text-align: center;
        margin-bottom: 3rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Main page
def main():
    # Header
    st.markdown('<div class="main-header">🛒 NexoCommerce Multi-Agent System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Intelligent Product Lifecycle Management powered by AI</div>', unsafe_allow_html=True)
    
    # Introduction
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🤖 Multi-Agent Architecture
        
        Three specialized AI agents working together:
        - **Analyst Agent**: Data analysis & ML predictions
        - **Strategist Agent**: Business recommendations
        - **Reporter Agent**: Executive reports
        """)
    
    with col2:
        st.markdown("""
        ### 📊 Key Features
        
        - Real-time product analysis
        - Batch processing capabilities
        - MLflow experiment tracking
        - Interactive visualizations
        - Export reports (JSON, Markdown, HTML)
        """)
    
    with col3:
        st.markdown("""
        ### 🎯 Business Value
        
        - Optimize product lifecycle
        - Increase revenue by 20-30%
        - Reduce inventory costs
        - Data-driven decision making
        - Automated insights generation
        """)
    
    st.markdown("---")
    
    # Quick Start Guide
    st.header("🚀 Quick Start Guide")
    
    with st.expander("📖 How to use this application", expanded=True):
        st.markdown("""
        ### Navigation
        
        Use the **sidebar** to navigate between different pages:
        
        1. **📊 Dashboard**: Overview of system metrics and recent analyses
        2. **🔍 Product Analysis**: Analyze individual products
        3. **📦 Batch Analysis**: Process multiple products at once
        4. **🤖 Multi-Agent Analysis**: Full multi-agent system analysis
        5. **📈 MLflow Tracking**: View ML experiments and model performance
        
        ### Getting Started
        
        1. Make sure the **FastAPI server** is running:
           ```bash
           cd src/api
           python main.py
           ```
        
        2. Navigate to any page from the sidebar
        3. Input product data or upload CSV files
        4. Click "Analyze" to get insights
        5. View results and export reports
        
        ### API Configuration
        
        The default API endpoint is `http://localhost:8000`. You can change this in the sidebar settings.
        """)
    
    # System Status
    st.header("🔧 System Status")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="API Status",
            value="🟢 Online",
            delta="Healthy"
        )
    
    with col2:
        st.metric(
            label="ML Model",
            value="✅ Loaded",
            delta="v1.0.0"
        )
    
    with col3:
        st.metric(
            label="Agents",
            value="3/3",
            delta="Active"
        )
    
    with col4:
        st.metric(
            label="MLflow",
            value="🟢 Running",
            delta="Tracking"
        )
    
    st.markdown("---")
    
    # Recent Activity
    st.header("📈 Recent Activity")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Latest Analyses")
        st.info("No recent analyses. Start by analyzing products!")
    
    with col2:
        st.subheader("System Metrics")
        st.info("System metrics will appear here after first analysis.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>NexoCommerce Multi-Agent System v1.0.0</p>
        <p>Built with ❤️ using Streamlit, FastAPI, and MLflow</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()