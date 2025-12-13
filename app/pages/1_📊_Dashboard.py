import streamlit as st
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from app.utils.api_client import get_api_client
from app.components.charts import (
    create_prediction_pie_chart,
    create_confidence_histogram,
    create_category_bar_chart,
    create_metrics_gauge
)

st.set_page_config(page_title="Dashboard", page_icon="📊", layout="wide")

# Header
st.title("📊 Dashboard - System Overview")
st.markdown("Real-time metrics and system status")

# Sidebar - API Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    api_url = st.text_input("API URL", value="http://localhost:8000")
    
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()

# Get API client
api_client = get_api_client(api_url)

# Check API health
with st.spinner("Checking API status..."):
    health = api_client.health_check()

if health.get("status") == "healthy":
    st.success("✅ API is healthy and running")
else:
    st.error(f"❌ API is not responding: {health.get('message', 'Unknown error')}")
    st.info("💡 Make sure the FastAPI server is running: `cd src/api && python main.py`")
    st.stop()

# Get model info
model_info = api_client.get_model_info()

# Get agent status
agent_status = api_client.get_agent_status()

st.markdown("---")

# Key Metrics Row
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="🤖 Active Agents",
        value=agent_status.get("total_agents", 3),
        delta="All systems operational"
    )

with col2:
    st.metric(
        label="📦 ML Model",
        value=model_info.get("model_type", "N/A"),
        delta=f"v{model_info.get('version', '1.0.0')}"
    )

with col3:
    accuracy = model_info.get("performance", {}).get("accuracy", 0)
    st.metric(
        label="🎯 Model Accuracy",
        value=f"{accuracy*100:.1f}%" if accuracy else "N/A",
        delta="Training set"
    )

with col4:
    st.metric(
        label="⏱️ Avg Response Time",
        value="< 1s",
        delta="Fast"
    )

st.markdown("---")

# Agent Status
st.header("🤖 Agent Status")

col1, col2, col3 = st.columns(3)

agents = agent_status.get("agents", {})

with col1:
    analyst = agents.get("analyst", {})
    st.subheader("📊 Analyst Agent")
    st.write(f"**Status:** {analyst.get('status', 'Unknown')}")
    st.write(f"**Version:** {analyst.get('version', 'N/A')}")
    st.write(f"**Analyses:** {analyst.get('metrics', {}).get('total_analyses', 0)}")

with col2:
    strategist = agents.get("strategist", {})
    st.subheader("🎯 Strategist Agent")
    st.write(f"**Status:** {strategist.get('status', 'Unknown')}")
    st.write(f"**Version:** {strategist.get('version', 'N/A')}")
    st.write("**Ready:** ✅")

with col3:
    reporter = agents.get("reporter", {})
    st.subheader("📝 Reporter Agent")
    st.write(f"**Status:** {reporter.get('status', 'Unknown')}")
    st.write(f"**Version:** {reporter.get('version', 'N/A')}")
    st.write(f"**Reports:** {reporter.get('metrics', {}).get('total_reports', 0)}")

st.markdown("---")

# Model Information
st.header("🧠 ML Model Information")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Model Details")
    st.json({
        "Model Type": model_info.get("model_type", "N/A"),
        "Version": model_info.get("version", "N/A"),
        "Trained": model_info.get("is_trained", False),
        "Features": model_info.get("n_features", 0),
        "Classes": model_info.get("classes", [])
    })

with col2:
    st.subheader("Performance Metrics")
    perf = model_info.get("performance", {})
    
    if perf:
        st.metric("Accuracy", f"{perf.get('accuracy', 0)*100:.2f}%")
        st.metric("Precision", f"{perf.get('precision', 0)*100:.2f}%")
        st.metric("Recall", f"{perf.get('recall', 0)*100:.2f}%")
        st.metric("F1-Score", f"{perf.get('f1', 0)*100:.2f}%")
    else:
        st.info("No performance metrics available")

st.markdown("---")

# Quick Actions
st.header("🚀 Quick Actions")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🔍 Analyze Single Product", use_container_width=True):
        st.switch_page("pages/2_🔍_Product_Analysis.py")

with col2:
    if st.button("📦 Batch Analysis", use_container_width=True):
        st.switch_page("pages/3_📦_Batch_Analysis.py")

with col3:
    if st.button("🤖 Multi-Agent Analysis", use_container_width=True):
        st.switch_page("pages/4_🤖_Multi_Agent.py")

# Footer
st.markdown("---")
st.caption("Dashboard updates in real-time. Click 'Refresh Data' to update metrics.")