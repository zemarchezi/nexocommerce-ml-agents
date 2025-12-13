"""
Multi-Agent Analysis - Full multi-agent system analysis
"""

import streamlit as st
import sys
from pathlib import Path
import pandas as pd
import json

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from app.utils.api_client import get_api_client
from app.utils.data_utils import csv_to_products, get_action_emoji, format_currency
from app.components.charts import (
    create_prediction_pie_chart,
    create_confidence_histogram,
    create_revenue_impact_chart
)

st.set_page_config(page_title="Multi-Agent Analysis", page_icon="🤖", layout="wide")

# Header
st.title("🤖 Multi-Agent Analysis")
st.markdown("Complete analysis with Analyst, Strategist, and Reporter agents working together")

# Sidebar - API Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    api_url = st.text_input("API URL", value="http://localhost:8000")
    
    st.markdown("---")
    
    st.header("🤖 Agent Pipeline")
    st.info("""
    **1. 📊 Analyst Agent**
    - Data analysis
    - ML predictions
    - Statistical insights
    
    **2. 🎯 Strategist Agent**
    - Business recommendations
    - Action prioritization
    - Impact estimation
    
    **3. 📝 Reporter Agent**
    - Executive summary
    - Comprehensive report
    - Export options
    """)
    
    st.markdown("---")
    
    st.header("📊 Sample Data")
    if st.button("Generate Sample CSV", use_container_width=True):
        sample_data = pd.DataFrame([
            {
                "product_id": f"PROD_{i:03d}",
                "price": 50 + i * 10,
                "rating": 3.5 + (i % 3) * 0.5,
                "num_reviews": 50 + i * 20,
                "stock_quantity": 100 - i * 5,
                "sales_last_30d": 20 + i * 3,
                "views_last_30d": 300 + i * 50,
                "category": ["Electronics", "Clothing", "Home"][i % 3],
                "brand": f"Brand_{chr(65 + i % 5)}",
                "days_since_launch": 100 + i * 20,
                "discount_percentage": i * 2,
                "return_rate": 0.03 + i * 0.01
            }
            for i in range(10)
        ])
        
        csv = sample_data.to_csv(index=False)
        st.download_button(
            label="📥 Download Sample CSV",
            data=csv,
            file_name="sample_products.csv",
            mime="text/csv",
            use_container_width=True
        )

# Get API client
api_client = get_api_client(api_url)

# Check API health
health = api_client.health_check()
if health.get("status") != "healthy":
    st.error(f"❌ API is not responding: {health.get('message', 'Unknown error')}")
    st.stop()

st.markdown("---")

# File Upload
st.subheader("📁 Upload Product Data")

uploaded_file = st.file_uploader(
    "Upload CSV file with product data",
    type=["csv"],
    help="CSV file should contain product information for multi-agent analysis"
)

if uploaded_file is not None:
    try:
        # Read CSV
        df = pd.read_csv(uploaded_file)
        
        st.success(f"✅ Loaded {len(df)} products from CSV")
        
        # Display preview
        with st.expander("👀 Preview Data", expanded=False):
            st.dataframe(df.head(10), use_container_width=True)
        
        st.markdown("---")
        
        # Analyze Button
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col2:
            analyze_button = st.button("🚀 Run Multi-Agent Analysis", use_container_width=True, type="primary")
        
        if analyze_button:
            # Convert to products list
            products = csv_to_products(df)
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Run multi-agent analysis
            status_text.text("🔄 Starting multi-agent analysis...")
            progress_bar.progress(10)
            
            with st.spinner("🤖 Agents are working..."):
                result = api_client.multi_agent_analysis(products)
            
            progress_bar.progress(100)
            status_text.text("✅ Analysis complete!")
            
            if "error" in result:
                st.error(f"❌ Error: {result['error']}")
            else:
                st.success("✅ Multi-agent analysis completed successfully!")
                
                st.markdown("---")
                
                # Analysis Overview
                st.header("📊 Analysis Overview")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        label="Analysis ID",
                        value=result.get("analysis_id", "N/A")[:15] + "..."
                    )
                
                with col2:
                    st.metric(
                        label="Products Analyzed",
                        value=result.get("products_analyzed", 0)
                    )
                
                with col3:
                    st.metric(
                        label="Status",
                        value=result.get("status", "N/A").upper()
                    )
                
                with col4:
                    st.metric(
                        label="Timestamp",
                        value=result.get("timestamp", "N/A")[:10]
                    )
                
                st.markdown("---")
                
                # Analyst Insights
                st.header("📊 Analyst Agent Insights")
                
                analyst_insights = result.get("analyst_insights", {})
                
                if analyst_insights:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("🔍 Key Insights")
                        insights = analyst_insights.get("insights", [])
                        if insights:
                            for insight in insights[:5]:
                                st.info(f"💡 {insight}")
                        else:
                            st.write("No insights available")
                    
                    with col2:
                        st.subheader("📈 Statistics")
                        stats = analyst_insights.get("statistics", {})
                        if stats:
                            st.json(stats)
                        else:
                            st.write("No statistics available")
                    
                    # Predictions visualization
                    predictions = analyst_insights.get("predictions", [])
                    if predictions:
                        predictions_df = pd.DataFrame(predictions)
                        
                        st.subheader("📊 Prediction Distribution")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            pred_dist = predictions_df["prediction_label"].value_counts().to_dict()
                            fig_pie = create_prediction_pie_chart(pred_dist)
                            st.plotly_chart(fig_pie, use_container_width=True)
                        
                        with col2:
                            confidences = predictions_df["confidence"].tolist()
                            fig_hist = create_confidence_histogram(confidences)
                            st.plotly_chart(fig_hist, use_container_width=True)
                
                st.markdown("---")
                
                # Strategist Recommendations
                st.header("🎯 Strategist Agent Recommendations")
                
                strategist_recommendations = result.get("strategist_recommendations", {})
                
                if strategist_recommendations:
                    recommendations = strategist_recommendations.get("recommendations", [])
                    
                    if recommendations:
                        st.subheader("💼 Priority Recommendations")
                        
                        for i, rec in enumerate(recommendations[:5], 1):
                            with st.expander(f"**{i}. {rec.get('title', 'Recommendation')}** - Priority: {rec.get('priority', 'N/A')}", expanded=(i==1)):
                                st.write(f"**Description:** {rec.get('description', 'N/A')}")
                                st.write(f"**Expected Impact:** {rec.get('expected_impact', 'N/A')}")
                                st.write(f"**Effort Required:** {rec.get('effort', 'N/A')}")
                                
                                actions = rec.get('actions', [])
                                if actions:
                                    st.write("**Action Items:**")
                                    for action in actions:
                                        st.write(f"- {action}")
                    
                    # Impact estimation
                    impact = strategist_recommendations.get("impact_estimation", {})
                    if impact:
                        st.subheader("💰 Impact Estimation")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            revenue_impact = impact.get("revenue_impact", 0)
                            st.metric(
                                label="Revenue Impact",
                                value=format_currency(revenue_impact),
                                delta="Estimated"
                            )
                        
                        with col2:
                            cost_reduction = impact.get("cost_reduction", 0)
                            st.metric(
                                label="Cost Reduction",
                                value=format_currency(cost_reduction),
                                delta="Potential"
                            )
                        
                        with col3:
                            roi = impact.get("roi", 0)
                            st.metric(
                                label="Expected ROI",
                                value=f"{roi:.1f}%",
                                delta="Projected"
                            )
                
                st.markdown("---")
                
                # Reporter Summary
                st.header("📝 Reporter Agent Summary")
                
                reporter_summary = result.get("reporter_summary", {})
                
                if reporter_summary:
                    # Executive Summary
                    exec_summary = reporter_summary.get("executive_summary", "")
                    if exec_summary:
                        st.subheader("📋 Executive Summary")
                        st.info(exec_summary)
                    
                    # Key Findings
                    key_findings = reporter_summary.get("key_findings", [])
                    if key_findings:
                        st.subheader("🔑 Key Findings")
                        for finding in key_findings:
                            st.success(f"✓ {finding}")
                    
                    # Action Items
                    action_items = reporter_summary.get("action_items", [])
                    if action_items:
                        st.subheader("✅ Action Items")
                        for item in action_items:
                            st.warning(f"→ {item}")
                    
                    # Metrics Dashboard
                    metrics = reporter_summary.get("metrics_dashboard", {})
                    if metrics:
                        st.subheader("📊 Metrics Dashboard")
                        
                        cols = st.columns(len(metrics))
                        for i, (key, value) in enumerate(metrics.items()):
                            with cols[i]:
                                st.metric(label=key.replace("_", " ").title(), value=value)
                
                st.markdown("---")
                
                # Export Options
                st.header("💾 Export Results")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    # Export as JSON
                    json_export = json.dumps(result, indent=2, ensure_ascii=False)
                    st.download_button(
                        label="📥 Download JSON",
                        data=json_export,
                        file_name=f"multi_agent_analysis_{result.get('analysis_id', 'report')}.json",
                        mime="application/json",
                        use_container_width=True
                    )
                
                with col2:
                    # Export predictions as CSV
                    if predictions:
                        csv_export = predictions_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download CSV",
                            data=csv_export,
                            file_name=f"predictions_{result.get('analysis_id', 'report')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                
                with col3:
                    # Export as Markdown
                    markdown_report = f"""# Multi-Agent Analysis Report

## Analysis ID: {result.get('analysis_id', 'N/A')}
**Date:** {result.get('timestamp', 'N/A')}
**Products Analyzed:** {result.get('products_analyzed', 0)}

---

## Executive Summary

{exec_summary}

---

## Key Findings

{chr(10).join([f"- {finding}" for finding in key_findings])}

---

## Recommendations

{chr(10).join([f"### {i}. {rec.get('title', 'N/A')}{chr(10)}{rec.get('description', 'N/A')}{chr(10)}" for i, rec in enumerate(recommendations[:5], 1)])}

---

## Action Items

{chr(10).join([f"- {item}" for item in action_items])}

---

*Generated by NexoCommerce Multi-Agent System*
"""
                    st.download_button(
                        label="📥 Download Markdown",
                        data=markdown_report,
                        file_name=f"report_{result.get('analysis_id', 'report')}.md",
                        mime="text/markdown",
                        use_container_width=True
                    )
                
                with col4:
                    # Export as HTML (if available from reporter)
                    html_report = reporter_summary.get("html_report", "")
                    if html_report:
                        st.download_button(
                            label="📥 Download HTML",
                            data=html_report,
                            file_name=f"report_{result.get('analysis_id', 'report')}.html",
                            mime="text/html",
                            use_container_width=True
                        )
                    else:
                        st.button("📥 HTML (N/A)", disabled=True, use_container_width=True)
    
    except Exception as e:
        st.error(f"❌ Error: {e}")
        import traceback
        st.code(traceback.format_exc())

else:
    st.info("👆 Upload a CSV file to start multi-agent analysis")
    
    st.markdown("---")
    
    # Information about multi-agent system
    st.subheader("🤖 About Multi-Agent Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 📊 Analyst Agent
        
        The Analyst Agent performs:
        - Data validation and cleaning
        - ML model predictions
        - Statistical analysis
        - Pattern recognition
        - Insight generation
        """)
    
    with col2:
        st.markdown("""
        ### 🎯 Strategist Agent
        
        The Strategist Agent provides:
        - Business recommendations
        - Action prioritization
        - Impact estimation
        - ROI calculations
        - Strategic planning
        """)
    
    with col3:
        st.markdown("""
        ### 📝 Reporter Agent
        
        The Reporter Agent creates:
        - Executive summaries
        - Comprehensive reports
        - Key findings
        - Action items
        - Export formats (JSON, MD, HTML)
        """)

# Footer
st.markdown("---")
st.caption("💡 Tip: Multi-agent analysis provides the most comprehensive insights for decision-making")