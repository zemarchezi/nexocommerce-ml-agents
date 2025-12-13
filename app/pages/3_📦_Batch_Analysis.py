"""
Batch Analysis - Process multiple products at once
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
    create_category_bar_chart
)

st.set_page_config(page_title="Batch Analysis", page_icon="📦", layout="wide")

# Header
st.title("📦 Batch Analysis")
st.markdown("Process multiple products at once and get comprehensive insights")

# Sidebar - API Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    api_url = st.text_input("API URL", value="http://localhost:8000")
    
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
    help="CSV file should contain columns: product_id, price, rating, num_reviews, stock_quantity, sales_last_30d, views_last_30d, category, brand, days_since_launch"
)

if uploaded_file is not None:
    try:
        # Read CSV
        df = pd.read_csv(uploaded_file)
        
        st.success(f"✅ Loaded {len(df)} products from CSV")
        
        # Display preview
        with st.expander("👀 Preview Data", expanded=True):
            st.dataframe(df.head(10), use_container_width=True)
        
        st.markdown("---")
        
        # Analyze Button
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col2:
            analyze_button = st.button("🚀 Analyze Batch", use_container_width=True, type="primary")
        
        if analyze_button:
            # Convert to products list
            products = csv_to_products(df)
            
            # Make batch prediction
            with st.spinner(f"🔄 Analyzing {len(products)} products..."):
                result = api_client.predict_batch(products)
            
            if "error" in result:
                st.error(f"❌ Error: {result['error']}")
            else:
                st.success("✅ Batch analysis completed successfully!")
                
                st.markdown("---")
                
                # Display Results
                st.header("📊 Batch Analysis Results")
                
                predictions = result.get("predictions", [])
                predictions_df = pd.DataFrame(predictions)
                
                # Summary Metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        label="Total Products",
                        value=len(predictions)
                    )
                
                with col2:
                    avg_confidence = predictions_df["confidence"].mean() if len(predictions_df) > 0 else 0
                    st.metric(
                        label="Avg Confidence",
                        value=f"{avg_confidence*100:.1f}%"
                    )
                
                with col3:
                    promover_count = len(predictions_df[predictions_df["prediction"] == "PROMOVER"])
                    st.metric(
                        label="🚀 Promote",
                        value=promover_count
                    )
                
                with col4:
                    descontinuar_count = len(predictions_df[predictions_df["prediction"] == "DESCONTINUAR"])
                    st.metric(
                        label="⛔ Discontinue",
                        value=descontinuar_count
                    )
                
                st.markdown("---")
                
                # Visualizations
                st.subheader("📈 Visualizations")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Prediction distribution
                    pred_dist = predictions_df["prediction"].value_counts().to_dict()
                    fig_pie = create_prediction_pie_chart(pred_dist)
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                with col2:
                    # Confidence distribution
                    confidences = predictions_df["confidence"].tolist()
                    fig_hist = create_confidence_histogram(confidences)
                    st.plotly_chart(fig_hist, use_container_width=True)
                
                # Category breakdown
                if "category" in predictions_df.columns:
                    category_dist = predictions_df["category"].value_counts().to_dict()
                    fig_cat = create_category_bar_chart(category_dist)
                    st.plotly_chart(fig_cat, use_container_width=True)
                
                st.markdown("---")
                
                # Detailed Results Table
                st.subheader("📋 Detailed Results")
                
                # Add emoji to predictions
                predictions_df["action"] = predictions_df["prediction"].apply(
                    lambda x: f"{get_action_emoji(x)} {x}"
                )
                
                # Format confidence
                predictions_df["confidence_pct"] = predictions_df["confidence"].apply(
                    lambda x: f"{x*100:.1f}%"
                )
                
                # Select columns to display
                display_cols = ["product_id", "action", "confidence_pct", "category", "brand", "price", "rating", "sales_last_30d"]
                display_cols = [col for col in display_cols if col in predictions_df.columns]
                
                st.dataframe(
                    predictions_df[display_cols],
                    use_container_width=True,
                    hide_index=True
                )
                
                st.markdown("---")
                
                # Action Breakdown
                st.subheader("🎯 Action Breakdown")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("### 🚀 PROMOVER")
                    promover_df = predictions_df[predictions_df["prediction"] == "PROMOVER"]
                    if len(promover_df) > 0:
                        st.write(f"**Count:** {len(promover_df)}")
                        st.write(f"**Avg Confidence:** {promover_df['confidence'].mean()*100:.1f}%")
                        st.write("**Top Products:**")
                        for pid in promover_df["product_id"].head(5):
                            st.write(f"- {pid}")
                    else:
                        st.info("No products to promote")
                
                with col2:
                    st.markdown("### ✅ MANTER")
                    manter_df = predictions_df[predictions_df["prediction"] == "MANTER"]
                    if len(manter_df) > 0:
                        st.write(f"**Count:** {len(manter_df)}")
                        st.write(f"**Avg Confidence:** {manter_df['confidence'].mean()*100:.1f}%")
                        st.write("**Top Products:**")
                        for pid in manter_df["product_id"].head(5):
                            st.write(f"- {pid}")
                    else:
                        st.info("No products to maintain")
                
                with col3:
                    st.markdown("### ⛔ DESCONTINUAR")
                    descontinuar_df = predictions_df[predictions_df["prediction"] == "DESCONTINUAR"]
                    if len(descontinuar_df) > 0:
                        st.write(f"**Count:** {len(descontinuar_df)}")
                        st.write(f"**Avg Confidence:** {descontinuar_df['confidence'].mean()*100:.1f}%")
                        st.write("**Top Products:**")
                        for pid in descontinuar_df["product_id"].head(5):
                            st.write(f"- {pid}")
                    else:
                        st.info("No products to discontinue")
                
                st.markdown("---")
                
                # Export Results
                st.subheader("💾 Export Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # Export as CSV
                    csv_export = predictions_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download CSV",
                        data=csv_export,
                        file_name="batch_analysis_results.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                
                with col2:
                    # Export as JSON
                    json_export = json.dumps(result, indent=2)
                    st.download_button(
                        label="📥 Download JSON",
                        data=json_export,
                        file_name="batch_analysis_results.json",
                        mime="application/json",
                        use_container_width=True
                    )
                
                with col3:
                    # Export summary as Markdown
                    markdown_report = f"""# Batch Analysis Report

## Summary
- **Total Products:** {len(predictions)}
- **Average Confidence:** {avg_confidence*100:.1f}%

## Action Distribution
- **🚀 PROMOVER:** {promover_count} products
- **✅ MANTER:** {len(predictions_df[predictions_df["prediction"] == "MANTER"])} products
- **⛔ DESCONTINUAR:** {descontinuar_count} products

---
*Generated by NexoCommerce Multi-Agent System*
"""
                    st.download_button(
                        label="📥 Download Report",
                        data=markdown_report,
                        file_name="batch_analysis_report.md",
                        mime="text/markdown",
                        use_container_width=True
                    )
    
    except Exception as e:
        st.error(f"❌ Error reading CSV: {e}")

else:
    st.info("👆 Upload a CSV file to start batch analysis")
    
    st.markdown("### 📋 Required CSV Columns:")
    st.code("""
product_id, price, rating, num_reviews, stock_quantity, 
sales_last_30d, views_last_30d, category, brand, days_since_launch
    """)
    
    st.markdown("### 📋 Optional CSV Columns:")
    st.code("""
discount_percentage, return_rate
    """)

# Footer
st.markdown("---")
st.caption("💡 Tip: Use the sidebar to generate a sample CSV file for testing")