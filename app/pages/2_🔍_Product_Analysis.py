"""
Product Analysis - Analyze individual products
"""

import streamlit as st
import sys
from pathlib import Path
import json

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from app.utils.api_client import get_api_client
from app.utils.data_utils import (
    validate_product_data, 
    create_sample_product,
    get_action_color,
    get_action_emoji,
    format_currency
)

st.set_page_config(page_title="Product Analysis", page_icon="🔍", layout="wide")

# Header
st.title("🔍 Product Analysis")
st.markdown("Analyze individual product lifecycle and get AI-powered recommendations")

# Sidebar - API Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    api_url = st.text_input("API URL", value="http://localhost:8000")
    
    st.markdown("---")
    
    st.header("📝 Quick Actions")
    if st.button("Load Sample Product", use_container_width=True):
        st.session_state.sample_loaded = True

# Get API client
api_client = get_api_client(api_url)

# Check API health
health = api_client.health_check()
if health.get("status") != "healthy":
    st.error(f"❌ API is not responding: {health.get('message', 'Unknown error')}")
    st.stop()

st.markdown("---")

# Input Method Selection
input_method = st.radio(
    "Select Input Method:",
    ["Manual Input", "JSON Input"],
    horizontal=True
)

product_data = None

if input_method == "Manual Input":
    st.subheader("📝 Product Information")
    
    # Load sample if requested
    if st.session_state.get('sample_loaded', False):
        sample = create_sample_product()
        st.session_state.sample_loaded = False
    else:
        sample = {}
    
    col1, col2 = st.columns(2)
    
    with col1:
        product_id = st.text_input("Product ID *", value=sample.get("product_id", ""))
        price = st.number_input("Price (R$) *", min_value=0.0, value=float(sample.get("price", 0.0)), step=0.01)
        rating = st.slider("Rating *", min_value=0.0, max_value=5.0, value=float(sample.get("rating", 0.0)), step=0.1)
        num_reviews = st.number_input("Number of Reviews *", min_value=0, value=int(sample.get("num_reviews", 0)), step=1)
        stock_quantity = st.number_input("Stock Quantity *", min_value=0, value=int(sample.get("stock_quantity", 0)), step=1)
        sales_last_30d = st.number_input("Sales (Last 30 days) *", min_value=0, value=int(sample.get("sales_last_30d", 0)), step=1)
    
    with col2:
        views_last_30d = st.number_input("Views (Last 30 days) *", min_value=0, value=int(sample.get("views_last_30d", 0)), step=1)
        category = st.text_input("Category *", value=sample.get("category", ""))
        brand = st.text_input("Brand *", value=sample.get("brand", ""))
        days_since_launch = st.number_input("Days Since Launch *", min_value=0, value=int(sample.get("days_since_launch", 0)), step=1)
        discount_percentage = st.number_input("Discount % (optional)", min_value=0.0, max_value=100.0, value=float(sample.get("discount_percentage", 0.0)), step=0.1)
        return_rate = st.number_input("Return Rate (optional)", min_value=0.0, max_value=1.0, value=float(sample.get("return_rate", 0.0)), step=0.01)
    
    product_data = {
        "product_id": product_id,
        "price": price,
        "rating": rating,
        "num_reviews": num_reviews,
        "stock_quantity": stock_quantity,
        "sales_last_30d": sales_last_30d,
        "views_last_30d": views_last_30d,
        "category": category,
        "brand": brand,
        "days_since_launch": days_since_launch,
        "discount_percentage": discount_percentage,
        "return_rate": return_rate
    }

else:  # JSON Input
    st.subheader("📄 JSON Input")
    
    sample_json = json.dumps(create_sample_product(), indent=2)
    
    json_input = st.text_area(
        "Paste product data in JSON format:",
        value=sample_json,
        height=300
    )
    
    try:
        product_data = json.loads(json_input)
    except json.JSONDecodeError as e:
        st.error(f"Invalid JSON: {e}")
        product_data = None

st.markdown("---")

# Analyze Button
col1, col2, col3 = st.columns([1, 1, 1])

with col2:
    analyze_button = st.button("🚀 Analyze Product", use_container_width=True, type="primary")

if analyze_button and product_data:
    # Validate data
    is_valid, error_msg = validate_product_data(product_data)
    
    if not is_valid:
        st.error(f"❌ Validation Error: {error_msg}")
    else:
        # Make prediction
        with st.spinner("🔄 Analyzing product..."):
            result = api_client.predict_single(product_data)
        
        if "error" in result:
            st.error(f"❌ Error: {result['error']}")
        else:
            st.success("✅ Analysis completed successfully!")
            
            st.markdown("---")
            
            # Display Results
            st.header("📊 Analysis Results")
            
            # Main prediction
            prediction = result.get("prediction", "N/A")
            confidence = result.get("confidence", 0)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    label="Recommended Action",
                    value=f"{get_action_emoji(prediction)} {prediction}"
                )
            
            with col2:
                st.metric(
                    label="Confidence",
                    value=f"{confidence*100:.1f}%"
                )
            
            with col3:
                st.metric(
                    label="Product ID",
                    value=product_data.get("product_id", "N/A")
                )
            
            st.markdown("---")
            
            # Detailed Information
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📦 Product Details")
                st.write(f"**Category:** {product_data.get('category', 'N/A')}")
                st.write(f"**Brand:** {product_data.get('brand', 'N/A')}")
                st.write(f"**Price:** {format_currency(product_data.get('price', 0))}")
                st.write(f"**Rating:** {product_data.get('rating', 0):.1f} ⭐")
                st.write(f"**Reviews:** {product_data.get('num_reviews', 0)}")
                st.write(f"**Days Since Launch:** {product_data.get('days_since_launch', 0)}")
            
            with col2:
                st.subheader("📈 Performance Metrics")
                st.write(f"**Stock:** {product_data.get('stock_quantity', 0)} units")
                st.write(f"**Sales (30d):** {product_data.get('sales_last_30d', 0)} units")
                st.write(f"**Views (30d):** {product_data.get('views_last_30d', 0)}")
                
                conversion_rate = (product_data.get('sales_last_30d', 0) / product_data.get('views_last_30d', 1)) * 100 if product_data.get('views_last_30d', 0) > 0 else 0
                st.write(f"**Conversion Rate:** {conversion_rate:.2f}%")
                
                if product_data.get('discount_percentage', 0) > 0:
                    st.write(f"**Discount:** {product_data.get('discount_percentage', 0):.1f}%")
                if product_data.get('return_rate', 0) > 0:
                    st.write(f"**Return Rate:** {product_data.get('return_rate', 0)*100:.1f}%")
            
            st.markdown("---")
            
            # Recommendation Box
            st.subheader("💡 Recommendation")
            
            if prediction == "PROMOVER":
                st.success("""
                **🚀 PROMOVER - Promote this product!**
                
                This product shows strong performance and has high potential for growth.
                
                **Suggested Actions:**
                - Increase marketing budget
                - Feature in promotional campaigns
                - Consider expanding inventory
                - Optimize pricing strategy
                """)
            elif prediction == "MANTER":
                st.info("""
                **✅ MANTER - Maintain current strategy**
                
                This product is performing adequately and should be maintained.
                
                **Suggested Actions:**
                - Monitor performance regularly
                - Maintain current stock levels
                - Continue existing marketing efforts
                - Watch for market trends
                """)
            else:  # DESCONTINUAR
                st.warning("""
                **⛔ DESCONTINUAR - Consider discontinuing**
                
                This product shows weak performance and may not be viable long-term.
                
                **Suggested Actions:**
                - Reduce inventory gradually
                - Implement clearance sales
                - Analyze reasons for poor performance
                - Consider product replacement
                """)
            
            st.markdown("---")
            
            # Export Results
            st.subheader("💾 Export Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.download_button(
                    label="📥 Download JSON",
                    data=json.dumps(result, indent=2),
                    file_name=f"analysis_{product_data.get('product_id', 'product')}.json",
                    mime="application/json",
                    use_container_width=True
                )
            
            with col2:
                # Create markdown report
                markdown_report = f"""# Product Analysis Report

## Product: {product_data.get('product_id', 'N/A')}

### Recommendation: {get_action_emoji(prediction)} {prediction}
**Confidence:** {confidence*100:.1f}%

### Product Details
- **Category:** {product_data.get('category', 'N/A')}
- **Brand:** {product_data.get('brand', 'N/A')}
- **Price:** {format_currency(product_data.get('price', 0))}
- **Rating:** {product_data.get('rating', 0):.1f} ⭐
- **Reviews:** {product_data.get('num_reviews', 0)}

### Performance Metrics
- **Stock:** {product_data.get('stock_quantity', 0)} units
- **Sales (30d):** {product_data.get('sales_last_30d', 0)} units
- **Views (30d):** {product_data.get('views_last_30d', 0)}
- **Conversion Rate:** {conversion_rate:.2f}%

---
*Generated by NexoCommerce Multi-Agent System*
"""
                st.download_button(
                    label="📥 Download Markdown",
                    data=markdown_report,
                    file_name=f"analysis_{product_data.get('product_id', 'product')}.md",
                    mime="text/markdown",
                    use_container_width=True
                )

# Footer
st.markdown("---")
st.caption("💡 Tip: Use the sidebar to load a sample product for testing")