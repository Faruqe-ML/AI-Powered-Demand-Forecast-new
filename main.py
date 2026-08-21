from datetime import datetime
import warnings

import streamlit as st

from app_pages.executive_dashboard import show_executive_dashboard
from app_pages.homepage import show_home
from app_pages.sales_analysis_dashboard import show_sales_analytics

st.set_page_config(
    page_title="RetailPulse AI Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

warnings.filterwarnings('ignore')

pages = {
    "🏠 Home": "home",
    "📈 Executive Dashboard": "executive",
    "💰 Sales Analytics": "sales",
    "📦 Product Performance": "products",
    "📊 Category Performance": "categories",
    "📦 Inventory Dashboard": "inventory",
    "⚠️ Stockout Risk": "stockout",
    "📦 Overstock Dashboard": "overstock",
    "🎯 Promotion Dashboard": "promotion",
    "📅 Seasonality Dashboard": "seasonality",
    "🔮 Forecast Dashboard": "forecast",
    "👥 Customer Insights": "customers",
    "💡 Recommendations": "recommendations",
    "🤖 ML Performance": "ml_performance",
    "📈 Demand Forecasting": "forecasting",
    "👥 Customer Segmentation": "segmentation",
    "⚠️ Churn Prediction": "churn",
    "📦 Inventory Optimization": "inventory_opt",
    "📊 Business Intelligence": "bi"
}

# Create navigation with icons
selected_page = st.sidebar.radio(
    "Navigate to:",
    list(pages.keys()),
    index=0
)
page_key = pages[selected_page]


# Sidebar Footer
st.sidebar.markdown("---")
st.sidebar.markdown(f"""
<div style='text-align: center; font-size: 12px; color: #666;'>
    <p>RetailPulse AI v2.0</p>
    <p>Data updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
    <p style='margin-top: 10px;'>📧 analytics@retailpulse.ai</p>
</div>
""", unsafe_allow_html=True)

if page_key == "home":
    show_home()
elif page_key == "executive":
    show_executive_dashboard()

elif page_key == "sales":
    show_sales_analytics()