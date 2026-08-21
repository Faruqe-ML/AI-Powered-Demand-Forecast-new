import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import warnings

from utility.executive_db_util import create_monthly_growth_chart, calculate_metrics, create_revenue_trend_chart, \
    create_units_sold_chart, create_category_revenue_chart, create_monthly_revenue_chart

warnings.filterwarnings('ignore')

import api

@st.cache_data
def load_data():
    """Load all required data"""

    product_summary = api.get_product_summary()
    sales_summary = api.get_sales_summary()

    sku = pd.DataFrame(api.get_skus())
    daily_sales = pd.DataFrame(api.get_daily_sales())

    # Convert date
    daily_sales["date"] = pd.to_datetime(
        daily_sales["date"],
        errors="coerce"
    )

    if "category" not in daily_sales.columns:
        daily_sales["category"] = "Unknown"
    else:
        daily_sales["category"] = (
            daily_sales["category"].fillna("Unknown")
        )

    daily = (
        daily_sales
        .groupby("date")
        .agg({
            "units_sold": "sum",
            "revenue": "sum",
            "unit_price": "mean"
        })
        .reset_index()
    )

    daily.columns = [
        "date",
        "units_sold",
        "revenue",
        "avg_price"
    ]

    daily_category = (
        daily_sales
        .groupby(["date", "category"])
        .agg({
            "units_sold": "sum",
            "revenue": "sum",
            "unit_price": "mean"
        })
        .reset_index()
    )

    daily_category.columns = [
        "date",
        "category",
        "units_sold",
        "revenue",
        "avg_price"
    ]

    return {
        "sku": sku,
        "daily_sales": daily_sales,
        "daily_sales_with_cat": daily_sales,
        "daily": daily,
        "daily_category": daily_category,
        "product_summary": product_summary,
        "sales_summary": sales_summary
    }

def show_executive_dashboard():
    """Display Executive Dashboard"""

    # Header
    st.markdown("""
    <div style='
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.5rem;
    '>
        <h1 style='color: white !important; margin: 0; font-size: 2rem;'>📈 Executive Dashboard</h1>
        <p style='color: rgba(255,255,255,0.9) !important; margin: 8px 0 0 0;'>
            Real-time business intelligence and key performance indicators
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ============================================
    # LOAD DATA
    # ============================================

    data = load_data()

    if data['daily'] is None or len(data['daily']) == 0:
        st.error("❌ No data available. Please check your data source.")
        if st.button("🔄 Retry Loading"):
            st.cache_data.clear()
            st.rerun()
        return

    # ============================================
    # CALCULATE METRICS
    # ============================================

    metrics = calculate_metrics(data['daily'], data['daily_sales_with_cat'])

    # ============================================
    # KPI ROW 1
    # ============================================

    st.markdown("""
                      <style>

                      /* Metric card */
                      [data-testid="stMetric"] {
                          background-color: white;
                          border: 1px solid #3b3b3b;
                          border-radius: 10px;
                          padding: 10px;
                          text-align: center;
                      }

                      /* Metric label */
                      [data-testid="stMetricLabel"] {
                          justify-content: center;
                      }

                      [data-testid="stMetricLabel"] p {
                          font-size: 12px !important;
                          font-weight: 500 !important;
                      }

                      /* Metric value */
                      [data-testid="stMetricValue"] {
                          font-size: 22px !important;
                          font-weight: 700 !important;
                      }

                      /* Reduce spacing */
                      div[data-testid="stMetric"] > div {
                          padding: 0 !important;
                          margin: 0 !important;
                      }

                      </style>
                      """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("💰 Total Revenue", f"₹{metrics['total_revenue']:,.0f}")

    with col2:
        st.metric("📦 Total Units Sold", f"{metrics['total_units']:,.0f}")

    with col3:
        st.metric("📊 Total SKUs", f"{metrics['total_skus']:,}")

    # ============================================
    # KPI ROW 2
    # ============================================

    col4, col5, col6 = st.columns(3)

    with col4:
        st.metric("💵 Avg Order Value", f"₹{metrics['avg_order_value']:,.2f}")

    with col5:
        st.metric("📅 Avg Monthly Revenue", f"₹{metrics['avg_monthly_revenue']:,.0f}")

    with col6:
        delta_color = "normal" if metrics['revenue_growth'] >= 0 else "inverse"
        st.metric("📈 Revenue Growth", f"{metrics['revenue_growth']:.1f}%", delta_color=delta_color)

    # ============================================
    # KPI ROW 3
    # ============================================

    col7, col8, col9 = st.columns(3)

    with col7:
        st.metric("⏳ Analysis Period", f"{metrics['days']} Days")

    with col8:
        start = metrics.get('start_date')
        st.metric("📅 Start Date", start.strftime("%Y-%m-%d") if pd.notna(start) else "N/A")

    with col9:
        end = metrics.get('end_date')
        st.metric("📅 End Date", end.strftime("%Y-%m-%d") if pd.notna(end) else "N/A")

    st.divider()

    # ============================================
    # CHARTS
    # ============================================

    # Chart 1: Revenue Trend
    fig1 = create_revenue_trend_chart(data['daily_sales'])
    st.plotly_chart(fig1, use_container_width=True)
    st.divider()

    # Chart 2: Units Sold Trend
    fig2 = create_units_sold_chart(data['daily_sales'])
    st.plotly_chart(fig2, use_container_width=True)
    st.divider()

    # Chart 3: Monthly Revenue
    fig3 = create_monthly_revenue_chart(metrics['monthly_rev'])
    st.plotly_chart(fig3, use_container_width=True)
    st.divider()

    # Chart 4: Category Revenue
    fig4 = create_category_revenue_chart(data['daily_category'])
    st.plotly_chart(fig4, use_container_width=True)
    st.divider()

    # Chart 5: Monthly Growth
    fig5 = create_monthly_growth_chart(metrics['monthly_rev'])
    st.plotly_chart(fig5, use_container_width=True)



    # ============================================
    # FOOTER
    # ============================================

    st.caption(f"""
        **RetailPulse AI** | Executive Dashboard  
        Data updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Powered by Zidio Data Science
        """)



