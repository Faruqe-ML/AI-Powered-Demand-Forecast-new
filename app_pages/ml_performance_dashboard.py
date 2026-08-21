import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import warnings

from utility.ml_performance_util import load_prophet_model, load_random_forest_model, load_linear_model, \
    get_model_metrics_from_data

warnings.filterwarnings('ignore')

import api

# Load data
daily_sales = pd.DataFrame(api.get_daily_sales())
daily_sales["date"] = pd.to_datetime(daily_sales["date"], errors="coerce")

def show_ml_performance():
    """Display ML Performance Dashboard"""

    # Header
    st.markdown("""
    <div style='
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.5rem;
    '>
        <h1 style='color: white !important; margin: 0; font-size: 2rem;'>🤖 ML Performance Dashboard</h1>
        <p style='color: rgba(255,255,255,0.9) !important; margin: 8px 0 0 0;'>
            Model Evaluation & Comparison for Demand Forecasting
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Load models and metrics
    prophet_model = load_prophet_model()
    rf_model = load_random_forest_model()
    linear_model = load_linear_model()
    metrics = get_model_metrics_from_data(daily_sales)

    # ============================================
    # MODEL STATUS
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

    st.markdown("### 📦 Model Status")

    col1, col2, col3 = st.columns(3)
    with col1:
        if prophet_model:
            st.success("✅ Prophet: Loaded")
        else:
            st.error("❌ Prophet: Not Found")

    with col2:
        if rf_model:
            st.success("✅ Random Forest: Loaded")
        else:
            st.error("❌ Random Forest: Not Found")

    with col3:
        if linear_model:
            st.success("✅ Linear Model: Loaded")
        else:
            st.error("❌ Linear Model: Not Found")

    st.divider()

    # ============================================
    # KPI CARDS
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

    st.markdown("### 📈 Key Performance Indicators")

    col1, col2, col3, col4 = st.columns(4)

    best_mae = metrics.loc[metrics['MAE'].idxmin()]
    best_acc2 = metrics.loc[metrics['Acc_±2'].idxmax()]

    with col1:
        st.metric(
            label="🏆 Best Model",
            value=best_mae['Model'],
            delta=f"MAE: {best_mae['MAE']:.2f}"
        )

    with col2:
        st.metric(
            label="📉 Best MAE",
            value=f"{best_mae['MAE']:.2f}",
            delta=best_mae['Model']
        )

    with col3:
        st.metric(
            label="🎯 Best Accuracy (±2)",
            value=f"{best_acc2['Acc_±2']:.1f}%",
            delta=best_acc2['Model']
        )

    with col4:
        st.metric(
            label="📊 Models Evaluated",
            value=f"{len(metrics)}"
        )

    st.divider()

    # ============================================
    # METRICS TABLE
    # ============================================

    st.markdown("### 📊 Model Performance Metrics")

    display_metrics = metrics.copy()
    display_metrics['MAE'] = display_metrics['MAE'].round(2)
    display_metrics['RMSE'] = display_metrics['RMSE'].round(2)
    display_metrics['Acc_±1'] = display_metrics['Acc_±1'].round(1)
    display_metrics['Acc_±2'] = display_metrics['Acc_±2'].round(1)

    # Rename columns for display
    display_metrics = display_metrics.rename(columns={
        'Acc_±1': 'Accuracy ±1 (%)',
        'Acc_±2': 'Accuracy ±2 (%)'
    })

    st.dataframe(
        display_metrics.style
        .highlight_min(subset=['MAE', 'RMSE'], color='lightgreen')
        .highlight_max(subset=['Accuracy ±1 (%)', 'Accuracy ±2 (%)'], color='lightgreen'),
        use_container_width=True
    )

    st.divider()

    # ============================================
    # BAR CHARTS
    # ============================================

    st.markdown("### 📊 Visual Comparison")

    fig_mae = px.bar(
        metrics,
        x="Model",
        y="MAE",
        color="Model",
        title="Mean Absolute Error (MAE)",
        text=metrics["MAE"].round(2),
        color_discrete_sequence=["#FF6B6B", "#4ECDC4", "#FFE66D"]
    )

    fig_mae.update_traces(
        textposition="outside",
        textfont=dict(
            color="white",
            size=13
        )
    )

    fig_mae.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        font=dict(
            color="white"
        ),
        title=dict(
            text="Mean Absolute Error (MAE)",
            x=0.5,
            xanchor="center",
            font=dict(
                color="white",
                size=18
            )
        ),
        xaxis=dict(
            title="Model",
            title_font=dict(color="white"),
            tickfont=dict(color="white")
        ),
        yaxis=dict(
            title="MAE",
            title_font=dict(color="white"),
            tickfont=dict(color="white")
        ),
        showlegend=False
    )

    st.plotly_chart(
        fig_mae,
        use_container_width=True
    )

    fig_acc = px.bar(
        metrics,
        x="Model",
        y="Acc_±2",
        color="Model",
        title="Accuracy ±2 Units",
        text=metrics["Acc_±2"].round(1).astype(str) + "%",
        color_discrete_sequence=["#FF6B6B", "#4ECDC4", "#FFE66D"]
    )

    fig_acc.update_traces(
        textposition="outside",
        textfont=dict(
            color="white",
            size=13
        )
    )

    fig_acc.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        font=dict(
            color="white"
        ),
        title=dict(
            text="Accuracy ±2 Units",
            x=0.5,
            xanchor="center",
            font=dict(
                color="white",
                size=18
            )
        ),
        xaxis=dict(
            title="Model",
            title_font=dict(color="white"),
            tickfont=dict(color="white")
        ),
        yaxis=dict(
            title="Accuracy (%)",
            title_font=dict(color="white"),
            tickfont=dict(color="white"),
            range=[0, 100]
        ),
        showlegend=False
    )

    st.plotly_chart(
        fig_acc,
        use_container_width=True
    )


    st.divider()

    st.divider()

    # ============================================
    # FEATURE IMPORTANCE (if Random Forest is loaded)
    # ============================================

    if rf_model:
        st.markdown("### 🔍 Feature Importance")

        try:
            # Try to load from CSV
            feature_importance = pd.read_csv('models/feature_importance.csv')

            fig_imp = px.bar(
                feature_importance.head(10),
                x='Importance',
                y='Feature',
                orientation='h',
                title='Top 10 Feature Importance (Random Forest)',
                color='Importance',
                color_continuous_scale='Viridis'
            )
            fig_imp.update_layout(
                template='plotly_dark',
                paper_bgcolor='#0e1117',
                plot_bgcolor='#0e1117',
                font=dict(color='white'),
                height=400
            )
            st.plotly_chart(fig_imp, use_container_width=True)
        except:
            st.info("ℹ️ Feature importance data not available. Please train the model and save feature_importance.csv.")

    st.divider()

    # ============================================
    # FOOTER
    # ============================================

    st.caption(f"""
        **Project FORESIGHT** | AI-Powered Demand & Inventory Intelligence Platform
        *ML Performance Dashboard v2.0 | Last Updated: {datetime.now().strftime('%B %Y')}*
        """)

