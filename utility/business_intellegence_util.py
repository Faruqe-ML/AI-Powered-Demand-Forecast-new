from datetime import timedelta
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def calculate_bi_metrics(df):
    """Calculate BI metrics"""
    if df.empty:
        return {}

    # Filter last 30 days
    last_30 = df[df['date'] >= df['date'].max() - timedelta(days=30)]

    metrics = {
        'total_revenue': df['revenue'].sum(),
        'total_units': df['units_sold'].sum(),
        'total_orders': df['receipt_id'].nunique() if 'receipt_id' in df.columns else len(df),
        'avg_order_value': df['revenue'].mean() if 'revenue' in df.columns else 0,
        'unique_skus': df['sku_id'].nunique(),
        'unique_customers': df['customer_id'].nunique() if 'customer_id' in df.columns else 0,
        'revenue_last_30': last_30['revenue'].sum() if not last_30.empty else 0,
        'units_last_30': last_30['units_sold'].sum() if not last_30.empty else 0,
        'avg_daily_revenue': last_30['revenue'].mean() if not last_30.empty else 0,
        'avg_daily_units': last_30['units_sold'].mean() if not last_30.empty else 0,
    }
    return metrics


def create_revenue_trend(df, period='Daily'):
    """Revenue trend chart"""

    if df.empty:
        return go.Figure()

    df = df.copy()

    # Make sure date is datetime
    df['date'] = pd.to_datetime(
        df['date'],
        errors='coerce'
    )

    df = df.dropna(subset=['date'])

    # Aggregate by period
    if period == 'Daily':

        trend = (
            df.groupby(
                'date',
                as_index=False
            )['revenue']
            .sum()
        )

        x_col = 'date'

    elif period == 'Weekly':

        iso = df['date'].dt.isocalendar()

        df['iso_year'] = iso.year
        df['week'] = iso.week

        trend = (
            df.groupby(
                ['iso_year', 'week'],
                as_index=False
            )['revenue']
            .sum()
        )

        trend['date'] = pd.to_datetime(
            trend['iso_year'].astype(str)
            + '-W'
            + trend['week'].astype(str).str.zfill(2)
            + '-1',
            format='%G-W%V-%u'
        )

        x_col = 'date'

    else:

        df['month'] = df['date'].dt.month
        df['year'] = df['date'].dt.year

        trend = (
            df.groupby(
                ['year', 'month'],
                as_index=False
            )['revenue']
            .sum()
        )

        trend['date'] = pd.to_datetime(
            trend['year'].astype(str)
            + '-'
            + trend['month'].astype(str).str.zfill(2)
            + '-01'
        )

        x_col = 'date'

    # Sort chronologically
    trend = trend.sort_values('date')

    # Create chart
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=trend[x_col],
            y=trend['revenue'],
            mode='lines+markers',
            name='Revenue',

            line=dict(
                color='#6366F1',
                width=3
            ),

            marker=dict(
                size=6,
                color='#6366F1'
            ),

            fill='tozeroy',

            fillcolor='rgba(99,102,241,0.15)',

            hovertemplate=(
                '<b>%{x|%d %b %Y}</b><br>'
                'Revenue: ₹%{y:,.0f}'
                '<extra></extra>'
            ),

            hoverlabel=dict(
                bgcolor='#ffffff',
                font=dict(
                    color='#000000',
                    size=13
                )
            )
        )
    )

    fig.update_layout(

        # ==============================
        # CENTERED TITLE
        # ==============================
        title=dict(
            text='Revenue Trend',
            x=0.5,
            xanchor='center',
            y=0.95,
            yanchor='top',
            font=dict(
                color='white',
                size=18
            )
        ),

        # ==============================
        # AXES
        # ==============================
        xaxis=dict(
            title=dict(
                text='Date',
                font=dict(
                    color='white',
                    size=13
                )
            ),
            tickfont=dict(
                color='white',
                size=11
            ),
            color='white',
            gridcolor='rgba(255,255,255,0.10)'
        ),

        yaxis=dict(
            title=dict(
                text='Revenue (₹)',
                font=dict(
                    color='white',
                    size=13
                )
            ),
            tickfont=dict(
                color='white',
                size=11
            ),
            color='white',
            gridcolor='rgba(255,255,255,0.10)'
        ),

        # ==============================
        # DARK BACKGROUND
        # ==============================
        template='plotly_dark',

        paper_bgcolor='#0e1117',
        plot_bgcolor='#0e1117',

        font=dict(
            color='white'
        ),

        height=350,

        hovermode='x unified',

        legend=dict(
            font=dict(
                color='white'
            )
        )
    )

    return fig



def create_sales_by_category(df):
    """Sales by category chart"""

    if df.empty or 'category' not in df.columns:
        return go.Figure()

    category_sales = (
        df.groupby('category')['revenue']
        .sum()
        .sort_values(ascending=True)
        .reset_index()
    )

    category_sales = category_sales.tail(10)

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=category_sales['revenue'],
            y=category_sales['category'],
            orientation='h',

            marker=dict(
                color=category_sales['revenue'],
                colorscale='Viridis',
                showscale=True,

                colorbar=dict(
                    title=dict(
                        text='Revenue',
                        font=dict(
                            color='white'
                        )
                    ),
                    tickfont=dict(
                        color='white'
                    )
                )
            ),

            text=category_sales['revenue'].apply(
                lambda x: f'₹{x:,.0f}'
            ),

            textposition='outside',

            textfont=dict(
                color='white',
                size=10
            ),

            hovertemplate=(
                '<b>%{y}</b><br>'
                'Revenue: ₹%{x:,.0f}'
                '<extra></extra>'
            ),

            hoverlabel=dict(
                bgcolor='white',
                font=dict(
                    color='black',
                    size=13
                )
            )
        )
    )

    fig.update_layout(

        # ==============================
        # CENTERED TITLE
        # ==============================
        title=dict(
            text='Top 10 Categories by Revenue',
            x=0.5,
            xanchor='center',
            y=0.95,
            yanchor='top',
            font=dict(
                color='white',
                size=18
            )
        ),

        # ==============================
        # X AXIS
        # ==============================
        xaxis=dict(
            title=dict(
                text='Revenue (₹)',
                font=dict(
                    color='white',
                    size=13
                )
            ),
            tickfont=dict(
                color='white',
                size=11
            ),
            color='white',
            gridcolor='rgba(255,255,255,0.10)'
        ),

        # ==============================
        # Y AXIS
        # ==============================
        yaxis=dict(
            title=dict(
                text='Category',
                font=dict(
                    color='white',
                    size=13
                )
            ),
            tickfont=dict(
                color='white',
                size=11
            ),
            color='white',
            gridcolor='rgba(255,255,255,0.10)'
        ),

        template='plotly_dark',

        paper_bgcolor='#0e1117',

        plot_bgcolor='#0e1117',

        font=dict(
            color='white'
        ),

        height=350,

        margin=dict(
            l=100,
            r=80,
            t=60,
            b=50
        )
    )

    return fig

def create_top_products(df):
    """Top products chart"""

    if df.empty:
        return go.Figure()

    # Get SKU names from the data
    if 'sku_name' in df.columns:

        top_products = (
            df.groupby('sku_name')['revenue']
            .sum()
            .sort_values(ascending=True)
            .reset_index()
            .tail(10)
        )

        label_col = 'sku_name'

    else:

        top_products = (
            df.groupby('sku_id')['revenue']
            .sum()
            .sort_values(ascending=True)
            .reset_index()
            .tail(10)
        )

        label_col = 'sku_id'

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=top_products['revenue'],
            y=top_products[label_col],
            orientation='h',

            marker=dict(
                color=top_products['revenue'],
                colorscale='Plasma',
                showscale=True,

                colorbar=dict(
                    title=dict(
                        text='Revenue',
                        font=dict(
                            color='white'
                        )
                    ),
                    tickfont=dict(
                        color='white'
                    )
                )
            ),

            text=top_products['revenue'].apply(
                lambda x: f'₹{x:,.0f}'
            ),

            textposition='outside',

            textfont=dict(
                color='white',
                size=10
            ),

            hovertemplate=(
                '<b>%{y}</b><br>'
                'Revenue: ₹%{x:,.0f}'
                '<extra></extra>'
            ),

            hoverlabel=dict(
                bgcolor='white',
                font=dict(
                    color='black',
                    size=13
                )
            )
        )
    )

    fig.update_layout(

        # ==============================
        # CENTERED TITLE
        # ==============================
        title=dict(
            text='Top 10 Products by Revenue',
            x=0.5,
            xanchor='center',
            y=0.95,
            yanchor='top',
            font=dict(
                color='white',
                size=18
            )
        ),

        # ==============================
        # X AXIS
        # ==============================
        xaxis=dict(
            title=dict(
                text='Revenue (₹)',
                font=dict(
                    color='white',
                    size=13
                )
            ),
            tickfont=dict(
                color='white',
                size=11
            ),
            color='white',
            gridcolor='rgba(255,255,255,0.10)'
        ),

        # ==============================
        # Y AXIS
        # ==============================
        yaxis=dict(
            title=dict(
                text='Product',
                font=dict(
                    color='white',
                    size=13
                )
            ),
            tickfont=dict(
                color='white',
                size=11
            ),
            color='white',
            gridcolor='rgba(255,255,255,0.10)'
        ),

        template='plotly_dark',

        paper_bgcolor='#0e1117',

        plot_bgcolor='#0e1117',

        font=dict(
            color='white'
        ),

        height=350,

        margin=dict(
            l=120,
            r=80,
            t=60,
            b=50
        )
    )

    return fig

def create_daily_sales_distribution(df):
    """Daily sales distribution heatmap"""

    if df.empty:
        return go.Figure()

    # Work on a copy so the original dataframe is not modified
    df = df.copy()

    # Make sure date is datetime
    df['date'] = pd.to_datetime(
        df['date'],
        errors='coerce'
    )

    df = df.dropna(
        subset=['date']
    )

    if df.empty:
        return go.Figure()

    # Create day of week
    df['day_of_week'] = df['date'].dt.dayofweek

    # Create hour
    if 'hour' in df.columns:
        df['hour'] = pd.to_numeric(
            df['hour'],
            errors='coerce'
        ).fillna(
            df['date'].dt.hour
        )
    else:
        df['hour'] = df['date'].dt.hour

    # Make sure units_sold is numeric
    df['units_sold'] = pd.to_numeric(
        df['units_sold'],
        errors='coerce'
    )

    df = df.dropna(
        subset=['units_sold']
    )

    # Create pivot table
    pivot = df.pivot_table(
        values='units_sold',
        index='hour',
        columns='day_of_week',
        aggfunc='mean',
        fill_value=0
    )

    # Ensure all days are present
    pivot = pivot.reindex(
        columns=range(7),
        fill_value=0
    )

    # Create heatmap
    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,

            x=[
                'Mon',
                'Tue',
                'Wed',
                'Thu',
                'Fri',
                'Sat',
                'Sun'
            ],

            y=pivot.index,

            colorscale='Viridis',

            colorbar=dict(
                title=dict(
                    text='Units Sold',
                    font=dict(
                        color='#111827'
                    )
                ),
                tickfont=dict(
                    color='#111827'
                )
            ),

            hovertemplate=(
                '<b>Day:</b> %{x}<br>'
                '<b>Hour:</b> %{y}:00<br>'
                '<b>Units:</b> %{z:.0f}'
                '<extra></extra>'
            )
        )
    )

    # Layout
    fig.update_layout(

        # ==============================
        # CENTERED TITLE
        # ==============================
        title=dict(
            text='Sales Distribution Heatmap',
            x=0.5,
            xanchor='center',
            y=0.95,
            yanchor='top',
            font=dict(
                size=20,
                color='#111827'
            )
        ),

        # ==============================
        # X AXIS
        # ==============================
        xaxis=dict(
            title=dict(
                text='Day of Week',
                font=dict(
                    color='#111827'
                )
            ),

            tickfont=dict(
                color='#111827'
            ),

            showgrid=False,
            zeroline=False
        ),

        # ==============================
        # Y AXIS
        # ==============================
        yaxis=dict(
            title=dict(
                text='Hour of Day',
                font=dict(
                    color='#111827'
                )
            ),

            tickfont=dict(
                color='#111827'
            ),

            showgrid=False,
            zeroline=False
        ),

        # ==============================
        # WHITE BACKGROUND
        # ==============================
        template='plotly_white',

        paper_bgcolor='white',

        plot_bgcolor='white',

        font=dict(
            color='#111827'
        ),

        # ==============================
        # SIZE
        # ==============================
        height=350,

        # ==============================
        # SPACING
        # ==============================
        margin=dict(
            l=70,
            r=70,
            t=70,
            b=60
        ),

        hoverlabel=dict(
            bgcolor='white',
            font=dict(
                color='#111827'
            )
        )
    )

    return fig


def create_monthly_trend(df):
    """Monthly revenue trend chart"""

    if df.empty:
        return go.Figure()

    # Work on a copy
    df = df.copy()

    # Make sure date is datetime
    df['date'] = pd.to_datetime(
        df['date'],
        errors='coerce'
    )

    # Make sure revenue is numeric
    df['revenue'] = pd.to_numeric(
        df['revenue'],
        errors='coerce'
    )

    # Remove invalid rows
    df = df.dropna(
        subset=['date', 'revenue']
    )

    if df.empty:
        return go.Figure()

    # ==============================
    # MONTHLY AGGREGATION
    # ==============================

    monthly = (
        df.groupby(
            df['date'].dt.to_period('M')
        )['revenue']
        .sum()
        .reset_index()
    )

    # Convert Period to datetime
    monthly['date'] = monthly['date'].dt.to_timestamp()

    # Display label
    monthly['month_label'] = monthly['date'].dt.strftime(
        '%b %Y'
    )

    # ==============================
    # CREATE FIGURE
    # ==============================

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=monthly['month_label'],

            y=monthly['revenue'],

            marker=dict(
                color=monthly['revenue'],
                colorscale='Blues',
                showscale=True,

                colorbar=dict(
                    title=dict(
                        text='Revenue',
                        font=dict(
                            color='white'
                        )
                    ),

                    tickfont=dict(
                        color='white'
                    )
                )
            ),

            text=monthly['revenue'].apply(
                lambda x: f'₹{x:,.0f}'
            ),

            textposition='outside',

            textfont=dict(
                color='white',
                size=10
            ),

            hovertemplate=(
                '<b>%{x}</b><br>'
                'Revenue: ₹%{y:,.0f}'
                '<extra></extra>'
            )
        )
    )

    # ==============================
    # LAYOUT
    # ==============================

    fig.update_layout(

        # ==============================
        # CENTERED TITLE
        # ==============================

        title=dict(
            text='Monthly Revenue Trend',

            x=0.5,

            xanchor='center',

            y=0.95,

            yanchor='top',

            font=dict(
                size=20,
                color='white'
            )
        ),

        # ==============================
        # X AXIS
        # ==============================

        xaxis=dict(
            title=dict(
                text='Month',

                font=dict(
                    color='white'
                )
            ),

            tickfont=dict(
                color='white'
            ),

            showgrid=False,

            zeroline=False,

            linecolor='rgba(255,255,255,0.3)',

            tickcolor='white'
        ),

        # ==============================
        # Y AXIS
        # ==============================

        yaxis=dict(
            title=dict(
                text='Revenue (₹)',

                font=dict(
                    color='white'
                )
            ),

            tickfont=dict(
                color='white'
            ),

            showgrid=True,

            gridcolor='rgba(255,255,255,0.12)',

            zeroline=False,

            linecolor='rgba(255,255,255,0.3)',

            tickcolor='white'
        ),

        # ==============================
        # BLACK THEME
        # ==============================

        template='plotly_dark',

        paper_bgcolor='#0e1117',

        plot_bgcolor='#0e1117',

        font=dict(
            color='white'
        ),

        # ==============================
        # SIZE
        # ==============================

        height=350,

        # ==============================
        # MARGINS
        # ==============================

        margin=dict(
            l=70,
            r=70,
            t=70,
            b=60
        ),

        # ==============================
        # HOVER
        # ==============================

        hoverlabel=dict(
            bgcolor='#1f2937',

            font=dict(
                color='white'
            ),

            bordercolor='rgba(255,255,255,0.2)'
        )
    )

    return fig
def create_channel_distribution(df):
    """Channel distribution chart"""

    if df.empty or 'channel' not in df.columns:
        return go.Figure()

    # Work on a copy
    df = df.copy()

    # Make sure revenue is numeric
    df['revenue'] = pd.to_numeric(
        df['revenue'],
        errors='coerce'
    )

    df = df.dropna(
        subset=['channel', 'revenue']
    )

    if df.empty:
        return go.Figure()

    # ==============================
    # CHANNEL AGGREGATION
    # ==============================

    channel_data = (
        df.groupby(
            'channel',
            as_index=False
        )['revenue']
        .sum()
        .sort_values(
            'revenue',
            ascending=False
        )
    )

    # ==============================
    # COLORS
    # ==============================

    colors = [
        '#6366F1',
        '#8B5CF6',
        '#EC4899',
        '#F59E0B',
        '#10B981'
    ]

    chart_colors = [
        colors[i % len(colors)]
        for i in range(len(channel_data))
    ]

    # ==============================
    # CREATE DONUT CHART
    # ==============================

    fig = go.Figure(
        data=[
            go.Pie(
                labels=channel_data['channel'],

                values=channel_data['revenue'],

                hole=0.4,

                marker=dict(
                    colors=chart_colors,

                    line=dict(
                        color='#0e1117',
                        width=2
                    )
                ),

                textinfo='label+percent',

                textposition='auto',

                textfont=dict(
                    color='white',
                    size=12
                ),

                hovertemplate=(
                    '<b>%{label}</b><br>'
                    'Revenue: ₹%{value:,.0f}<br>'
                    'Share: %{percent}'
                    '<extra></extra>'
                )
            )
        ]
    )

    # ==============================
    # LAYOUT
    # ==============================

    fig.update_layout(

        # ==============================
        # CENTERED TITLE
        # ==============================

        title=dict(
            text='Revenue by Channel',

            x=0.5,

            xanchor='center',

            y=0.95,

            yanchor='top',

            font=dict(
                size=20,
                color='white'
            )
        ),

        # ==============================
        # BLACK THEME
        # ==============================

        template='plotly_dark',

        paper_bgcolor='#0e1117',

        plot_bgcolor='#0e1117',

        font=dict(
            color='white'
        ),

        # ==============================
        # SIZE
        # ==============================

        height=350,

        # ==============================
        # LEGEND
        # ==============================

        legend=dict(
            orientation='h',

            yanchor='bottom',

            y=-0.1,

            xanchor='center',

            x=0.5,

            font=dict(
                color='white'
            )
        ),

        # ==============================
        # MARGINS
        # ==============================

        margin=dict(
            l=30,
            r=30,
            t=70,
            b=70
        ),

        # ==============================
        # HOVER
        # ==============================

        hoverlabel=dict(
            bgcolor='#1f2937',

            font=dict(
                color='white'
            ),

            bordercolor='rgba(255,255,255,0.2)'
        )
    )

    return fig

def create_kpi_card(label, value, delta=None, delta_color='normal', icon='📊'):
    """Create a KPI card with consistent styling"""

    if delta:
        if delta_color == 'inverse':
            delta_color = 'inverse'
        else:
            delta_color = 'normal'

    return st.metric(
        label=f"{icon} {label}",
        value=value,
        delta=delta,
        delta_color=delta_color
    )
