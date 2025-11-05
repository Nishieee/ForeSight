"""
Visualization Utilities for ForeSight Dashboard
Helper functions for creating charts and plots.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np


def plot_fsi_distribution(df: pd.DataFrame, title: str = "Financial Stability Index Distribution") -> go.Figure:
    """
    Create histogram of FSI scores with color-coded risk zones.
    
    Args:
        df: DataFrame with FSI scores
        title: Chart title
        
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    # Define risk zones
    risk_zones = [
        {'min': 85, 'max': 100, 'color': '#22c55e', 'label': '🟢 Stable'},
        {'min': 60, 'max': 85, 'color': '#eab308', 'label': '🟡 Watchlist'},
        {'min': 40, 'max': 60, 'color': '#f97316', 'label': '🟠 Early Distress'},
        {'min': 0, 'max': 40, 'color': '#ef4444', 'label': '🔴 High Risk'}
    ]
    
    # Add histogram
    fig.add_trace(go.Histogram(
        x=df['FSI'],
        nbinsx=50,
        marker_color='rgba(59, 130, 246, 0.7)',
        name='FSI Distribution',
        hovertemplate='FSI: %{x}<br>Count: %{y}<extra></extra>'
    ))
    
    # Add risk zone rectangles
    for zone in risk_zones:
        fig.add_vrect(
            x0=zone['min'],
            x1=zone['max'],
            fillcolor=zone['color'],
            opacity=0.1,
            layer="below",
            line_width=0,
            annotation_text=zone['label'],
            annotation_position="top left"
        )
    
    fig.update_layout(
        title=title,
        xaxis_title="Financial Stability Index",
        yaxis_title="Number of Users",
        hovermode='closest',
        height=400
    )
    
    return fig


def plot_risk_breakdown(df: pd.DataFrame) -> go.Figure:
    """
    Create pie chart of risk category distribution.
    
    Args:
        df: DataFrame with risk categories
        
    Returns:
        Plotly figure
    """
    risk_counts = df['risk_label'].value_counts()
    
    # Define colors for each category
    color_map = {
        '🟢 Stable': '#22c55e',
        '🟡 Watchlist': '#eab308',
        '🟠 Early Distress': '#f97316',
        '🔴 High Risk': '#ef4444'
    }
    
    colors = [color_map.get(label, '#94a3b8') for label in risk_counts.index]
    
    fig = go.Figure(data=[go.Pie(
        labels=risk_counts.index,
        values=risk_counts.values,
        hole=0.4,
        marker_colors=colors,
        textinfo='label+percent',
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
    )])
    
    fig.update_layout(
        title="Risk Category Breakdown",
        height=400
    )
    
    return fig


def plot_shap_waterfall(shap_values: list, feature_names: list, expected_value: float, title: str = "SHAP Explanation") -> go.Figure:
    """
    Create waterfall plot for SHAP values.
    
    Args:
        shap_values: List of SHAP contribution values
        feature_names: List of feature names
        expected_value: Base prediction value
        title: Chart title
        
    Returns:
        Plotly figure
    """
    fig = go.Figure(go.Waterfall(
        orientation="v",
        measure=["absolute"] + ["relative"] * len(feature_names) + ["total"],
        x=["Base Value"] + feature_names + ["Final Prediction"],
        y=[expected_value] + shap_values + [expected_value + sum(shap_values)],
        connector={"line": {"color": "rgb(63, 63, 63)"}},
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Features",
        yaxis_title="SHAP Value",
        height=500
    )
    
    return fig


def plot_shap_bar(shap_data: dict, top_n: int = 10) -> go.Figure:
    """
    Create horizontal bar chart of top SHAP feature contributions.
    
    Args:
        shap_data: Dict with feature contributions
        top_n: Number of top features to show
        
    Returns:
        Plotly figure
    """
    # Sort by absolute value
    sorted_features = sorted(
        shap_data.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:top_n]
    
    features, values = zip(*sorted_features)
    
    # Color positive and negative differently
    colors = ['#ef4444' if v > 0 else '#22c55e' for v in values]
    
    fig = go.Figure(data=[go.Bar(
        x=values,
        y=features,
        orientation='h',
        marker_color=colors,
        text=[f"{v:.4f}" for v in values],
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>SHAP Value: %{x:.4f}<extra></extra>'
    )])
    
    fig.update_layout(
        title="Top Feature Contributions (SHAP)",
        xaxis_title="SHAP Value",
        yaxis_title="Features",
        height=400,
        yaxis={'categoryorder': 'total ascending'}
    )
    
    return fig


def plot_user_timeline(user_data: pd.DataFrame, user_id: int) -> go.Figure:
    """
    Create time series plot of user's financial metrics over time.
    
    Args:
        user_data: DataFrame with user data across months
        user_id: User ID to plot
        
    Returns:
        Plotly figure
    """
    user_df = user_data[user_data['user_id'] == user_id].sort_values('month')
    
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('Income & Spending', 'FSI Score', 'Credit Utilization'),
        vertical_spacing=0.1,
        row_heights=[0.4, 0.3, 0.3]
    )
    
    # Income & Spending
    fig.add_trace(
        go.Scatter(x=user_df['month'], y=user_df['income'], name='Income', line=dict(color='#22c55e')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=user_df['month'], y=user_df['total_spend'], name='Spending', line=dict(color='#ef4444')),
        row=1, col=1
    )
    
    # FSI Score
    fig.add_trace(
        go.Scatter(x=user_df['month'], y=user_df['FSI'], name='FSI', fill='tozeroy', line=dict(color='#3b82f6')),
        row=2, col=1
    )
    
    # Add personalized dynamic threshold if available
    if 'dynamic_threshold' in user_df.columns:
        threshold_value = user_df['dynamic_threshold'].iloc[0]  # Same for all months for a user
        fig.add_hline(
            y=threshold_value, 
            line_dash="dash", 
            line_color="red", 
            opacity=0.8,
            line_width=2,
            annotation_text=f"Personalized Threshold: {threshold_value:.1f}",
            annotation_position="top right",
            row=2, col=1
        )
    
    # Add static FSI thresholds as reference
    fig.add_hline(y=85, line_dash="dot", line_color="green", opacity=0.3, row=2, col=1, annotation_text="Stable")
    fig.add_hline(y=60, line_dash="dot", line_color="yellow", opacity=0.3, row=2, col=1, annotation_text="Watchlist")
    fig.add_hline(y=40, line_dash="dot", line_color="orange", opacity=0.3, row=2, col=1, annotation_text="Early Distress")
    
    # Credit Utilization
    fig.add_trace(
        go.Scatter(x=user_df['month'], y=user_df['credit_utilization']*100, name='Credit Util %', 
                  fill='tozeroy', line=dict(color='#8b5cf6')),
        row=3, col=1
    )
    
    fig.update_xaxes(title_text="Month", row=3, col=1)
    fig.update_yaxes(title_text="Amount ($)", row=1, col=1)
    fig.update_yaxes(title_text="FSI Score", row=2, col=1)
    fig.update_yaxes(title_text="Percentage (%)", row=3, col=1)
    
    fig.update_layout(
        title=f"User {user_id} Financial Timeline",
        height=700,
        showlegend=True
    )
    
    return fig


def plot_portfolio_summary(df: pd.DataFrame) -> go.Figure:
    """
    Create dashboard summary with key metrics.
    
    Args:
        df: DataFrame with user data
        
    Returns:
        Plotly figure with subplots
    """
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('Average FSI by Month', 'Risk Distribution', 'Avg Income vs Spending'),
        specs=[[{"type": "scatter"}, {"type": "bar"}, {"type": "scatter"}]]
    )
    
    # FSI trend over time
    monthly_fsi = df.groupby('month')['FSI'].mean()
    fig.add_trace(
        go.Scatter(x=monthly_fsi.index, y=monthly_fsi.values, name='Avg FSI', mode='lines+markers'),
        row=1, col=1
    )
    
    # Risk distribution
    risk_dist = df['risk_category'].value_counts()
    fig.add_trace(
        go.Bar(x=risk_dist.index, y=risk_dist.values, name='Risk Count'),
        row=1, col=2
    )
    
    # Income vs Spending scatter
    avg_data = df.groupby('user_id').agg({
        'income': 'mean',
        'total_spend': 'mean'
    }).sample(min(1000, len(df.groupby('user_id'))))
    
    fig.add_trace(
        go.Scatter(x=avg_data['income'], y=avg_data['total_spend'], mode='markers', name='Users',
                  hovertemplate='Income: $%{x}<br>Spending: $%{y}<extra></extra>'),
        row=1, col=3
    )
    
    fig.update_layout(height=400, showlegend=False)
    
    return fig

