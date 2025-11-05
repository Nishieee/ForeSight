"""
ForeSight - AI-Powered Early Warning Advisor
Streamlit Dashboard for Financial Distress Detection
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.append('..')
sys.path.append('.')

# Import modules directly
from fsi_calculator import FSICalculator
from insights import OpenAIInsightGenerator
from visual_utils import (
    plot_fsi_distribution,
    plot_risk_breakdown,
    plot_shap_bar,
    plot_user_timeline
)


# Page configuration
st.set_page_config(
    page_title="ForeSight - Early Warning Advisor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .insight-box {
        background-color: #fff9e6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #f59e0b;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Load pre-computed data."""
    # Try multiple paths
    for data_path in ["data/synthetic_financial_data.csv", "../data/synthetic_financial_data.csv"]:
        if Path(data_path).exists():
            return pd.read_csv(data_path)
    return None


@st.cache_data
def load_fsi_results():
    """Load FSI results."""
    # Try multiple paths
    for fsi_path in ["data/fsi_results.csv", "../data/fsi_results.csv"]:
        if Path(fsi_path).exists():
            return pd.read_csv(fsi_path)
    
    # Compute if not exists
    with st.spinner("Computing FSI scores... This may take a few minutes."):
        config_path = "config/settings.yaml" if Path("config/settings.yaml").exists() else "../config/settings.yaml"
        calculator = FSICalculator(config_path=config_path)
        results = calculator.compute_all_fsi()
        
        # Save for future use
        save_path = "data/fsi_results.csv" if Path("data").exists() else "../data/fsi_results.csv"
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(save_path, index=False)
        
        return results


@st.cache_data
def load_user_explanations():
    """Load SHAP user explanations."""
    # Try multiple paths
    for explanations_path in ["data/user_explanations.json", "../data/user_explanations.json"]:
        if Path(explanations_path).exists():
            with open(explanations_path, 'r') as f:
                return json.load(f)
    return {}


def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<div class="main-header">📊 ForeSight - Early Warning Advisor</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Navigation")
        page = st.radio(
            "Select Page",
            ["📈 Dashboard Overview", "👤 User Insights", "🎯 Risk Analysis"]
        )
        
        st.markdown("---")
        st.header("ℹ️ About")
        st.markdown("""
        **ForeSight** is an AI-powered early-warning system that:
        
        - Detects personalized financial distress signals
        - Explains why risks occur using SHAP
        - Provides actionable prevention advice via OpenAI
        - Combines ML models into a unified FSI score
        """)
    
    # Main content
    if page == "📈 Dashboard Overview":
        show_dashboard()
    elif page == "👤 User Insights":
        show_user_insights()
    elif page == "🎯 Risk Analysis":
        show_risk_analysis()


def show_dashboard():
    """Display main dashboard with portfolio overview."""
    st.header("📈 Portfolio Dashboard")
    
    # Load data
    df = load_data()
    if df is None:
        st.error("Data not found. Please generate data first.")
        return
    
    # Load FSI results
    fsi_results = load_fsi_results()
    
    # Merge with original data
    df_merged = df.merge(fsi_results, on=['user_id', 'month'], how='inner')
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Users",
            df_merged['user_id'].nunique(),
            help="Number of unique users in the portfolio"
        )
    
    with col2:
        avg_fsi = df_merged['FSI'].mean()
        st.metric(
            "Average FSI",
            f"{avg_fsi:.1f}",
            delta=None,
            help="Average Financial Stability Index across all users"
        )
    
    with col3:
        risk_users = df_merged[df_merged['risk_category'].isin(['high_risk', 'early_distress'])]['user_id'].nunique()
        st.metric(
            "Users at Risk",
            risk_users,
            help="Users with FSI < 60"
        )
    
    with col4:
        distress_rate = df_merged['distress_label'].mean()
        st.metric(
            "Distress Rate",
            f"{distress_rate:.1%}",
            help="Percentage of users in distress"
        )
    
    st.markdown("---")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("FSI Distribution")
        fig_dist = plot_fsi_distribution(fsi_results)
        st.plotly_chart(fig_dist, use_container_width=True)
    
    with col2:
        st.subheader("Risk Category Breakdown")
        fig_pie = plot_risk_breakdown(fsi_results)
        st.plotly_chart(fig_pie, use_container_width=True)
    
    # Monthly trend
    st.subheader("Monthly FSI Trend")
    monthly_avg = df_merged.groupby('month')['FSI'].mean()
    st.line_chart(monthly_avg)


def show_user_insights():
    """Display individual user insights with SHAP and OpenAI advice."""
    st.header("👤 User Insights")
    
    # Load data
    df = load_data()
    if df is None:
        st.error("Data not found. Please generate data first.")
        return
    
    fsi_results = load_fsi_results()
    df_merged = df.merge(fsi_results, on=['user_id', 'month'], how='inner')
    user_explanations = load_user_explanations()
    
    # User selection
    col1, col2 = st.columns([1, 3])
    
    with col1:
        unique_users = sorted(df_merged['user_id'].unique())
        user_id = st.selectbox("Select User ID", unique_users)
    
    with col2:
        user_data = df_merged[df_merged['user_id'] == user_id]
        months = user_data['month'].tolist()
        month = st.selectbox("Select Month", months)
    
    # Get user data for selected month
    user_month_data = df_merged[(df_merged['user_id'] == user_id) & (df_merged['month'] == month)].iloc[0]
    
    st.markdown("---")
    
    # User metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("FSI Score", f"{user_month_data['FSI']:.1f}")
    
    with col2:
        st.metric("Risk Probability", f"{user_month_data['risk_probability']:.2%}")
    
    with col3:
        st.metric("Anomaly Score", f"{user_month_data['normalized_anomaly']:.2f}")
    
    with col4:
        st.metric("Risk Label", user_month_data['risk_label'])
    
    st.markdown("---")
    
    # Personalized dynamic threshold information
    st.subheader("🎯 Personalized Risk Assessment")
    
    # Check if dynamic threshold columns exist
    if 'dynamic_threshold' in user_month_data and 'risk_flag' in user_month_data:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            current_fsi = user_month_data['FSI']
            threshold = user_month_data['dynamic_threshold']
            st.metric("Current FSI", f"{current_fsi:.1f}")
        
        with col2:
            threshold_delta = current_fsi - threshold
            delta_color = "normal" if threshold_delta >= 0 else "inverse"
            st.metric(
                "Personalized Threshold", 
                f"{threshold:.1f}",
                delta=f"{threshold_delta:.1f}",
                delta_color=delta_color
            )
        
        with col3:
            risk_flag = user_month_data['risk_flag']
            personalized_label = user_month_data.get('personalized_risk_label', user_month_data['risk_label'])
            
            if risk_flag == 1:
                st.error(f"⚠️ **Alert**: Below personalized threshold!")
                st.write(f"Classification: {personalized_label}")
            else:
                st.success(f"✅ **Normal**: Above personalized threshold")
                st.write(f"Classification: {personalized_label}")
        
        # Display volatility stats
        with st.expander("📊 Personal Baseline Statistics"):
            if 'fsi_mean' in user_month_data and 'fsi_std' in user_month_data:
                st.write(f"**Mean FSI**: {user_month_data['fsi_mean']:.2f}")
                st.write(f"**FSI Volatility (std)**: {user_month_data['fsi_std']:.2f}")
                st.write(f"**Threshold**: {user_month_data['dynamic_threshold']:.2f}")
                st.write(f"**Sensitivity**: 1.5 (alpha parameter)")
                st.info(
                    "💡 This user's risk threshold adapts to their historical stability patterns. "
                    "Users with higher volatility get more tolerant thresholds."
                )
    else:
        st.info("Personalized thresholds not available. Computing...")
    
    st.markdown("---")
    
    # Financial details
    st.subheader("📊 Financial Summary")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Income & Spending**")
        st.write(f"Income: ${user_month_data['income']:,.2f}")
        st.write(f"Total Spending: ${user_month_data['total_spend']:,.2f}")
        st.write(f"Essential: ${user_month_data['essential_spend']:,.2f}")
        st.write(f"Luxury: ${user_month_data['luxury_spend']:,.2f}")
    
    with col2:
        st.markdown("**Credit & Payments**")
        st.write(f"Credit Utilization: {user_month_data['credit_utilization']:.1%}")
        st.write(f"Missed Payments: {int(user_month_data['missed_payments'])}")
    
    st.markdown("---")
    
    # SHAP explanation
    st.subheader("🔍 SHAP Feature Explanation")
    
    key = f"{user_id}_m{month}"
    explanation = user_explanations.get(key)
    
    if explanation:
        shap_data = {
            feat['feature']: feat['shap_value']
            for feat in explanation['top_features']
        }
        
        fig_shap = plot_shap_bar(shap_data, top_n=10)
        st.plotly_chart(fig_shap, use_container_width=True)
        
        # Feature descriptions
        with st.expander("View Top Risk Factors"):
            for feat in explanation['top_features'][:5]:
                direction = "increases" if feat['direction'] == 'increases_risk' else "decreases"
                st.write(f"• **{feat['feature'].replace('_', ' ').title()}**: {direction} risk (SHAP = {feat['shap_value']:.4f})")
    else:
        st.info("SHAP explanation not available for this user.")
    
    # OpenAI insight
    st.markdown("---")
    st.subheader("💡 AI-Generated Insight")
    
    # Use personalized threshold if available, otherwise fallback to static 60
    threshold_for_insight = user_month_data.get('dynamic_threshold', 60)
    is_at_risk = user_month_data.get('risk_flag', user_month_data['FSI'] < 60)
    
    if is_at_risk:  # Show insight for users at risk (below personalized threshold)
        if st.button("Generate AI Insight"):
            with st.spinner("Generating personalized advice..."):
                try:
                    config_path = "config/settings.yaml" if Path("config/settings.yaml").exists() else "../config/settings.yaml"
                    insight_gen = OpenAIInsightGenerator(config_path=config_path)
                    insight = insight_gen.generate_insight(user_id, month)
                    
                    st.markdown(f'<div class="insight-box">{insight}</div>', unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error generating insight: {str(e)}")
                    st.info("Make sure OPENAI_API_KEY is set in your environment.")
    else:
        st.success(f"✅ User is in stable condition (FSI = {user_month_data['FSI']:.1f}). No insights needed.")
    
    # Timeline
    st.markdown("---")
    st.subheader("📈 Financial Timeline")
    
    try:
        fig_timeline = plot_user_timeline(df_merged, user_id)
        st.plotly_chart(fig_timeline, use_container_width=True)
    except Exception as e:
        st.error(f"Error plotting timeline: {str(e)}")


def show_risk_analysis():
    """Display risk analysis and filtering options."""
    st.header("🎯 Risk Analysis")
    
    # Load data
    df = load_data()
    if df is None:
        st.error("Data not found. Please generate data first.")
        return
    
    fsi_results = load_fsi_results()
    df_merged = df.merge(fsi_results, on=['user_id', 'month'], how='inner')
    
    # Filters
    st.subheader("🔍 Filters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        risk_category = st.multiselect(
            "Risk Category",
            options=df_merged['risk_category'].unique(),
            default=df_merged['risk_category'].unique()
        )
    
    with col2:
        fsi_range = st.slider(
            "FSI Range",
            min_value=float(df_merged['FSI'].min()),
            max_value=float(df_merged['FSI'].max()),
            value=(float(df_merged['FSI'].min()), float(df_merged['FSI'].max())),
            step=1.0
        )
    
    with col3:
        show_distress = st.checkbox("Only Show Distressed Users", value=False)
    
    # Apply filters
    filtered_df = df_merged[df_merged['risk_category'].isin(risk_category)]
    filtered_df = filtered_df[
        (filtered_df['FSI'] >= fsi_range[0]) &
        (filtered_df['FSI'] <= fsi_range[1])
    ]
    
    if show_distress:
        filtered_df = filtered_df[filtered_df['distress_label'] == 1]
    
    # Display filtered results
    st.subheader(f"📋 Results: {len(filtered_df)} Records")
    
    # Summary table
    if len(filtered_df) > 0:
        summary_cols = ['user_id', 'month', 'FSI', 'risk_label', 'risk_probability', 'income', 'total_spend', 'credit_utilization']
        st.dataframe(
            filtered_df[summary_cols].head(100),
            use_container_width=True,
            height=400
        )
        
        # Download button
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="Download Filtered Data",
            data=csv,
            file_name="filtered_risk_data.csv",
            mime="text/csv"
        )
    else:
        st.info("No records match the selected filters.")


if __name__ == "__main__":
    main()

