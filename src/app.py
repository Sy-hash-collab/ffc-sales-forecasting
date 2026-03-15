import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import xgboost as xgb
import os
import pickle
from processing import prepare_data
from datetime import timedelta

# Corporate Enterprise Page Config
st.set_page_config(
    page_title="FFC Sales Forecaster",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Enterprise UI
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* Global Typography Reset */
    html, body, [class*="css"], [class*="st-"] {
        font-family: 'Inter', sans-serif !important;
    }
    
    /* App Environment Background */
    .stApp {
        background-color: #f8faf9;
    }
    
    /* Header Transparency */
    header[data-testid="stHeader"] {
        background-color: transparent !important;
    }

    /* Corporate Branding Decoration Bar */
    .decoration-bar {
        height: 6px;
        width: 100%;
        background: linear-gradient(90deg, #0a3f20 0%, #2e8b57 50%, #facc15 100%);
        position: fixed;
        top: 0;
        left: 0;
        z-index: 999999;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e5e7eb;
        box-shadow: 2px 0 10px rgba(0,0,0,0.02);
    }
    
    /* High-Fidelity KPI Cards */
    .metric-container {
        display: flex;
        gap: 1.5rem;
        margin-bottom: 2.5rem;
        margin-top: 1.5rem;
    }
    .metric-card {
        background: #ffffff;
        padding: 1.75rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -2px rgba(0, 0, 0, 0.05);
        border: 1px solid #f3f4f6;
        flex: 1;
        position: relative;
        overflow: hidden;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 20px -5px rgba(0, 0, 0, 0.08), 0 4px 6px -4px rgba(0, 0, 0, 0.05);
        border-color: #e5e7eb;
    }
    /* Top Accent Line on Cards */
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 4px;
        background: linear-gradient(90deg, #0a3f20 0%, #2e8b57 100%);
    }
    .metric-label {
        font-size: 0.85rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 0.75rem;
        display: block;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: 800;
        color: #0f172a;
        line-height: 1.1;
        letter-spacing: -0.03em;
    }
    .metric-unit {
        font-size: 1.1rem;
        color: #64748b;
        font-weight: 500;
        margin-left: 0.35rem;
        letter-spacing: 0;
    }
    
    /* Typography Overrides */
    .stMarkdown p, .stMarkdown span {
        color: #334155;
    }
    .stMarkdown h1 {
        color: #0a3f20 !important;
        font-weight: 800 !important;
        font-size: 2.75rem !important;
        letter-spacing: -0.04em !important;
        margin-bottom: 0.25rem !important;
        padding-top: 1rem !important;
    }
    h2, h3 {
        color: #0a3f20 !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em !important;
    }
    .system-subtitle {
        font-size: 1.15rem;
        color: #64748b;
        font-weight: 400;
        margin-bottom: 2rem;
        margin-top: 0rem;
    }
    .section-desc {
        color: #64748b;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
        font-weight: 400;
    }
    
    /* Primary CTA Button Polish */
    .stButton>button {
        background-color: #0a3f20;
        color: white;
        border-radius: 8px;
        padding: 0.75rem 1.75rem;
        font-weight: 600;
        font-size: 1rem;
        border: 1px solid #0a3f20;
        box-shadow: 0 4px 6px -1px rgba(10, 63, 32, 0.15);
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
        width: auto;
    }
    .stButton>button:hover {
        background-color: #115e32;
        border-color: #115e32;
        box-shadow: 0 6px 12px -2px rgba(17, 94, 50, 0.3);
        color: white;
        transform: translateY(-1px);
    }
    .stButton>button:active {
        transform: scale(0.98) translateY(0);
    }
    
    /* Divider Logic */
    hr {
        border-top: 1px solid #e2e8f0;
        margin: 3rem 0;
    }
    
    /* Sidebar Details */
    .sidebar-heading {
        color: #0a3f20;
        font-weight: 700;
        font-size: 1.25rem;
        margin-bottom: 1rem;
    }
    .sidebar-subheading {
        color: #64748b;
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.75rem;
        margin-top: 1.5rem;
    }
    
    /* DataFrame Integration */
    [data-testid="stDataFrame"] {
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
    }
    </style>
    
    <!-- Top Branding Accent -->
    <div class="decoration-bar"></div>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    df = pd.read_csv('data/ffc_sales.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    return df

@st.cache_resource
def load_model():
    model = xgb.XGBRegressor()
    model.load_model('models/xgboost_sales.json')
    return model

@st.cache_resource
def load_metrics():
    import json
    try:
        with open('models/metrics.json', 'r') as f:
            return json.load(f)
    except:
        return None

@st.cache_resource
def load_features():
    with open('models/features.pkl', 'rb') as f:
        return pickle.load(f)

def main():
    # Hero Section
    st.markdown("<h1>FFC Sales Forecasting System</h1>", unsafe_allow_html=True)
    st.markdown("<p class='system-subtitle'>Predictive inventory analytics powered by Extreme Gradient Boosting for Fauji Fertilizer Company.</p>", unsafe_allow_html=True)
    
    # Initialization
    try:
        df = load_data()
        model = load_model()
        feature_names = load_features()
        metrics = load_metrics()
    except FileNotFoundError as e:
        st.error(f"System Component Missing: {e}")
        st.info("Execute data generation and model training scripts to initialize the core system.")
        return
    except Exception as e:
        st.error(f"Initialization Error: {e}")
        return

    # Sidebar Configuration
    st.sidebar.markdown("<div class='sidebar-heading'>System Configuration</div>", unsafe_allow_html=True)
    stores = sorted(df['Store'].unique())
    selected_store = st.sidebar.selectbox("Target Distribution Center", stores)
    
    if metrics:
        st.sidebar.markdown("<hr style='margin: 1.5rem 0;'>", unsafe_allow_html=True)
        st.sidebar.markdown("<div class='sidebar-subheading'>Model Diagnostics</div>", unsafe_allow_html=True)
        st.sidebar.write(f"**Root Mean Squared Error (RMSE):** {metrics['rmse']}")
        st.sidebar.write(f"**Mean Abs. Percentage Error:** {metrics['mape']*100:.2f}%")
        st.sidebar.caption(f"System Trained: {metrics['training_date']}")
    
    # Data Processing
    store_data = df[df['Store'] == selected_store].copy()
    total_sales = store_data['Sales'].sum()
    avg_sales = store_data['Sales'].mean()
    max_sales = store_data['Sales'].max()
    
    # Executive KPIs
    st.markdown(f'''
        <div class="metric-container">
            <div class="metric-card">
                <span class="metric-label">Total Volume Distributed</span>
                <span class="metric-value">{total_sales:,.0f}<span class="metric-unit">tons</span></span>
            </div>
            <div class="metric-card">
                <span class="metric-label">Average Daily Demand</span>
                <span class="metric-value">{avg_sales:,.1f}<span class="metric-unit">tons</span></span>
            </div>
            <div class="metric-card">
                <span class="metric-label">Peak Historical Output</span>
                <span class="metric-value">{max_sales:,.0f}<span class="metric-unit">tons</span></span>
            </div>
        </div>
    ''', unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # Global Chart Presentation Rules
    chart_font = dict(family="Inter, sans-serif", size=13, color="#475569")
    chart_layout = dict(
        template="plotly_white",
        margin=dict(l=10, r=20, t=50, b=20),
        hovermode="x unified",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False, title="", color="#64748b", tickfont=dict(color="#64748b")),
        yaxis=dict(showgrid=True, gridcolor='#f1f5f9', title="Volume (Tons)", color="#64748b", tickfont=dict(color="#64748b")),
        font=chart_font
    )
    
    # Application State - Historical Review
    st.markdown(f"<h3>Sales Trajectory: {selected_store.replace('_', ' ')}</h3>", unsafe_allow_html=True)
    
    fig_hist = px.line(store_data, x='Date', y='Sales')
    fig_hist.update_traces(line_color='#94a3b8', line_width=2.5)
    fig_hist.update_layout(**chart_layout, title=dict(text="Historical Performance Archive", font=dict(color="#0f172a", size=18, family="Inter")))
    st.plotly_chart(fig_hist, width="stretch")
    
    # Subroutine: Predict Future Values
    st.markdown("<h3 style='margin-top: 2rem;'>Predictive Forecast Engine</h3>", unsafe_allow_html=True)
    st.markdown("<p class='section-desc'>Execute the machine learning engine to project demand and mitigate stock-outs for the next 30 operational days.</p>", unsafe_allow_html=True)
    
    if st.button("Initialize Forecast Sequence"):
        with st.spinner('Calculating multivariate time-series projections...'):
            last_date = store_data['Date'].max()
            future_dates = pd.date_range(last_date + timedelta(days=1), periods=30)
            
            future_df = pd.DataFrame({'Date': future_dates, 'Store': selected_store, 'Sales': 0})
            combined = pd.concat([store_data.tail(60), future_df], ignore_index=True)
            
            from processing import create_features
            combined_processed = create_features(combined)
            
            pred_data = combined_processed[combined_processed['Date'] > last_date].copy()
            
            if pred_data.empty:
                 st.error("Feature matrix generation failed. Verify historical trailing data limits.")
            else:
                X_pred = pred_data[feature_names]
                preds = model.predict(X_pred)
                pred_data['Predicted_Sales'] = preds
                
                # Projection Visualization
                fig_pred = go.Figure()
                
                # Historical Base line (Last 60 days)
                fig_pred.add_trace(go.Scatter(
                    x=store_data['Date'].tail(60), 
                    y=store_data['Sales'].tail(60), 
                    mode='lines', 
                    name='Historical Standard',
                    line=dict(color='#cbd5e1', width=3)
                ))
                
                # Projected Line
                fig_pred.add_trace(go.Scatter(
                    x=pred_data['Date'], 
                    y=pred_data['Predicted_Sales'], 
                    mode='lines+markers', 
                    name='Algorithmic Projection', 
                    line=dict(color='#0a3f20', width=3.5, dash='solid'),
                    marker=dict(size=7, color='#0a3f20', symbol='circle', line=dict(color='white', width=1))
                ))
                
                fig_pred.update_layout(
                    title=dict(text="30-Day Forward Trajectory", font=dict(color="#0f172a", size=18, family="Inter")),
                    legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1, title="", font=chart_font),
                    **chart_layout
                )
                st.plotly_chart(fig_pred, width="stretch")
                
                # Exact Quantities Matrix
                with st.expander("Review Raw Projection Matrix"):
                    st.dataframe(
                        pred_data[['Date', 'Predicted_Sales']].rename(columns={'Predicted_Sales': 'Projected Volume (Tons)'}).reset_index(drop=True),
                        width="stretch"
                    )

    # Attribution Logic
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<h3>Algorithmic Attribution</h3>", unsafe_allow_html=True)
    st.markdown("<p class='section-desc'>Relative importance of contextual features utilized by the XGBoost trees to determine the final prediction.</p>", unsafe_allow_html=True)
    
    importance = model.feature_importances_
    feat_imp = pd.DataFrame({'Feature': feature_names, 'Importance': importance}).sort_values('Importance', ascending=True)
    feat_imp['Feature'] = feat_imp['Feature'].str.replace('_', ' ').str.title()
    
    fig_imp = px.bar(feat_imp, x='Importance', y='Feature', orientation='h')
    fig_imp.update_traces(marker_color='#2e8b57', opacity=0.9, width=0.6)
    fig_imp.update_layout(**chart_layout, height=450)
    fig_imp.update_xaxes(title="Information Gain (Weight)", showgrid=True, gridcolor='#f1f5f9', color="#64748b")
    fig_imp.update_yaxes(title="", showgrid=False, color="#0f172a", tickfont=dict(weight="bold", size=12))
    st.plotly_chart(fig_imp, width="stretch")

if __name__ == "__main__":
    main()
