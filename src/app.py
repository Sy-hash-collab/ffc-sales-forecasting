import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import xgboost as xgb
import os
import pickle
from processing import prepare_data

# Page Config
st.set_page_config(
    page_title="FFC Sales Forecasting | AI Powered",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for "Expert Level" Look (FFC Branding colors: Green/White/Red)
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stAppHeader {
        background-color: #004d00;
        color: white;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 25px;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        border-left: 5px solid #004d00;
        margin-bottom: 20px;
    }
    .metric-value {
        font-size: 28px;
        font-weight: bold;
        color: #004d00;
    }
    .metric-label {
        font-size: 16px;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
        color: #004d00;
    }
    .stButton>button {
        background-color: #004d00;
        color: white;
        border-radius: 20px;
        padding: 10px 25px;
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #006400;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    </style>
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
    st.title("🌱 FFC Sales Forecasting Dashboard")
    st.markdown("### Predicting Future Store Sales with AI (XGBoost)")
    st.markdown("---")
    
    # Load Data & Model
    try:
        df = load_data()
        model = load_model()
        feature_names = load_features()
        metrics = load_metrics()
    except Exception as e:
        st.error(f"Error loading system: {e}")
        st.warning("Please ensure data is generated and model is trained.")
        return

    # Sidebar
    st.sidebar.header("🎛️ Dashboard Controls")
    
    stores = df['Store'].unique()
    selected_store = st.sidebar.selectbox("Select Target Store", stores)
    
    if metrics:
        st.sidebar.markdown("---")
        st.sidebar.subheader("📉 Model Performance")
        st.sidebar.write(f"**Test RMSE:** {metrics['rmse']}")
        st.sidebar.write(f"**Test MAPE:** {metrics['mape']*100:.2f}%")
        st.sidebar.caption(f"Last trained: {metrics['training_date']}")
    
    # KPIs
    store_data = df[df['Store'] == selected_store].copy()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        total_sales = store_data['Sales'].sum()
        st.markdown(f'''
            <div class="metric-card">
                <div class="metric-label">Total Sales (3 Yrs)</div>
                <div class="metric-value">{total_sales:,.0f} tons</div>
            </div>
        ''', unsafe_allow_html=True)
    with col2:
        avg_sales = store_data['Sales'].mean()
        st.markdown(f'''
            <div class="metric-card">
                <div class="metric-label">Avg Daily Sales</div>
                <div class="metric-value">{avg_sales:,.1f} tons</div>
            </div>
        ''', unsafe_allow_html=True)
    with col3:
        max_sales = store_data['Sales'].max()
        st.markdown(f'''
            <div class="metric-card">
                <div class="metric-label">Peak Sale Day</div>
                <div class="metric-value">{max_sales:,.0f} tons</div>
            </div>
        ''', unsafe_allow_html=True)

    st.markdown("---")

    # Forecasting Interaction
    st.subheader(f"Sales Trend & Forecast: {selected_store}")
    
    # Plotting Historical
    fig = px.line(store_data, x='Date', y='Sales', title=f"Historical Sales - {selected_store}")
    fig.update_layout(xaxis_title="Date", yaxis_title="Sales (Tons)", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)
    
    # Make Future Prediction (Demo: Predict next 30 days)
    st.markdown("### 🔮 Future Forecast (Next 30 Days)")
    
    if st.button("Generate Forecast"):
        last_date = store_data['Date'].max()
        future_dates = pd.date_range(last_date + timedelta(days=1), periods=30)
        
        future_df = pd.DataFrame({'Date': future_dates, 'Store': selected_store, 'Sales': 0}) # Dummy sales
        
        # We need historical data to generate lags for future prediction.
        # This is a complex recursive step for correct time-series forecasting.
        # For this DEMO/Internship project, we will simplify by appending future rows to past and re-calculating features.
        
        # Combine past (last 60 days to be safe for 30d lags) and future
        combined = pd.concat([store_data.tail(60), future_df], ignore_index=True)
        
        # Re-generate features using the processing logic (we need to import the function)
        from processing import create_features
        combined_processed = create_features(combined)
        
        # Filter for the prediction window
        pred_data = combined_processed[combined_processed['Date'] > last_date].copy()
        
        if pred_data.empty:
             st.error("Feature engineering resulted in empty prediction set. (Likely due to lag creation on short horizon).")
        else:
            X_pred = pred_data[feature_names]
            preds = model.predict(X_pred)
            
            pred_data['Predicted_Sales'] = preds
            
            # Plot Forecast
            fig_pred = go.Figure()
            # Historical tail
            fig_pred.add_trace(go.Scatter(
                x=store_data['Date'].tail(90), 
                y=store_data['Sales'].tail(90), 
                mode='lines', 
                name='Historical Sales',
                line=dict(color='#888', width=2)
            ))
            # Forecast
            fig_pred.add_trace(go.Scatter(
                x=pred_data['Date'], 
                y=pred_data['Predicted_Sales'], 
                mode='lines+markers', 
                name='AI Forecast', 
                line=dict(color='#004d00', width=3, dash='solid'),
                marker=dict(size=6, color='#004d00')
            ))
            
            fig_pred.update_layout(
                title=dict(text=f"🏢 30-Day Predictive Analysis: {selected_store}", font=dict(size=20)),
                template="plotly_white",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                xaxis_title="Timeline",
                yaxis_title="Quantity (Tons)",
                hovermode="x unified"
            )
            st.plotly_chart(fig_pred, use_container_width=True)
            
            # Show Data
            st.dataframe(pred_data[['Date', 'Predicted_Sales']].reset_index(drop=True))

    # Insights / Explainability
    st.markdown("---")
    st.subheader("Model Insights")
    
    # Feature Importance
    importance = model.feature_importances_
    feat_imp = pd.DataFrame({'Feature': feature_names, 'Importance': importance}).sort_values('Importance', ascending=True)
    
    fig_imp = px.bar(feat_imp, x='Importance', y='Feature', orientation='h', title="Feature Importance (XGBoost)")
    st.plotly_chart(fig_imp, use_container_width=True)

if __name__ == "__main__":
    from datetime import timedelta
    main()
