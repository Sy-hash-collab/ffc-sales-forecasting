import pandas as pd
import numpy as np

def create_features(df):
    """
    Creates time-series features for forecasting.
    Expects a DataFrame with 'Date', 'Store', 'Sales' columns.
    """
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(['Store', 'Date'])
    
    # Date Features
    df['day_of_week'] = df['Date'].dt.dayofweek
    df['day_of_year'] = df['Date'].dt.dayofyear
    df['month'] = df['Date'].dt.month
    df['year'] = df['Date'].dt.year
    df['quarter'] = df['Date'].dt.quarter
    
    # Seasonality Flags (Domain Knowledge: Rabi/Kharif)
    # Kharif Peak: June-July (Month 6, 7)
    # Rabi Peak: Nov-Dec (Month 11, 12)
    df['is_peak_season'] = df['month'].isin([6, 7, 11, 12]).astype(int)
    
    # Lag Features (Past Sales)
    # Assuming we want to predict 1 day ahead, but in reality we might want longer horizons.
    # For this demo, let's use 7, 14, 30 day lags to capture weekly/monthly trends.
    lags = [1, 7, 14, 30]
    for lag in lags:
        df[f'lag_{lag}'] = df.groupby('Store')['Sales'].shift(lag)
        
    # Rolling Means (Window stats)
    windows = [7, 30]
    for window in windows:
        df[f'rolling_mean_{window}'] = df.groupby('Store')['Sales'].shift(1).rolling(window=window).mean()
        df[f'rolling_std_{window}'] = df.groupby('Store')['Sales'].shift(1).rolling(window=window).std()
        
    # Drop rows with NaN values created by lag/rolling features
    df = df.dropna()
    
    return df

def prepare_data(filepath='data/ffc_sales.csv'):
    df = pd.read_csv(filepath)
    df = create_features(df)
    return df

if __name__ == "__main__":
    df = prepare_data()
    print(df.head())
    print(df.columns)
