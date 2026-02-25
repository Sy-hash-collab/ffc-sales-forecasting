import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate_synthetic_data(start_date='2021-01-01', end_date='2023-12-31', num_stores=5):
    """
    Generates synthetic fertilizer sales data for FFC.
    Simulates seasonal peaks for Rabi (Check Oct-Dec) and Kharif (Check May-Jul) crops.
    """
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    data = []

    stores = [f'Store_{i}' for i in range(1, num_stores + 1)]
    
    # Base baseline sales for each store
    store_baselines = {store: np.random.randint(50, 200) for store in stores}
    
    for date in date_range:
        month = date.month
        day_of_year = date.dayofyear
        
        # Seasonality: 
        # Kharif (Rice, Maize, Cotton): Sowing May-June. Peak fertilizer demand ~ June-July
        # Rabi (Wheat): Sowing Oct-Nov. Peak fertilizer demand ~ Nov-Dec
        
        # Simple seasonal waves using sine functions
        # Peak 1 (Kharif): Center around Day 180 (End June)
        kharif_factor = np.exp(-((day_of_year - 180)**2) / (2 * 30**2)) 
        
        # Peak 2 (Rabi): Center around Day 330 (End Nov) coverage for wheat sowing
        rabi_factor = np.exp(-((day_of_year - 320)**2) / (2 * 30**2))
        
        # Combined seasonality (0 to ~1 scale)
        seasonality = kharif_factor + rabi_factor
        
        # Yearly Trend (slight increase over years)
        days_since_start = (date - datetime.strptime(start_date, '%Y-%m-%d')).days
        trend = days_since_start * 0.05 
        
        # Weekly pattern (higher on weekdays maybe? let's say higher on market days Mon-Sat)
        weekday_factor = 1.2 if date.weekday() < 6 else 0.8
        
        for store in stores:
            base = store_baselines[store]
            
            # Random noise
            noise = np.random.normal(0, 15)
            
            # Calculate Sales
            # Base + (Seasonality * Amplitude) + Trend + Noise
            sales = base + (seasonality * base * 1.5) + trend + noise
            sales = sales * weekday_factor
            
            # Ensure no negative sales
            sales = max(0, int(sales))
            
            data.append([date, store, sales])

    df = pd.DataFrame(data, columns=['Date', 'Store', 'Sales'])
    
    # Add some holiday spikes/drops? Maybe not critical for this demo.
    
    output_dir = 'data'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'ffc_sales.csv')
    df.to_csv(output_path, index=False)
    print(f"Data generated at {output_path}")
    print(df.head())
    print(f"Total Rows: {len(df)}")
    return df

if __name__ == "__main__":
    generate_synthetic_data()
