import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import pickle
import os
import json
from processing import prepare_data
from datetime import datetime

def train_model():
    print("Loading and processing data...")
    df = prepare_data('data/ffc_sales.csv')
    
    # Train/Test Split based on time
    # Train: 2023-2025
    # Test: 2026
    train = df[df['year'] < 2026]
    test = df[df['year'] == 2026]
    
    features = [col for col in df.columns if col not in ['Date', 'Sales', 'Store']]
    target = 'Sales'
    
    X_train = train[features]
    y_train = train[target]
    X_test = test[features]
    y_test = test[target]
    
    print(f"Training on {len(X_train)} samples, Testing on {len(X_test)} samples.")
    
    model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=5,
        early_stopping_rounds=50,
        objective='reg:squarederror',
        n_jobs=-1
    )
    
    print("Training XGBoost model...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=100
    )
    
    # Evaluation
    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds) ** 0.5
    mape = mean_absolute_percentage_error(y_test, preds)
    
    print(f"Test RMSE: {rmse:.2f}")
    print(f"Test MAPE: {mape:.2%}")
    
    # Save model, features and metrics
    model_dir = 'models'
    os.makedirs(model_dir, exist_ok=True)
    
    # 1. Save Metrics
    metrics = {
        'rmse': round(rmse, 2),
        'mape': round(mape, 4),
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M')
    }
    with open(os.path.join(model_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f)
        
    # 2. Save Trained Model (Used by Dashboard)
    model.save_model(os.path.join(model_dir, 'xgboost_sales.json'))
    
    # 3. Save Feature Names (Used by Dashboard for consistency)
    with open(os.path.join(model_dir, 'features.pkl'), 'wb') as f:
        pickle.dump(features, f)
        
    print(f"Success: Model, features, and metrics saved to {model_dir}/")

if __name__ == "__main__":
    train_model()
