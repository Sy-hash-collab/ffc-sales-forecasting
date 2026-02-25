# FFC Sales Forecasting Project 🌾

**Predicting Future Store Sales with AI**  
*Developed for FFC Internship Application (Level 6)*

## 📌 Project Overview
Fauji Fertilizer Company (FFC) operates with a massive distribution network dependent on accurate sales forecasting. This expert-level project demonstrates a **Time-Series Forecasting System** designed to predict fertilizer sales, optimizing inventory and supply chain decisions.

### Key Features
- **Synthetic Data Generation**: Simulates realistic Pakistani agricultural cycles (Rabi/Kharif seasons).
- **Advanced Modeling**: Uses **XGBoost Regressor** with lag features and rolling window statistics.
- **Interactive Dashboard**: A professional **Streamlit** app for executives to visualize trends and future forecasts.

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Create virtual environment (optional but recommended)
python -m venv venv
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Data
Create 3 years of synthetic daily sales data for 5 stores.
```bash
python src/data_generator.py
```

### 3. Train Model
Train the XGBoost model on historical data (2021-2022) and validate on 2023.
```bash
python src/model.py
```
*Outputs RMSE/MAPE metrics and saves model to `models/` directory.*

### 4. Run Dashboard
Launch the interactive application.
```bash
streamlit run src/app.py
```

## 📊 Technical Details
- **Data**: 5 Stores, Daily resolution. Features include Date components, peaks for Wheat/Rice seasons.
- **Model**: XGBoost (Gradient Boosting Trees).
- **Metrics**: RMSE (Root Mean Squared Error), MAPE (Mean Absolute Percentage Error).

## ⚠️ Disclaimer
This project uses **synthetic data** generated to mimic FFC's business logic, as proprietary sales data is confidential.
