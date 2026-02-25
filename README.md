# FFC Sales Forecasting System: AI-Powered Predictive Analytics

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Regressor-14B3E4?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Data](https://img.shields.io/badge/Synthetic-Data_Generation-green?style=for-the-badge)

## Executive Summary
Fauji Fertilizer Company (FFC) operates an extensive distribution network across Pakistan. Efficient supply chain management in the fertilizer industry necessitates precise anticipation of seasonal demand cycles.

This system establishes a professional Time-Series Forecasting framework designed for the FFC operational ecosystem. It utilizes Gradient Boosted Trees (XGBoost) to predict daily store-level sales volumes, enabling stakeholders to optimize inventory allocation, mitigate stock-out risks during peak agricultural seasons, and enhance logistics efficiency.

---

## Key Features
- **Agricultural Cycle Modeling**: Custom synthetic data engine designed to replicate Pakistani agricultural patterns, specifically the Rabi and Kharif cycles.
- **Advanced Machine Learning Architecture**: Implements an XGBoost Regressor with comprehensive feature engineering, including temporal lags, rolling statistics, and seasonal indicators.
- **Corporate Dashboard**: A high-performance Streamlit application featuring interactive Plotly visualizations, predictive trend analysis, and model interpretability via Feature Importance metrics.
- **Performance Evaluation**: Systematic calculation of industry-standard metrics, including Root Mean Squared Error (RMSE) and Mean Absolute Percentage Error (MAPE).

---

## Technical Architecture

### 1. Data Pipeline (src/data_generator.py & src/processing.py)
- Simulates daily sales across multiple distribution centers.
- Models demand fluctuations aligned with Pakistani sowing and harvesting windows for major crops (Wheat, Rice, Maize).
- Feature engineering extracts temporal components (day, month, seasonality) and historical lags to capture cyclical patterns.

### 2. Predictive Engine (src/model.py)
- **Algorithm**: Extreme Gradient Boosting (XGBoost).
- **Validation Strategy**: Time-series validation split (Training: 2021-2022, Evaluation: 2023).
- **Inference**: Models are serialized for low-latency deployment within the analytical dashboard.

### 3. Analytics Interface (src/app.py)
- Professional interface optimized for executive review, featuring FFC corporate color schemes.
- Real-time forecast generation for localized distribution points.
- Model transparency module demonstrating the primary drivers of sales predictions.

---

## Deployment and Configuration

### 1. Environment Initialization
Clone the repository and configure a dedicated virtual environment:
```powershell
# Create virtual environment
python -m venv ffc_sales_project

# Activate environment
.\ffc_sales_project\Scripts\Activate.ps1

# Install required dependencies
pip install -r requirements.txt
```

### 2. Operational Sequence
Execute the following steps in order to initialize the system:

| Step | Command | Description |
| :--- | :--- | :--- |
| **1. Data Generation** | `python src/data_generator.py` | Generates three years of synthetic sales records. |
| **2. Model Training** | `python src/model.py` | Executes training and generates performance diagnostics. |
| **3. System Launch** | `streamlit run src/app.py` | Deploys the interactive analytical dashboard. |

---

## Performance Diagnostics
The predictive engine is validated using standardized regression metrics:
- **RMSE**: Quantifies the average standard deviation of predictive errors.
- **MAPE**: Provides relative accuracy metrics essential for high-level supply chain planning.

---

## Disclaimer
This project is a technical demonstration of predictive analytics capabilities. It utilizes advanced synthetic data engines to simulate operational logic while ensuring the complete confidentiality of proprietary data.

---
**Technical Lead: [Your Name/GitHub Profile]**  
*Predictive Analytics Solutions for Agricultural Supply Chains*
