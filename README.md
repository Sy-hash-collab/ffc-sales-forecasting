# 🌱 FFC Sales Forecasting System: AI-Powered Predictive Analytics

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Regressor-14B3E4?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Data](https://img.shields.io/badge/Synthetic-Data_Generation-green?style=for-the-badge)

## 📌 Executive Summary
Fauji Fertilizer Company (FFC) manages one of Pakistan's most extensive distribution networks. Efficient supply chain management in the fertilizer industry requires precise anticipation of seasonal demand. 

This project implements an **Expert-Level Time-Series Forecasting System** tailored for the FFC ecosystem. It leverages Gradient Boosted Trees (XGBoost) to predict daily store-level sales, enabling stakeholders to optimize inventory levels, reduce stock-outs during peak seasons, and streamline logistics operations.

---

## ✨ Key Features
- **🌾 Agricultural Cycle Simulation**: Custom synthetic data engine that mimics realistic Pakistani agricultural patterns, specifically the **Rabi** and **Kharif** cycles.
- **🤖 Advanced ML Architecture**: Employs an XGBoost Regressor engine with deep feature engineering, including temporal lags, rolling averages, and seasonal indicators.
- **📊 Executive Dashboard**: A high-performance Streamlit application featuring interactive Plotly visualizations, predictive trend analysis, and model explainability (Feature Importance).
- **📈 KPI Tracking**: Automated calculation of business-critical metrics such as RMSE (Root Mean Squared Error) and MAPE (Mean Absolute Percentage Error).

---

## 🛠️ Technical Architecture

### 1. Data Pipeline (`src/data_generator.py` & `src/processing.py`)
- Simulates daily sales for multiple distribution centers (stores).
- Incorporates demand spikes aligned with Pakistani sowing and harvesting seasons (Wheat, Rice, Maize).
- Feature engineering extracts date-based features (day of week, month, seasonality) and historical lag features to capture autocorrelation.

### 2. Predictive Modeling (`src/model.py`)
- **Algorithm**: Extreme Gradient Boosting (XGBoost).
- **Validation Strategy**: Time-series split (Historical training on 2021-2022, Validation on 2023).
- **Persistence**: Models are serialized to JSON/PKL for high-speed inference in the production dashboard.

### 3. Interactive Interface (`src/app.py`)
- Professional UI/UX designed with FFC corporate branding (Green/White/Red).
- Real-time forecasting generation for any selected store.
- Visual breakdown of "Model Insights" to explain which factors are driving sales predictions.

---

## 🚀 Deployment & Quick Start

### 1. Environment Setup
Clone the repository and initialize a virtual environment:
```powershell
# Create environment
python -m venv ffc_sales_project

# Activate environment
.\ffc_sales_project\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 2. Execution Pipeline
Follow this sequence to initialize the system:

| Step | Command | Description |
| :--- | :--- | :--- |
| **1. Data Generation** | `python src/data_generator.py` | Generates 3 years of synthetic sales history. |
| **2. Model Training** | `python src/model.py` | Trains the AI engine and generates performance telemetry. |
| **3. Launch Dashboard** | `streamlit run src/app.py` | Starts the interactive web application. |

---

## 📊 Performance Metrics
The system is evaluated using standard regression diagnostics:
- **RMSE**: Quantifies the average deviation in sales volume.
- **MAPE**: Provides a percentage-based accuracy indicator, crucial for business planning.

---

## ⚠️ Disclaimer
This project is developed as a technical demonstration for FFC. It utilizes advanced synthetic data engines designed to simulate business logic while maintaining the confidentiality of proprietary FFC operational data.

---
**Developed by [Your Name/GitHub Profile]**  
*Strategic AI Solutions for Agricultural Supply Chains*
