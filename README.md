# FFC Sales Forecasting System: AI-Powered Predictive Analytics

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Regressor-14B3E4?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-Enterprise_UI-0A3F20?style=for-the-badge&logo=Streamlit&logoColor=white)
![Analytics](https://img.shields.io/badge/Supply_Chain_Analytics-2E8B57?style=for-the-badge)

## 📌 Executive Summary
Efficient supply chain management in the fertilizer industry strongly relies on predicting seasonal demand. Running out of stock precisely when the sowing window opens can severely damage crop yields and compromise institutional reputation. 

This project establishes an **Enterprise Time-Series Forecasting framework**, specifically conceptualized for the Fauji Fertilizer Company (FFC) operational ecosystem. It deploys an advanced Extreme Gradient Boosting (XGBoost) architecture to predict 30-day forward, store-level sales volumes. This transitions warehousing strategy from reactive replenishment to predictive, data-driven stock allocation.

---

## 🚀 Core Architecture & Capabilities

### 1. Domain-Specific Data Synthesis (`src/data_generator.py`)
To ensure complete confidentiality of proprietary corporate data while maintaining architectural validity, this system employs a robust mathematical data generator. 
- Simulates realistic 2023–2026 distribution metrics across multiple localized centers.
- Utilizes Gaussian curves to mathematically replicate the **Rabi Season** (Wheat sowing: Oct-Dec) and the **Kharif Season** (Rice/Cotton sowing: May-Jul) demand volatility inherent to the Pakistani agricultural market.

### 2. Time-Series Feature Engineering (`src/processing.py`)
Translates raw temporal sequences into a stationary feature matrix suitable for machine learning optimization:
- Extrapolates temporal aggregates (Day, Month, Quarter) and Domain Flags (`is_peak_season`).
- Calculates dynamic rolling statistical windows (7-day and 30-day momentum).
- Engineers historical lag variables (T-1, T-7, T-14, T-30) to capture sales momentum vectors.

### 3. Machine Learning Engine (`src/model.py`)
- **Algorithm**: Extreme Gradient Boosting (XGBoost) chosen for its industry-leading performance on structured tabular data.
- **Validation Protocol**: Strict chronological validation splitting (Training sequence: 2023–2025 | Evaluation sequence: 2026 out-of-sample data) explicitly implemented to eradicate data leakage. 
- **Performance Diagnostics**: Evaluated precisely via Root Mean Squared Error (RMSE) and Mean Absolute Percentage Error (MAPE) to quantify standard deviation of predictive errors.

### 4. Executive Analytics Interface (`src/app.py`)
The predictive engine is surfaced via a high-performance **Streamlit & Plotly** web application. 
- Designed with strict corporate UI/UX principles reflecting the FFC institutional design language (Deep Green/Slate styling).
- Features dynamic, single-click recursive 30-day algorithmic forecasting capability mapped against historical inventory trajectories.
- Incorporates a transparent "Algorithmic Attribution" (Feature Importance) module, allowing executives to visually verify the underlying drivers of the AI projections.

---

## 🛠️ Deployment Instructions

### 1. Environment Initialization
Clone the repository to your local operating system and configure the Python virtual environment:

```powershell
# Create an isolated environment
python -m venv ffc_sales_project

# Activate environment (Windows)
.\ffc_sales_project\Scripts\Activate.ps1

# Install architectural dependencies
pip install -r requirements.txt
```

### 2. Operational Execution Sequence
Initialize the data pipeline, train the ML architecture, and launch the user interface sequentially:

| Executable Step | Terminal Command | System Action |
| :--- | :--- | :--- |
| **Data Generation** | `python src/data_generator.py` | Synthesizes historical dataset up to current 2026 timeline. |
| **Model Training** | `python src/model.py` | Compiles training sequence, exports binary models & metrics to `/models`. |
| **Dashboard Launch** | `streamlit run src/app.py` | Boots local host for the interactive corporate dashboard. |

---

## ⚠️ Disclaimer
This codebase is an independent technical demonstration of predictive supply chain modeling capabilities. It successfully utilizes advanced synthetic data engines to mimic corporate operational logic while ensuring the complete confidentiality and security of any actual proprietary institutional data.

---
**Technical Development Lead: [Your Name Here]**  
*Predictive Analytics Solutions for Agricultural Supply Chains* | *[Link to your LinkedIn Profile]*
