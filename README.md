
# 🏭 ML-Powered Pre-Purchase Decision Matrix

A machine learning-powered decision support system that optimizes warehouse capacities and storage costs against global commodity price fluctuations.

The system fetches Producer Price Indices (PPI) via the **FRED (Federal Reserve Economic Data) API**, predicts the next quarter's prices using 5 different forecasting models, and recommends the most financially profitable purchasing action to the user.

## 🚀 Features

- **Multi-Model Optimization:** Automatically tests Holt-Winters, ARIMA, Random Forest, XGBoost, and Prophet algorithms (Walk-forward backtest) and selects the one with the lowest MAPE (Mean Absolute Percentage Error).
- **Real-Time Data:** Fetches US Producer Price Indices directly via FRED API integration.
- **Warehouse Constraint Simulation:** Calculates the **risk-free additional purchase** amount per raw material based on the total entered square meter (m²) capacity.
- **Financial Analysis:** Provides the estimated financial impact ($) with clear directives such as "Increase Purchase" or "Standard Plan".
- **Dynamic Reporting:** Exports a detailed Markdown report of all tests performed and parameters used.

## 🛠️ Installation

1. Clone the repository to your local machine:
   ```bash
   git clone [https://github.com/mesudegrn/ML-powered-Pre-Purchase-Decision-Matrix.git](https://github.com/mesudegrn/ML-powered-Pre-Purchase-Decision-Matrix.git)
   cd ML-powered-Pre-Purchase-Decision-Matrix

2. Install the required Python libraries (Using a virtual environment is recommended):
   
   pip install -r requirements.txt

4. Run the application:
   
   streamlit run app.py

## 🔑 Usage
To use the application, you need to get a free FRED API Key. Once the application is running, you can start fetching data by entering your API key in the "System Settings" section on the left sidebar.


https://github.com/user-attachments/assets/68f88cd9-28f6-4278-ad87-95e4654dd4e0


