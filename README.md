# 📈 Stock Prediction Application

A web-based stock forecasting application built using **Streamlit**, **Prophet**, and **Yahoo Finance API**. This app allows users to visualize historical stock data and predict future trends using machine learning.

---

## 🚀 Features

* 📊 Interactive stock data visualization (Open vs Close prices)
* 🔮 Future stock price prediction using Prophet
* 📅 Customizable prediction period (1–4 years)
* 📉 Trend and seasonality analysis
* 🌐 Dynamic stock selection (supports global and Indian stocks)
* ⚡ Fast and responsive UI with Streamlit

---

## 🛠️ Tech Stack

* **Frontend/UI:** Streamlit
* **Data Source:** yfinance (Yahoo Finance API)
* **Data Processing:** pandas
* **Visualization:** Plotly
* **Forecasting Model:** Prophet (by Meta)

---

## 📦 Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/stock-prediction-app.git
cd stock-prediction-app
```

### 2. Create virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate   # For Linux/Mac
.venv\Scripts\activate      # For Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

```bash
streamlit run main.py
```

---

## 📊 How It Works

1. User selects a stock (or enters a custom ticker)
2. Historical data is fetched using Yahoo Finance
3. Data is cleaned and formatted
4. Prophet model is trained on closing prices
5. Future dates are generated based on user input
6. Forecast is displayed with interactive charts