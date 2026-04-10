import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.graph_objs as go
import pandas as pd
import logging

# Disable Prophet logs
logging.getLogger("cmdstanpy").disabled = True

START = "2020-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title("Stock Prediction Application")

# -----------------------------
# Load NIFTY 50 Stocks
# -----------------------------
@st.cache_data
def load_nifty50_tickers():
    url = "https://archives.nseindia.com/content/indices/ind_nifty50list.csv"
    df = pd.read_csv(url)
    df["Ticker"] = df["Symbol"] + ".NS"
    return df[["Company Name", "Ticker"]]

nifty_data = load_nifty50_tickers()

selected_company = st.selectbox(
    "Select Company",
    nifty_data["Company Name"]
)

ticker = nifty_data[
    nifty_data["Company Name"] == selected_company
]["Ticker"].values[0]

selected_stock = ticker

# -----------------------------
# Prediction Slider
# -----------------------------
n_years = st.slider("Years of Prediction:", 1, 5)
period = n_years * 365

# -----------------------------
# Load Stock Data
# -----------------------------
@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

with st.spinner("Fetching stock data..."):
    data = load_data(selected_stock)

if data.empty:
    st.error("No data found!")
    st.stop()

# Fix MultiIndex
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

# Clean Data
data["Date"] = pd.to_datetime(data["Date"])
data["Open"] = pd.to_numeric(data["Open"], errors="coerce")
data["Close"] = pd.to_numeric(data["Close"], errors="coerce")

data = data.dropna()

# -----------------------------
# Add Moving Average
# -----------------------------
data["MA50"] = data["Close"].rolling(50).mean()
data = data.dropna(subset=["MA50"])

# -----------------------------
# Raw Data
# -----------------------------
st.subheader("Raw Data")
st.write(data.tail())

# -----------------------------
# Plot Raw Data (Line Graph)
# -----------------------------
def plot_raw_data():
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=data["Date"],
        y=data["Open"],
        mode='lines',
        name="Open Price"
    ))

    fig.add_trace(go.Scatter(
        x=data["Date"],
        y=data["Close"],
        mode='lines',
        name="Close Price"
    ))

    fig.update_layout(
        title="Stock Price Over Time",
        xaxis_rangeslider_visible=True
    )

    st.plotly_chart(fig)

plot_raw_data()

# -----------------------------
# Moving Average + Trend
# -----------------------------
st.subheader("Trend Analysis")

fig_ma = go.Figure()

fig_ma.add_trace(go.Scatter(
    x=data["Date"],
    y=data["Close"],
    mode='lines',
    name="Close Price"
))

fig_ma.add_trace(go.Scatter(
    x=data["Date"],
    y=data["MA50"],
    mode='lines',
    name="MA50"
))

st.plotly_chart(fig_ma)

# -----------------------------
# Buy/Sell Signal
# -----------------------------
st.subheader("Trading Signal")

latest_close = data["Close"].iloc[-1]
latest_ma50 = data["MA50"].iloc[-1]

if latest_close > latest_ma50:
    st.success("Buy Signal")
else:
    st.error("Sell Signal")

# -----------------------------
# Daily % Change
# -----------------------------
change = ((latest_close - data["Close"].iloc[-2]) / data["Close"].iloc[-2]) * 100
st.metric("Daily Change (%)", f"{change:.2f}%")

# -----------------------------
# Prophet Forecast
# -----------------------------
@st.cache_resource
def train_model(df):
    model = Prophet()
    model.fit(df)
    return model

df_train = data[["Date", "Close"]]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

df_train["ds"] = pd.to_datetime(df_train["ds"])
df_train["y"] = pd.to_numeric(df_train["y"], errors="coerce")
df_train = df_train.dropna()

m = train_model(df_train)

future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# -----------------------------
# Forecast Output
# -----------------------------
st.subheader(" Forecast Data")
st.write(forecast.tail())

fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

# -----------------------------
# Forecast Components
# -----------------------------
st.subheader(" Forecast Components")
fig2 = m.plot_components(forecast)
st.write(fig2)