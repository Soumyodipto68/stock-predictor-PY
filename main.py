import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly  
#from plotly import graph_objs as go
import plotly.graph_objs as go
import pandas as pd

START = "2020-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title("Stock Prediction Application")

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

ticker = nifty_data[nifty_data["Company Name"] == selected_company]["Ticker"].values[0]
selected_stock = st.selectbox("Select dataset for prediction", [ticker])

n_years = st.slider("Years of Prediction:", 1, 4)
period = n_years * 365

@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text("Loading data...")
data = load_data(selected_stock)
data_load_state.text("Loading data... done!")

st.subheader("Raw Data")
st.write(data.tail())
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

data["Date"] = pd.to_datetime(data["Date"])
data["Open"] = pd.to_numeric(data["Open"], errors="coerce")
data["Close"] = pd.to_numeric(data["Close"], errors="coerce")

data = data.dropna()

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data["Date"], y=data["Open"],mode='lines', name="stock_open",line=dict(color='green')))
    fig.add_trace(go.Scatter(x=data["Date"], y=data["Close"], mode='lines', name="stock_close",line=dict(color='red')))
    fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()  

if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

df_train = data[["Date", "Close"]]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

df_train["ds"] = pd.to_datetime(df_train["ds"])
df_train["y"] = pd.to_numeric(df_train["y"], errors="coerce")
df_train = df_train.dropna()

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

st.subheader("Forecast Data")
st.write(forecast.tail())

st.write("Forecast Data")
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write("Forecast Componets")
fig2 = m.plot_components(forecast)
st.write(fig2)
