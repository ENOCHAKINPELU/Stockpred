import streamlit as st
from prophet import Prophet
from datetime import date
import yfinance as yf
from prophet.plot import plot_plotly
from plotly import graph_objs as go


START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")


st.title("Stock Prediction App")

stocks = ('AAPL', 'GOOG', 'MSFT', 'AMZN')

selected_stocks = st.selectbox("Select stock", stocks)

num_of_years = st.slider("Years of Prediction: ", 1,4)
period = num_of_years * 365

@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text("load data...")
data = load_data(selected_stocks)
data_load_state.text("Loading data...done!")


st.subheader('Raw Data')
st.write(data.tail())

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='stock_open'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='stock_close'))
    fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={'Date':'ds','Close':'y'})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)



st.subheader('Forecast data')
st.write(forecast.tail())

st.write("Forecast Data")
fig1 = plot_plotly(m,forecast)

new_color = 'red'
for trace in fig1['data']:
    trace['line']['color'] = new_color

fig1.update_layout(plot_bgcolor='white')

st.plotly_chart(fig1)
