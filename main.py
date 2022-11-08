import streamlit as st
from datetime import date
import yfinance as yf
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go

START = "2020-01-01"#start of stock data
TODAY = date.today().strftime("%Y-%m-%d")

st.title("Stock App") # centered title 

stocks = ("AAPL", "GOOG","MSFT", "GME")
selected_stocks = st.selectbox("Select stock for prediction", stocks)# drop down box select for tuple above

#search_stocks = st.text_input("Search stocks for prediction", stocks)

n_years = st.slider("select number of years prediction", 1, 4) #number of years slider!
period =  n_years * 365 #number of days of prediction period

@st.cache
def load_data(ticker):
    data= yf.download(ticker, START, TODAY)
    data.reset_index(inplace = True)
    return data

data_load_state = st.text("Load data ...")
data = load_data(selected_stocks)
data_load_state.text("Loading data ... Done!")
def view():
    st.subheader('Raw data')
    st.write(data.tail())

st.checkbox("View Raw", value=True, on_change=view())




def plot_raw_data():
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name = 'Open'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name = 'Close'))
    fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible = True)
    st.plotly_chart(fig)

plot_raw_data()



#----------------------------------------------------------------
#forecasting
# Predict forecast with Prophet.
df_train = data[['Date','Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})


#================================================================#
# m = NeuralProphet()
# metrics = m.fit(df_train, freq = "D")
# future = m.make_future_dataframe(df_train, periods=period)
# forecast = m.predict(future)

# # Show and plot forecast
# st.subheader('Forecast data')
# #st.write(forecast.tail())

# # create forecast
# df_future = m.make_future_dataframe(df_train, periods=365)
# forecast = m.predict(df_future)
# # create plots

# fig_forecast = m.plot(forecast)
# fig_components = m.plot_components(forecast)
# #fig_model = m.plot_parameters()

# st.write(f'Forecast plot for {n_years} years')
# #fig1 = plot_plotly(fig_forecast)
# st.plotly_chart(fig_forecast)

# st.write("Forecast components")
# fig2 = m.plot_components(fig_components)
# st.write(fig2)

#================================================================#

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# Show and plot forecast
st.subheader('Forecast data')
st.write(forecast.tail())
    
st.write(f'Forecast plot for {n_years} years')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write("Forecast components")
fig2 = m.plot_components(forecast)
st.write(fig2)
