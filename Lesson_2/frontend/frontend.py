# frontend.py
import streamlit as st
import requests
import pandas as pd
import altair as alt

# Streamlit UI Layout
st.title("Stock Price Forecast Application")

# Input section
ticker = st.text_input("Enter Stock Ticker Symbol", "AAPL")
forecast_days = st.slider("Select Days for Forecast", min_value=1, max_value=365, value=30)

# Fetch data and forecast
if st.button("Get Forecast"):
    # API Call to FastAPI Backend
    response = requests.post("http://fastapi:8000/forecast", json={"ticker": ticker, "periods": forecast_days})

    if response.status_code == 200:
        data = response.json()

        # Historical data
        historical = pd.DataFrame(data["historical"])
        forecast = pd.DataFrame(data["forecast"])

        # Combine historical and forecast data
        historical['type'] = 'Historical'
        forecast['type'] = 'Forecast'
        combined_data = pd.concat([historical[['ds', 'y', 'type']], forecast[['ds', 'yhat', 'type']].rename(columns={'yhat': 'y'})])

        # Create the Altair chart with larger font sizes
        st.write(f"## Stock Prices and Forecast for {ticker}")
        line_chart = alt.Chart(combined_data).mark_line().encode(
            x=alt.X('ds:T', title='Date', axis=alt.Axis(format='%Y-%m-%d', labelFontSize=18, titleFontSize=20)),
            y=alt.Y('y:Q', title='Stock Price (USD)', axis=alt.Axis(labelFontSize=18, titleFontSize=20)),
            color=alt.Color('type:N', legend=alt.Legend(title="Data Type", titleFontSize=20, labelFontSize=18),
                            scale=alt.Scale(domain=['Historical', 'Forecast'], range=['blue', 'orange'])),
            tooltip=['ds:T', 'y:Q', 'type:N']
        ).properties(
            width=700,
            height=400,
            title=alt.TitleParams(f"Stock Prices and Forecast for {ticker}", fontSize=20)
        ).configure_legend(
            titleFontSize=16,
            labelFontSize=14
        ).interactive()  # Add zoom and pan functionality

        st.altair_chart(line_chart, use_container_width=True)

    else:
        st.error("Error fetching data. Please check the ticker symbol.")