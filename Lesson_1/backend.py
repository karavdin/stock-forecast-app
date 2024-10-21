# backend.py
from fastapi import FastAPI
from pydantic import BaseModel
import yfinance as yf
from prophet import Prophet
import pandas as pd

app = FastAPI()

# Define the request model
class ForecastRequest(BaseModel):
    ticker: str
    periods: int  # Number of days to forecast

@app.get("/")
def root():
    return {"message": "Stock Forecast API"}

@app.post("/forecast")
def forecast_sales(request: ForecastRequest):
    # Fetch stock data from Yahoo Finance
    data = yf.download(request.ticker, period="1y", interval="1d")
    
    if data.empty:
        return {"error": "No data found for the ticker"}

    # Prepare data for Prophet
    df = data.reset_index()[['Date', 'Close']]
    df.columns = ['ds', 'y']  # Prophet expects columns named 'ds' and 'y'

    # Remove timezone information from the 'ds' (Date) column
    df['ds'] = df['ds'].dt.tz_localize(None)
    
    # Train Prophet model
    model = Prophet(daily_seasonality=True)
    model.fit(df)
    
    # Make future predictions
    future = model.make_future_dataframe(periods=request.periods)
    forecast = model.predict(future)

    # Return forecast and historical data
    forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    return {
        "historical": df.to_dict(orient='records'),
        "forecast": forecast.to_dict(orient='records')
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=True)