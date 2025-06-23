# Forecasting models
# backend/ml_engine/time_series.py

from typing import List, Dict, Optional
import pandas as pd

try:
    from prophet import Prophet
    _has_prophet = True
except ImportError:
    _has_prophet = False

class TimeSeriesForecaster:
    """
    Forecasts future expenses, income, or market trends using time series models.
    """

    def __init__(self):
        if _has_prophet:
            self.model = Prophet()
        else:
            self.model = None

    def fit_predict(self, history: List[Dict], periods: int = 6) -> Optional[List[Dict]]:
        """
        history: List of {"date": "YYYY-MM-DD", "value": float}
        periods: months to forecast
        """
        if not _has_prophet:
            raise ImportError("Prophet is not installed. Please install it for time series forecasting.")

        df = pd.DataFrame(history)
        df.rename(columns={"date": "ds", "value": "y"}, inplace=True)
        self.model.fit(df)
        future = self.model.make_future_dataframe(periods=periods, freq='M')
        forecast = self.model.predict(future)
        results = [
            {"date": row["ds"].strftime("%Y-%m-%d"), "forecast": float(row["yhat"])}
            for _, row in forecast.tail(periods).iterrows()
        ]
        return results
