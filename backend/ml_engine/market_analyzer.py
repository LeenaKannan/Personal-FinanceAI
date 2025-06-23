# Stock market analysis
# backend/ml_engine/market_analyzer.py

import yfinance as yf
from typing import List, Dict

class MarketAnalyzer:
    """
    Provides market analysis for stocks and mutual funds.
    """

    def __init__(self):
        pass

    def get_stock_summary(self, symbol: str) -> Dict:
        """
        Fetches latest stock info using yfinance.
        """
        ticker = yf.Ticker(symbol)
        info = ticker.info
        return {
            "symbol": symbol,
            "current_price": info.get("currentPrice"),
            "previous_close": info.get("previousClose"),
            "day_high": info.get("dayHigh"),
            "day_low": info.get("dayLow"),
            "market_cap": info.get("marketCap"),
            "sector": info.get("sector"),
            "long_name": info.get("longName"),
        }

    def get_historical_prices(self, symbol: str, period: str = "1mo") -> List[Dict]:
        """
        Returns historical prices for a given period.
        """
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=period)
        return [
            {"date": str(idx.date()), "close": row["Close"]}
            for idx, row in hist.iterrows()
        ]
