# backend/ml_engine/market_analyzer.py

import yfinance as yf
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum
import logging
import requests
from dataclasses import dataclass
import statistics
import warnings

# Suppress yfinance warnings
warnings.filterwarnings("ignore")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MarketSegment(str, Enum):
    """Market segments for analysis."""
    LARGE_CAP = "large_cap"
    MID_CAP = "mid_cap"
    SMALL_CAP = "small_cap"
    MUTUAL_FUND = "mutual_fund"
    ETF = "etf"
    COMMODITY = "commodity"
    CURRENCY = "currency"

class RiskLevel(str, Enum):
    """Risk levels for investments."""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"

@dataclass
class StockMetrics:
    """Data class for stock metrics."""
    symbol: str
    current_price: float
    price_change: float
    price_change_percent: float
    volume: int
    market_cap: Optional[float]
    pe_ratio: Optional[float]
    pb_ratio: Optional[float]
    dividend_yield: Optional[float]
    beta: Optional[float]
    fifty_two_week_high: Optional[float]
    fifty_two_week_low: Optional[float]
    risk_level: RiskLevel
    recommendation: str

@dataclass
class MarketTrend:
    """Data class for market trend analysis."""
    trend_direction: str  # "bullish", "bearish", "sideways"
    strength: float  # 0-1 scale
    support_level: Optional[float]
    resistance_level: Optional[float]
    volatility: float
    momentum_score: float

class MarketAnalyzer:
    """
    Advanced market analyzer for Indian and global markets.
    Provides comprehensive analysis for stocks, mutual funds, and market trends.
    """

    def __init__(self):
        """Initialize the market analyzer with Indian market data."""
        self.indian_indices = {
            "^NSEI": "NIFTY 50",
            "^BSESN": "SENSEX",
            "^NSEBANK": "NIFTY BANK",
            "^NSEIT": "NIFTY IT",
            "^NSEAUTO": "NIFTY AUTO",
            "^NSEPHARMA": "NIFTY PHARMA"
        }
        
        self.popular_indian_stocks = {
            "RELIANCE.NS": "Reliance Industries",
            "TCS.NS": "Tata Consultancy Services",
            "HDFCBANK.NS": "HDFC Bank",
            "INFY.NS": "Infosys",
            "ICICIBANK.NS": "ICICI Bank",
            "HINDUNILVR.NS": "Hindustan Unilever",
            "ITC.NS": "ITC Limited",
            "SBIN.NS": "State Bank of India",
            "BHARTIARTL.NS": "Bharti Airtel",
            "KOTAKBANK.NS": "Kotak Mahindra Bank"
        }
        
        self.sector_mapping = {
            "Technology": ["TCS.NS", "INFY.NS", "WIPRO.NS", "HCLTECH.NS"],
            "Banking": ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "KOTAKBANK.NS"],
            "Energy": ["RELIANCE.NS", "ONGC.NS", "BPCL.NS", "IOC.NS"],
            "FMCG": ["HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS", "BRITANNIA.NS"],
            "Pharma": ["SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS"]
        }

    def get_stock_summary(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch comprehensive stock information with error handling.
        
        Args:
            symbol: Stock symbol (e.g., 'RELIANCE.NS' for NSE, 'AAPL' for NASDAQ)
            
        Returns:
            Dict containing detailed stock information
        """
        try:
            # Ensure Indian stocks have proper suffix
            if not any(suffix in symbol for suffix in ['.NS', '.BO', '.']) and symbol.isupper():
                symbol = f"{symbol}.NS"  # Default to NSE
            
            ticker = yf.Ticker(symbol)
            info = ticker.info
            hist = ticker.history(period="1d")
            
            if hist.empty:
                raise ValueError(f"No data available for symbol {symbol}")
            
            current_price = hist['Close'].iloc[-1] if not hist.empty else info.get("currentPrice", 0)
            previous_close = info.get("previousClose", current_price)
            
            price_change = current_price - previous_close
            price_change_percent = (price_change / previous_close * 100) if previous_close else 0
            
            # Calculate risk level
            beta = info.get("beta", 1.0)
            risk_level = self._calculate_risk_level(beta, info.get("sector", ""))
            
            # Generate recommendation
            recommendation = self._generate_recommendation(info, price_change_percent)
            
            return {
                "symbol": symbol,
                "company_name": info.get("longName", info.get("shortName", symbol)),
                "current_price": round(current_price, 2),
                "previous_close": round(previous_close, 2),
                "price_change": round(price_change, 2),
                "price_change_percent": round(price_change_percent, 2),
                "day_high": info.get("dayHigh"),
                "day_low": info.get("dayLow"),
                "volume": info.get("volume", 0),
                "avg_volume": info.get("averageVolume", 0),
                "market_cap": info.get("marketCap"),
                "market_cap_formatted": self._format_market_cap(info.get("marketCap")),
                "pe_ratio": info.get("trailingPE"),
                "pb_ratio": info.get("priceToBook"),
                "dividend_yield": info.get("dividendYield"),
                "beta": beta,
                "fifty_two_week_high": info.get("fiftyTwoWeekHigh"),
                "fifty_two_week_low": info.get("fiftyTwoWeekLow"),
                "sector": info.get("sector"),
                "industry": info.get("industry"),
                "country": info.get("country"),
                "currency": info.get("currency", "INR" if ".NS" in symbol else "USD"),
                "risk_level": risk_level.value,
                "recommendation": recommendation,
                "last_updated": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get stock summary for {symbol}: {e}")
            return {
                "symbol": symbol,
                "error": str(e),
                "last_updated": datetime.now().isoformat()
            }

    def get_historical_prices(self, symbol: str, period: str = "1mo") -> List[Dict[str, Any]]:
        """
        Get historical price data with technical indicators.
        
        Args:
            symbol: Stock symbol
            period: Time period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            
        Returns:
            List of historical price data with technical indicators
        """
        try:
            # Ensure Indian stocks have proper suffix
            if not any(suffix in symbol for suffix in ['.NS', '.BO', '.']) and symbol.isupper():
                symbol = f"{symbol}.NS"
            
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period)
            
            if hist.empty:
                return []
            
            # Calculate technical indicators
            hist = self._add_technical_indicators(hist)
            
            result = []
            for idx, row in hist.iterrows():
                result.append({
                    "date": idx.strftime("%Y-%m-%d"),
                    "open": round(row["Open"], 2),
                    "high": round(row["High"], 2),
                    "low": round(row["Low"], 2),
                    "close": round(row["Close"], 2),
                    "volume": int(row["Volume"]),
                    "sma_20": round(row.get("SMA_20", 0), 2),
                    "sma_50": round(row.get("SMA_50", 0), 2),
                    "rsi": round(row.get("RSI", 0), 2),
                    "macd": round(row.get("MACD", 0), 2),
                    "bollinger_upper": round(row.get("BB_Upper", 0), 2),
                    "bollinger_lower": round(row.get("BB_Lower", 0), 2)
                })
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to get historical prices for {symbol}: {e}")
            return []

    def analyze_market_trend(self, symbol: str, period: str = "3mo") -> MarketTrend:
        """
        Analyze market trend using technical analysis.
        
        Args:
            symbol: Stock symbol
            period: Analysis period
            
        Returns:
            MarketTrend object with trend analysis
        """
        try:
            hist_data = self.get_historical_prices(symbol, period)
            
            if len(hist_data) < 20:
                return MarketTrend("unknown", 0.0, None, None, 0.0, 0.0)
            
            prices = [item["close"] for item in hist_data]
            volumes = [item["volume"] for item in hist_data]
            
            # Calculate trend direction
            recent_prices = prices[-10:]
            older_prices = prices[-20:-10]
            
            recent_avg = statistics.mean(recent_prices)
            older_avg = statistics.mean(older_prices)
            
            if recent_avg > older_avg * 1.02:
                trend_direction = "bullish"
                strength = min((recent_avg / older_avg - 1) * 10, 1.0)
            elif recent_avg < older_avg * 0.98:
                trend_direction = "bearish"
                strength = min((1 - recent_avg / older_avg) * 10, 1.0)
            else:
                trend_direction = "sideways"
                strength = 0.5
            
            # Calculate support and resistance
            support_level = min(prices[-20:])
            resistance_level = max(prices[-20:])
            
            # Calculate volatility
            returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
            volatility = statistics.stdev(returns) if len(returns) > 1 else 0.0
            
            # Calculate momentum score
            momentum_score = (prices[-1] - prices[-10]) / prices[-10] if len(prices) >= 10 else 0.0
            
            return MarketTrend(
                trend_direction=trend_direction,
                strength=strength,
                support_level=support_level,
                resistance_level=resistance_level,
                volatility=volatility,
                momentum_score=momentum_score
            )
            
        except Exception as e:
            logger.error(f"Failed to analyze market trend for {symbol}: {e}")
            return MarketTrend("unknown", 0.0, None, None, 0.0, 0.0)

    def get_sector_performance(self, sector: str = None) -> Dict[str, Any]:
        """
        Get sector-wise performance analysis.
        
        Args:
            sector: Specific sector to analyze (optional)
            
        Returns:
            Dict containing sector performance data
        """
        try:
            sector_performance = {}
            
            sectors_to_analyze = [sector] if sector else self.sector_mapping.keys()
            
            for sector_name in sectors_to_analyze:
                if sector_name not in self.sector_mapping:
                    continue
                
                sector_stocks = self.sector_mapping[sector_name]
                sector_data = []
                
                for stock_symbol in sector_stocks:
                    try:
                        stock_info = self.get_stock_summary(stock_symbol)
                        if "error" not in stock_info:
                            sector_data.append({
                                "symbol": stock_symbol,
                                "company": stock_info.get("company_name", ""),
                                "price_change_percent": stock_info.get("price_change_percent", 0),
                                "market_cap": stock_info.get("market_cap", 0)
                            })
                    except Exception as e:
                        logger.warning(f"Failed to get data for {stock_symbol}: {e}")
                        continue
                
                if sector_data:
                    # Calculate sector metrics
                    avg_change = statistics.mean([s["price_change_percent"] for s in sector_data])
                    best_performer = max(sector_data, key=lambda x: x["price_change_percent"])
                    worst_performer = min(sector_data, key=lambda x: x["price_change_percent"])
                    
                    sector_performance[sector_name] = {
                        "average_change": round(avg_change, 2),
                        "best_performer": best_performer,
                        "worst_performer": worst_performer,
                        "total_stocks": len(sector_data),
                        "gainers": len([s for s in sector_data if s["price_change_percent"] > 0]),
                        "losers": len([s for s in sector_data if s["price_change_percent"] < 0])
                    }
            
            return sector_performance
            
        except Exception as e:
            logger.error(f"Failed to get sector performance: {e}")
            return {}

    def get_market_indices(self) -> Dict[str, Any]:
        """
        Get current status of major Indian market indices.
        
        Returns:
            Dict containing indices data
        """
        try:
            indices_data = {}
            
            for symbol, name in self.indian_indices.items():
                try:
                    index_info = self.get_stock_summary(symbol)
                    if "error" not in index_info:
                        indices_data[name] = {
                            "symbol": symbol,
                            "current_level": index_info.get("current_price", 0),
                            "change": index_info.get("price_change", 0),
                            "change_percent": index_info.get("price_change_percent", 0),
                            "day_high": index_info.get("day_high", 0),
                            "day_low": index_info.get("day_low", 0)
                        }
                except Exception as e:
                    logger.warning(f"Failed to get data for index {symbol}: {e}")
                    continue
            
            return indices_data
            
        except Exception as e:
            logger.error(f"Failed to get market indices: {e}")
            return {}

    def get_investment_recommendations(self, risk_profile: str, investment_amount: float, time_horizon: str) -> List[Dict[str, Any]]:
        """
        Get personalized investment recommendations based on user profile.
        
        Args:
            risk_profile: User's risk tolerance ('conservative', 'moderate', 'aggressive')
            investment_amount: Amount to invest
            time_horizon: Investment time horizon ('short', 'medium', 'long')
            
        Returns:
            List of investment recommendations
        """
        try:
            recommendations = []
            
            # Define allocation based on risk profile
            allocations = {
                'conservative': {'equity': 0.3, 'debt': 0.6, 'gold': 0.1},
                'moderate': {'equity': 0.6, 'debt': 0.3, 'gold': 0.1},
                'aggressive': {'equity': 0.8, 'debt': 0.15, 'gold': 0.05}
            }
            
            allocation = allocations.get(risk_profile, allocations['moderate'])
            
            # Equity recommendations
            if allocation['equity'] > 0:
                equity_amount = investment_amount * allocation['equity']
                equity_stocks = self._get_top_performing_stocks(5)
                
                recommendations.append({
                    "category": "Equity",
                    "allocation_percent": allocation['equity'] * 100,
                    "recommended_amount": equity_amount,
                    "instruments": equity_stocks,
                    "rationale": "For long-term wealth creation and inflation beating returns"
                })
            
            # Debt recommendations
            if allocation['debt'] > 0:
                debt_amount = investment_amount * allocation['debt']
                
                recommendations.append({
                    "category": "Debt",
                    "allocation_percent": allocation['debt'] * 100,
                    "recommended_amount": debt_amount,
                    "instruments": [
                        {"name": "Government Bonds", "allocation": 0.4},
                        {"name": "Corporate Bonds", "allocation": 0.3},
                        {"name": "Fixed Deposits", "allocation": 0.3}
                    ],
                    "rationale": "For stable returns and capital preservation"
                })
            
            # Gold recommendations
            if allocation['gold'] > 0:
                gold_amount = investment_amount * allocation['gold']
                
                recommendations.append({
                    "category": "Gold",
                    "allocation_percent": allocation['gold'] * 100,
                    "recommended_amount": gold_amount,
                    "instruments": [
                        {"name": "Gold ETF", "allocation": 0.6},
                        {"name": "Digital Gold", "allocation": 0.4}
                    ],
                    "rationale": "For portfolio diversification and hedge against inflation"
                })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Failed to get investment recommendations: {e}")
            return []

    def compare_stocks(self, symbols: List[str]) -> Dict[str, Any]:
        """
        Compare multiple stocks across various metrics.
        
        Args:
            symbols: List of stock symbols to compare
            
        Returns:
            Dict containing comparison data
        """
        try:
            comparison_data = {}
            
            for symbol in symbols:
                stock_info = self.get_stock_summary(symbol)
                if "error" not in stock_info:
                    comparison_data[symbol] = stock_info
            
            if not comparison_data:
                return {}
            
            # Calculate relative metrics
            metrics = ['pe_ratio', 'pb_ratio', 'dividend_yield', 'beta', 'price_change_percent']
            
            comparison_result = {
                "stocks": comparison_data,
                "best_in_category": {},
                "comparison_summary": {}
            }
            
            for metric in metrics:
                values = {symbol: data.get(metric, 0) for symbol, data in comparison_data.items() if data.get(metric) is not None}
                
                if values:
                    if metric in ['dividend_yield', 'price_change_percent']:
                        best_stock = max(values.items(), key=lambda x: x[1])
                    else:  # Lower is better for PE, PB, Beta
                        best_stock = min(values.items(), key=lambda x: x[1])
                    
                    comparison_result["best_in_category"][metric] = {
                        "symbol": best_stock[0],
                        "value": best_stock[1]
                    }
            
            return comparison_result
            
        except Exception as e:
            logger.error(f"Failed to compare stocks: {e}")
            return {}

    def get_mutual_fund_analysis(self, fund_symbol: str) -> Dict[str, Any]:
        """
        Analyze mutual fund performance and characteristics.
        
        Args:
            fund_symbol: Mutual fund symbol
            
        Returns:
            Dict containing mutual fund analysis
        """
        try:
            # Note: This is a simplified implementation
            # In practice, you'd use specialized APIs for mutual fund data
            
            fund_info = self.get_stock_summary(fund_symbol)
            historical_data = self.get_historical_prices(fund_symbol, "1y")
            
            if not historical_data:
                return {"error": "No data available for this fund"}
            
            # Calculate returns
            returns = self._calculate_returns(historical_data)
            
            # Calculate risk metrics
            risk_metrics = self._calculate_risk_metrics(historical_data)
            
            return {
                "fund_info": fund_info,
                "returns": returns,
                "risk_metrics": risk_metrics,
                "recommendation": self._generate_fund_recommendation(returns, risk_metrics)
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze mutual fund {fund_symbol}: {e}")
            return {"error": str(e)}

    # Private helper methods

    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to price data."""
        try:
            # Simple Moving Averages
            df['SMA_20'] = df['Close'].rolling(window=20).mean()
            df['SMA_50'] = df['Close'].rolling(window=50).mean()
            
            # RSI
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # MACD
            exp1 = df['Close'].ewm(span=12).mean()
            exp2 = df['Close'].ewm(span=26).mean()
            df['MACD'] = exp1 - exp2
            
            # Bollinger Bands
            df['BB_Middle'] = df['Close'].rolling(window=20).mean()
            bb_std = df['Close'].rolling(window=20).std()
            df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
            df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to add technical indicators: {e}")
            return df

    def _calculate_risk_level(self, beta: Optional[float], sector: str) -> RiskLevel:
        """Calculate risk level based on beta and sector."""
        if beta is None:
            beta = 1.0
        
        # Sector-based risk adjustment
        high_risk_sectors = ["Technology", "Biotechnology", "Energy"]
        low_risk_sectors = ["Utilities", "Consumer Staples", "Healthcare"]
        
        base_risk = beta
        
        if sector in high_risk_sectors:
            base_risk += 0.2
        elif sector in low_risk_sectors:
            base_risk -= 0.2
        
        if base_risk < 0.8:
            return RiskLevel.LOW
        elif base_risk < 1.2:
            return RiskLevel.MODERATE
        elif base_risk < 1.5:
            return RiskLevel.HIGH
        else:
            return RiskLevel.VERY_HIGH

    def _generate_recommendation(self, info: Dict, price_change_percent: float) -> str:
        """Generate investment recommendation based on stock metrics."""
        try:
            pe_ratio = info.get("trailingPE", 0)
            pb_ratio = info.get("priceToBook", 0)
            dividend_yield = info.get("dividendYield", 0)
            
            score = 0
            
            # PE ratio scoring
            if pe_ratio and pe_ratio < 15:
                score += 2
            elif pe_ratio and pe_ratio < 25:
                score += 1
            
            # PB ratio scoring
            if pb_ratio and pb_ratio < 1.5:
                score += 2
            elif pb_ratio and pb_ratio < 3:
                score += 1
            
            # Dividend yield scoring
            if dividend_yield and dividend_yield > 0.03:
                score += 1
            
            # Price momentum scoring
            if price_change_percent > 5:
                score += 1
            elif price_change_percent < -5:
                score -= 1
            
            if score >= 4:
                return "Strong Buy"
            elif score >= 2:
                return "Buy"
            elif score >= 0:
                return "Hold"
            else:
                return "Sell"
                
        except Exception:
            return "Hold"

    def _format_market_cap(self, market_cap: Optional[float]) -> str:
        """Format market cap in Indian numbering system."""
        if not market_cap:
            return "N/A"
        
        if market_cap >= 1e12:
            return f"₹{market_cap/1e12:.2f} Lakh Cr"
        elif market_cap >= 1e10:
            return f"₹{market_cap/1e10:.2f} Thousand Cr"
        elif market_cap >= 1e7:
            return f"₹{market_cap/1e7:.2f} Cr"
        else:
            return f"₹{market_cap/1e5:.2f} Lakh"

    def _get_top_performing_stocks(self, count: int) -> List[Dict[str, Any]]:
        """Get top performing stocks from popular Indian stocks."""
        try:
            stock_performance = []
            
            for symbol, name in list(self.popular_indian_stocks.items())[:count*2]:
                try:
                    stock_info = self.get_stock_summary(symbol)
                    if "error" not in stock_info:
                        stock_performance.append({
                            "symbol": symbol,
                            "name": name,
                            "change_percent": stock_info.get("price_change_percent", 0),
                            "recommendation": stock_info.get("recommendation", "Hold")
                        })
                except Exception:
                    continue
            
            # Sort by performance and return top performers
            stock_performance.sort(key=lambda x: x["change_percent"], reverse=True)
            return stock_performance[:count]
            
        except Exception as e:
            logger.error(f"Failed to get top performing stocks: {e}")
            return []

    def _calculate_returns(self, historical_data: List[Dict]) -> Dict[str, float]:
        """Calculate various return metrics."""
        try:
            prices = [item["close"] for item in historical_data]
            
            if len(prices) < 2:
                return {}
            
            # Calculate returns
            total_return = (prices[-1] - prices[0]) / prices[0] * 100
            
            # Annualized return (assuming daily data)
            days = len(prices)
            annualized_return = ((prices[-1] / prices[0]) ** (365 / days) - 1) * 100
            
            return {
                "total_return": round(total_return, 2),
                "annualized_return": round(annualized_return, 2)
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate returns: {e}")
            return {}

    def _calculate_risk_metrics(self, historical_data: List[Dict]) -> Dict[str, float]:
        """Calculate risk metrics."""
        try:
            prices = [item["close"] for item in historical_data]
            
            if len(prices) < 2:
                return {}
            
            # Calculate daily returns
            returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
            
            # Volatility (standard deviation of returns)
            volatility = statistics.stdev(returns) * (252 ** 0.5) * 100  # Annualized
            
            # Maximum drawdown
            peak = prices[0]
            max_drawdown = 0
            
            for price in prices:
                if price > peak:
                    peak = price
                drawdown = (peak - price) / peak
                max_drawdown = max(max_drawdown, drawdown)
            
            return {
                "volatility": round(volatility, 2),
                "max_drawdown": round(max_drawdown * 100, 2)
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate risk metrics: {e}")
            return {}

    def _generate_fund_recommendation(self, returns: Dict, risk_metrics: Dict) -> str:
        """Generate recommendation for mutual fund."""
        try:
            annualized_return = returns.get("annualized_return", 0)
            volatility = risk_metrics.get("volatility", 0)
            
            # Simple risk-adjusted return
            if volatility > 0:
                sharpe_ratio = annualized_return / volatility
            else:
                sharpe_ratio = 0
            
            if sharpe_ratio > 1.5:
                return "Excellent"
            elif sharpe_ratio > 1.0:
                return "Good"
            elif sharpe_ratio > 0.5:
                return "Average"
            else:
                return "Below Average"
                
        except Exception:
            return "Average"

# Example usage
if __name__ == "__main__":
    # Test the market analyzer
    analyzer = MarketAnalyzer()
    
    # Test stock summary
    print("Testing stock summary for Reliance...")
    reliance_info = analyzer.get_stock_summary("RELIANCE.NS")
    print(f"Reliance current price: ₹{reliance_info.get('current_price', 'N/A')}")
    
    # Test market indices
    print("\nTesting market indices...")
    indices = analyzer.get_market_indices()
    for name, data in indices.items():
        print(f"{name}: {data.get('current_level', 'N/A')} ({data.get('change_percent', 'N/A')}%)")
    
    # Test sector performance
    print("\nTesting sector performance...")
    sector_perf = analyzer.get_sector_performance("Technology")
    print(f"Technology sector performance: {sector_perf}")
    
    print("Market analyzer tests completed!")
