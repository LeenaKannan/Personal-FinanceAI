# backend/ml_engine/time_series.py

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
from enum import Enum
import logging
import warnings
from dataclasses import dataclass
import statistics

# Suppress warnings
warnings.filterwarnings("ignore")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optional imports with fallbacks
try:
    from prophet import Prophet
    _has_prophet = True
except ImportError:
    _has_prophet = False
    logger.warning("Prophet not available. Install with: pip install prophet")

try:
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    _has_sklearn = True
except ImportError:
    _has_sklearn = False
    logger.warning("Scikit-learn not available. Install with: pip install scikit-learn")

try:
    import statsmodels.api as sm
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    _has_statsmodels = True
except ImportError:
    _has_statsmodels = False
    logger.warning("Statsmodels not available. Install with: pip install statsmodels")

class ForecastMethod(str, Enum):
    """Available forecasting methods."""
    PROPHET = "prophet"
    LINEAR_REGRESSION = "linear_regression"
    RANDOM_FOREST = "random_forest"
    EXPONENTIAL_SMOOTHING = "exponential_smoothing"
    MOVING_AVERAGE = "moving_average"
    SEASONAL_NAIVE = "seasonal_naive"
    AUTO = "auto"

class SeasonalityType(str, Enum):
    """Types of seasonality patterns."""
    NONE = "none"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"
    FESTIVAL = "festival"  # Indian festival seasonality

@dataclass
class ForecastResult:
    """Data class for forecast results."""
    dates: List[str]
    forecasts: List[float]
    lower_bounds: List[float]
    upper_bounds: List[float]
    confidence_level: float
    method_used: str
    accuracy_metrics: Dict[str, float]
    seasonality_detected: bool
    trend_direction: str

@dataclass
class ModelPerformance:
    """Data class for model performance metrics."""
    mae: float  # Mean Absolute Error
    mse: float  # Mean Squared Error
    rmse: float  # Root Mean Squared Error
    mape: float  # Mean Absolute Percentage Error
    accuracy_score: float  # Custom accuracy score (0-1)

class TimeSeriesForecaster:
    """
    Advanced time series forecasting for Indian personal finance data.
    Supports multiple algorithms with automatic model selection and
    Indian financial calendar awareness.
    """

    def __init__(self, default_method: ForecastMethod = ForecastMethod.AUTO):
        """
        Initialize the time series forecaster.
        
        Args:
            default_method: Default forecasting method to use
        """
        self.default_method = default_method
        self.models = {}
        self.indian_financial_calendar = self._get_indian_financial_calendar()
        self.festival_dates = self._get_festival_dates()
        
        # Initialize available models
        self._initialize_models()

    def _initialize_models(self):
        """Initialize available forecasting models."""
        if _has_prophet:
            self.models[ForecastMethod.PROPHET] = self._create_prophet_model
        if _has_sklearn:
            self.models[ForecastMethod.LINEAR_REGRESSION] = self._create_linear_model
            self.models[ForecastMethod.RANDOM_FOREST] = self._create_rf_model
        if _has_statsmodels:
            self.models[ForecastMethod.EXPONENTIAL_SMOOTHING] = self._create_exp_smoothing_model
        
        # Always available methods
        self.models[ForecastMethod.MOVING_AVERAGE] = self._create_moving_average_model
        self.models[ForecastMethod.SEASONAL_NAIVE] = self._create_seasonal_naive_model

    def _get_indian_financial_calendar(self) -> Dict[str, List[int]]:
        """Get Indian financial calendar events."""
        return {
            'financial_year_end': [3],  # March
            'bonus_months': [3, 10, 11],  # March, October, November
            'festival_season': [9, 10, 11],  # September to November
            'tax_filing': [1, 2, 3],  # January to March
            'monsoon_season': [6, 7, 8, 9],  # June to September
            'wedding_season': [11, 12, 1, 2]  # November to February
        }

    def _get_festival_dates(self) -> Dict[str, List[str]]:
        """Get major Indian festival dates (simplified)."""
        return {
            'diwali': ['2024-11-01', '2025-10-20', '2026-11-08'],
            'holi': ['2024-03-25', '2025-03-14', '2026-03-03'],
            'dussehra': ['2024-10-12', '2025-10-02', '2026-10-21'],
            'eid': ['2024-04-11', '2025-03-31', '2026-03-20']
        }

    def fit_predict(
        self, 
        history: List[Dict[str, Any]], 
        periods: int = 6,
        method: Optional[ForecastMethod] = None,
        confidence_level: float = 0.95,
        include_seasonality: bool = True,
        custom_seasonality: Optional[Dict[str, Any]] = None
    ) -> Optional[ForecastResult]:
        """
        Fit model and generate forecasts.
        
        Args:
            history: Historical data [{"date": "YYYY-MM-DD", "value": float}]
            periods: Number of periods to forecast
            method: Forecasting method to use
            confidence_level: Confidence level for prediction intervals
            include_seasonality: Whether to include seasonal patterns
            custom_seasonality: Custom seasonality parameters
            
        Returns:
            ForecastResult object with predictions and metadata
        """
        try:
            if not history or len(history) < 2:
                logger.error("Insufficient historical data for forecasting")
                return None

            # Prepare data
            df = self._prepare_data(history)
            if df is None or len(df) < 2:
                return None

            # Select forecasting method
            selected_method = method or self._select_best_method(df)
            
            # Add seasonality features if requested
            if include_seasonality:
                df = self._add_seasonality_features(df, custom_seasonality)

            # Generate forecast
            forecast_result = self._generate_forecast(
                df, periods, selected_method, confidence_level
            )

            # Add Indian market context
            if forecast_result:
                forecast_result = self._add_indian_market_context(forecast_result, periods)

            return forecast_result

        except Exception as e:
            logger.error(f"Forecasting failed: {e}")
            return self._generate_fallback_forecast(history, periods)

    def _prepare_data(self, history: List[Dict[str, Any]]) -> Optional[pd.DataFrame]:
        """Prepare and validate historical data."""
        try:
            df = pd.DataFrame(history)
            
            # Validate required columns
            if 'date' not in df.columns or 'value' not in df.columns:
                logger.error("Missing required columns: 'date' and 'value'")
                return None

            # Convert date column
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df = df.dropna(subset=['date', 'value'])

            # Sort by date
            df = df.sort_values('date').reset_index(drop=True)

            # Remove duplicates (keep last)
            df = df.drop_duplicates(subset=['date'], keep='last')

            # Handle missing values
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            df = df.dropna(subset=['value'])

            # Detect and handle outliers
            df = self._handle_outliers(df)

            return df

        except Exception as e:
            logger.error(f"Data preparation failed: {e}")
            return None

    def _handle_outliers(self, df: pd.DataFrame, method: str = "iqr") -> pd.DataFrame:
        """Handle outliers in the data."""
        try:
            if method == "iqr":
                Q1 = df['value'].quantile(0.25)
                Q3 = df['value'].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Cap outliers instead of removing them
                df['value'] = df['value'].clip(lower=lower_bound, upper=upper_bound)

            return df

        except Exception as e:
            logger.error(f"Outlier handling failed: {e}")
            return df

    def _add_seasonality_features(self, df: pd.DataFrame, custom_seasonality: Optional[Dict] = None) -> pd.DataFrame:
        """Add seasonality features to the dataframe."""
        try:
            df = df.copy()
            
            # Extract date features
            df['month'] = df['date'].dt.month
            df['quarter'] = df['date'].dt.quarter
            df['day_of_year'] = df['date'].dt.dayofyear
            df['week_of_year'] = df['date'].dt.isocalendar().week

            # Indian financial year (April to March)
            df['financial_year'] = df['date'].apply(
                lambda x: x.year if x.month >= 4 else x.year - 1
            )

            # Festival season indicator
            df['is_festival_season'] = df['month'].isin([9, 10, 11]).astype(int)

            # Bonus season indicator
            df['is_bonus_season'] = df['month'].isin([3, 10, 11]).astype(int)

            # Monsoon season indicator
            df['is_monsoon'] = df['month'].isin([6, 7, 8, 9]).astype(int)

            # Custom seasonality
            if custom_seasonality:
                for feature_name, feature_months in custom_seasonality.items():
                    df[f'is_{feature_name}'] = df['month'].isin(feature_months).astype(int)

            return df

        except Exception as e:
            logger.error(f"Adding seasonality features failed: {e}")
            return df

    def _select_best_method(self, df: pd.DataFrame) -> ForecastMethod:
        """Automatically select the best forecasting method."""
        try:
            data_length = len(df)
            
            # Check for seasonality
            has_seasonality = self._detect_seasonality(df)
            
            # Check for trend
            has_trend = self._detect_trend(df)

            # Method selection logic
            if data_length < 12:
                return ForecastMethod.MOVING_AVERAGE
            elif data_length < 24:
                if _has_sklearn:
                    return ForecastMethod.LINEAR_REGRESSION
                else:
                    return ForecastMethod.MOVING_AVERAGE
            elif has_seasonality and _has_prophet:
                return ForecastMethod.PROPHET
            elif _has_statsmodels and (has_trend or has_seasonality):
                return ForecastMethod.EXPONENTIAL_SMOOTHING
            elif _has_sklearn:
                return ForecastMethod.RANDOM_FOREST
            else:
                return ForecastMethod.MOVING_AVERAGE

        except Exception as e:
            logger.error(f"Method selection failed: {e}")
            return ForecastMethod.MOVING_AVERAGE

    def _detect_seasonality(self, df: pd.DataFrame) -> bool:
        """Detect seasonality in the data."""
        try:
            if len(df) < 24:  # Need at least 2 years for reliable seasonality detection
                return False

            # Simple seasonality detection using autocorrelation
            values = df['value'].values
            
            # Check for 12-month seasonality
            if len(values) >= 12:
                seasonal_corr = np.corrcoef(values[:-12], values[12:])[0, 1]
                if seasonal_corr > 0.3:
                    return True

            return False

        except Exception as e:
            logger.error(f"Seasonality detection failed: {e}")
            return False

    def _detect_trend(self, df: pd.DataFrame) -> bool:
        """Detect trend in the data."""
        try:
            if len(df) < 6:
                return False

            # Simple trend detection using linear regression
            x = np.arange(len(df)).reshape(-1, 1)
            y = df['value'].values

            if _has_sklearn:
                from sklearn.linear_model import LinearRegression
                model = LinearRegression()
                model.fit(x, y)
                
                # Check if slope is significant
                slope = model.coef_[0]
                mean_value = np.mean(y)
                
                # Trend is significant if slope > 1% of mean value
                return abs(slope) > 0.01 * mean_value
            else:
                # Fallback: simple slope calculation
                slope = (df['value'].iloc[-1] - df['value'].iloc[0]) / len(df)
                mean_value = df['value'].mean()
                return abs(slope) > 0.01 * mean_value

        except Exception as e:
            logger.error(f"Trend detection failed: {e}")
            return False

    def _generate_forecast(
        self, 
        df: pd.DataFrame, 
        periods: int, 
        method: ForecastMethod, 
        confidence_level: float
    ) -> Optional[ForecastResult]:
        """Generate forecast using the specified method."""
        try:
            if method not in self.models:
                logger.warning(f"Method {method} not available, using moving average")
                method = ForecastMethod.MOVING_AVERAGE

            # Get the model creation function
            model_func = self.models[method]
            
            # Generate forecast
            return model_func(df, periods, confidence_level)

        except Exception as e:
            logger.error(f"Forecast generation failed with method {method}: {e}")
            return self._create_moving_average_model(df, periods, confidence_level)

    def _create_prophet_model(self, df: pd.DataFrame, periods: int, confidence_level: float) -> ForecastResult:
        """Create forecast using Prophet."""
        try:
            # Prepare data for Prophet
            prophet_df = df[['date', 'value']].rename(columns={'date': 'ds', 'value': 'y'})

            # Initialize Prophet with Indian holidays
            model = Prophet(
                interval_width=confidence_level,
                yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False
            )

            # Add custom seasonalities for Indian context
            model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
            model.add_seasonality(name='quarterly', period=91.25, fourier_order=3)

            # Fit model
            model.fit(prophet_df)

            # Create future dataframe
            future = model.make_future_dataframe(periods=periods, freq='M')
            
            # Generate forecast
            forecast = model.predict(future)

            # Extract results
            forecast_data = forecast.tail(periods)
            
            dates = [date.strftime("%Y-%m-%d") for date in forecast_data['ds']]
            forecasts = forecast_data['yhat'].tolist()
            lower_bounds = forecast_data['yhat_lower'].tolist()
            upper_bounds = forecast_data['yhat_upper'].tolist()

            # Calculate accuracy metrics on historical data
            historical_forecast = forecast.head(len(df))
            accuracy_metrics = self._calculate_accuracy_metrics(
                df['value'].tolist(), 
                historical_forecast['yhat'].tolist()
            )

            return ForecastResult(
                dates=dates,
                forecasts=forecasts,
                lower_bounds=lower_bounds,
                upper_bounds=upper_bounds,
                confidence_level=confidence_level,
                method_used="Prophet",
                accuracy_metrics=accuracy_metrics,
                seasonality_detected=True,
                trend_direction=self._determine_trend_direction(forecasts)
            )

        except Exception as e:
            logger.error(f"Prophet forecasting failed: {e}")
            raise

    def _create_linear_model(self, df: pd.DataFrame, periods: int, confidence_level: float) -> ForecastResult:
        """Create forecast using Linear Regression."""
        try:
            from sklearn.linear_model import LinearRegression

            # Prepare features
            X = np.arange(len(df)).reshape(-1, 1)
            y = df['value'].values

            # Fit model
            model = LinearRegression()
            model.fit(X, y)

            # Generate future predictions
            future_X = np.arange(len(df), len(df) + periods).reshape(-1, 1)
            forecasts = model.predict(future_X)

            # Generate dates
            last_date = df['date'].iloc[-1]
            dates = [(last_date + timedelta(days=30 * (i + 1))).strftime("%Y-%m-%d") for i in range(periods)]

            # Calculate prediction intervals (simplified)
            residuals = y - model.predict(X)
            std_error = np.std(residuals)
            margin = 1.96 * std_error  # 95% confidence interval

            lower_bounds = (forecasts - margin).tolist()
            upper_bounds = (forecasts + margin).tolist()

            # Calculate accuracy metrics
            historical_pred = model.predict(X)
            accuracy_metrics = self._calculate_accuracy_metrics(y.tolist(), historical_pred.tolist())

            return ForecastResult(
                dates=dates,
                forecasts=forecasts.tolist(),
                lower_bounds=lower_bounds,
                upper_bounds=upper_bounds,
                confidence_level=confidence_level,
                method_used="Linear Regression",
                accuracy_metrics=accuracy_metrics,
                seasonality_detected=False,
                trend_direction=self._determine_trend_direction(forecasts.tolist())
            )

        except Exception as e:
            logger.error(f"Linear regression forecasting failed: {e}")
            raise

    def _create_rf_model(self, df: pd.DataFrame, periods: int, confidence_level: float) -> ForecastResult:
        """Create forecast using Random Forest."""
        try:
            from sklearn.ensemble import RandomForestRegressor

            # Prepare features with lag variables
            features = []
            targets = []

            # Create lag features
            for i in range(3, len(df)):  # Use 3 lags
                feature_row = [
                    df['value'].iloc[i-1],  # lag 1
                    df['value'].iloc[i-2],  # lag 2
                    df['value'].iloc[i-3],  # lag 3
                    df['month'].iloc[i],    # month
                    df.get('is_festival_season', pd.Series([0] * len(df))).iloc[i]  # festival season
                ]
                features.append(feature_row)
                targets.append(df['value'].iloc[i])

            if len(features) < 5:
                raise ValueError("Insufficient data for Random Forest model")

            X = np.array(features)
            y = np.array(targets)

            # Fit model
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)

            # Generate forecasts
            forecasts = []
            last_values = df['value'].tail(3).tolist()
            
            for i in range(periods):
                # Create feature vector for prediction
                future_month = ((df['date'].iloc[-1].month + i) % 12) + 1
                is_festival = 1 if future_month in [9, 10, 11] else 0
                
                feature_vector = last_values + [future_month, is_festival]
                prediction = model.predict([feature_vector])[0]
                forecasts.append(prediction)
                
                # Update last_values for next prediction
                last_values = last_values[1:] + [prediction]

            # Generate dates
            last_date = df['date'].iloc[-1]
            dates = [(last_date + timedelta(days=30 * (i + 1))).strftime("%Y-%m-%d") for i in range(periods)]

            # Calculate prediction intervals using model's estimators
            predictions_all = np.array([tree.predict([feature_vector])[0] for tree in model.estimators_])
            std_pred = np.std(predictions_all)
            margin = 1.96 * std_pred

            lower_bounds = [f - margin for f in forecasts]
            upper_bounds = [f + margin for f in forecasts]

            # Calculate accuracy metrics
            historical_pred = model.predict(X)
            accuracy_metrics = self._calculate_accuracy_metrics(y.tolist(), historical_pred.tolist())

            return ForecastResult(
                dates=dates,
                forecasts=forecasts,
                lower_bounds=lower_bounds,
                upper_bounds=upper_bounds,
                confidence_level=confidence_level,
                method_used="Random Forest",
                accuracy_metrics=accuracy_metrics,
                seasonality_detected=True,
                trend_direction=self._determine_trend_direction(forecasts)
            )

        except Exception as e:
            logger.error(f"Random Forest forecasting failed: {e}")
            raise

    def _create_exp_smoothing_model(self, df: pd.DataFrame, periods: int, confidence_level: float) -> ForecastResult:
        """Create forecast using Exponential Smoothing."""
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing

            values = df['value'].values
            
            # Determine seasonality
            seasonal = 'add' if len(values) >= 24 else None
            seasonal_periods = 12 if seasonal else None

            # Fit model
            model = ExponentialSmoothing(
                values,
                trend='add',
                seasonal=seasonal,
                seasonal_periods=seasonal_periods
            )
            fitted_model = model.fit()

            # Generate forecast
            forecast = fitted_model.forecast(periods)
            
            # Get prediction intervals
            forecast_ci = fitted_model.get_prediction(
                start=len(values), 
                end=len(values) + periods - 1
            ).conf_int(alpha=1-confidence_level)

            # Generate dates
            last_date = df['date'].iloc[-1]
            dates = [(last_date + timedelta(days=30 * (i + 1))).strftime("%Y-%m-%d") for i in range(periods)]

            # Calculate accuracy metrics
            fitted_values = fitted_model.fittedvalues
            accuracy_metrics = self._calculate_accuracy_metrics(
                values[1:].tolist(),  # Skip first value as it's not fitted
                fitted_values.tolist()
            )

            return ForecastResult(
                dates=dates,
                forecasts=forecast.tolist(),
                lower_bounds=forecast_ci.iloc[:, 0].tolist(),
                upper_bounds=forecast_ci.iloc[:, 1].tolist(),
                confidence_level=confidence_level,
                method_used="Exponential Smoothing",
                accuracy_metrics=accuracy_metrics,
                seasonality_detected=seasonal is not None,
                trend_direction=self._determine_trend_direction(forecast.tolist())
            )

        except Exception as e:
            logger.error(f"Exponential Smoothing forecasting failed: {e}")
            raise

    def _create_moving_average_model(self, df: pd.DataFrame, periods: int, confidence_level: float) -> ForecastResult:
        """Create forecast using Moving Average (fallback method)."""
        try:
            values = df['value'].values
            
            # Use last 6 months for moving average
            window = min(6, len(values))
            moving_avg = np.mean(values[-window:])
            
            # Simple forecast: repeat moving average
            forecasts = [moving_avg] * periods
            
            # Calculate standard deviation for confidence intervals
            std_dev = np.std(values[-window:])
            margin = 1.96 * std_dev  # 95% confidence interval
            
            lower_bounds = [moving_avg - margin] * periods
            upper_bounds = [moving_avg + margin] * periods
            
            # Generate dates
            last_date = df['date'].iloc[-1]
            dates = [(last_date + timedelta(days=30 * (i + 1))).strftime("%Y-%m-%d") for i in range(periods)]
            
            # Calculate accuracy metrics (using naive forecast)
            naive_forecast = [values[-1]] * (len(values) - 1)
            actual_values = values[1:].tolist()
            accuracy_metrics = self._calculate_accuracy_metrics(actual_values, naive_forecast)

            return ForecastResult(
                dates=dates,
                forecasts=forecasts,
                lower_bounds=lower_bounds,
                upper_bounds=upper_bounds,
                confidence_level=confidence_level,
                method_used="Moving Average",
                accuracy_metrics=accuracy_metrics,
                seasonality_detected=False,
                trend_direction="stable"
            )

        except Exception as e:
            logger.error(f"Moving Average forecasting failed: {e}")
            raise

    def _create_seasonal_naive_model(self, df: pd.DataFrame, periods: int, confidence_level: float) -> ForecastResult:
        """Create forecast using Seasonal Naive method."""
        try:
            values = df['value'].values
            
            if len(values) < 12:
                # Fall back to simple naive
                forecasts = [values[-1]] * periods
            else:
                # Use seasonal pattern (12 months)
                seasonal_pattern = values[-12:]
                forecasts = []
                
                for i in range(periods):
                    seasonal_index = i % 12
                    forecasts.append(seasonal_pattern[seasonal_index])
            
            # Calculate confidence intervals
            std_dev = np.std(values)
            margin = 1.96 * std_dev
            
            lower_bounds = [f - margin for f in forecasts]
            upper_bounds = [f + margin for f in forecasts]
            
            # Generate dates
            last_date = df['date'].iloc[-1]
            dates = [(last_date + timedelta(days=30 * (i + 1))).strftime("%Y-%m-%d") for i in range(periods)]
            
            # Calculate accuracy metrics
            if len(values) >= 12:
                seasonal_forecast = []
                for i in range(12, len(values)):
                    seasonal_forecast.append(values[i - 12])
                actual_values = values[12:].tolist()
                accuracy_metrics = self._calculate_accuracy_metrics(actual_values, seasonal_forecast)
            else:
                accuracy_metrics = {"mae": 0, "mse": 0, "rmse": 0, "mape": 0}

            return ForecastResult(
                dates=dates,
                forecasts=forecasts,
                lower_bounds=lower_bounds,
                upper_bounds=upper_bounds,
                confidence_level=confidence_level,
                method_used="Seasonal Naive",
                accuracy_metrics=accuracy_metrics,
                seasonality_detected=len(values) >= 12,
                trend_direction=self._determine_trend_direction(forecasts)
            )

        except Exception as e:
            logger.
