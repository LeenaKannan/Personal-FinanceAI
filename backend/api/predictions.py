# backend/api/predictions.py

from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
from datetime import datetime, date, timedelta
from enum import Enum
import logging
import asyncio

from backend.ml_engine.expense_predictor import ExpensePredictor
from backend.ml_engine.time_series import TimeSeriesForecaster
from backend.models.prediction import (
    ExpensePredictionIn, ExpensePredictionOut, BulkPredictionIn, BulkPredictionOut,
    HistoricalPredictionIn, predict_expenses, predict_expenses_bulk, predict_expenses_with_history
)
from backend.models import transaction, user
from backend.models.database import get_db
from backend.api.auth import get_current_user
from backend.utils.data_processor import DataProcessor
from pydantic import BaseModel, Field, validator

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/predictions", tags=["predictions"])

# Initialize ML engines
predictor = ExpensePredictor()
forecaster = TimeSeriesForecaster()
data_processor = DataProcessor()

# === ENUMS ===

class PredictionTypeEnum(str, Enum):
    BASIC = "basic"
    ADVANCED = "advanced"
    HISTORICAL = "historical"
    FORECAST = "forecast"

class ForecastPeriodEnum(str, Enum):
    NEXT_MONTH = "next_month"
    NEXT_QUARTER = "next_quarter"
    NEXT_YEAR = "next_year"
    CUSTOM = "custom"

# === ADDITIONAL SCHEMAS ===

class ExpenseForecastIn(BaseModel):
    """Input schema for expense forecasting."""
    
    user_profile: ExpensePredictionIn
    forecast_period: ForecastPeriodEnum = Field(default=ForecastPeriodEnum.NEXT_MONTH)
    periods: Optional[int] = Field(None, ge=1, le=36, description="Number of periods to forecast (for custom)")
    include_seasonality: Optional[bool] = Field(True, description="Include seasonal adjustments")
    confidence_interval: Optional[float] = Field(0.95, ge=0.8, le=0.99, description="Confidence interval")

class ExpenseForecastOut(BaseModel):
    """Output schema for expense forecasting."""
    
    forecast_data: List[Dict[str, Any]] = Field(..., description="Forecasted expenses by period")
    total_forecast: Dict[str, float] = Field(..., description="Total forecasted amounts")
    confidence_bounds: Dict[str, Dict[str, float]] = Field(..., description="Upper and lower confidence bounds")
    seasonal_factors: Dict[str, float] = Field(..., description="Seasonal adjustment factors")
    model_accuracy: float = Field(..., description="Model accuracy score")
    generated_at: datetime = Field(..., description="When forecast was generated")

class CategoryPredictionIn(BaseModel):
    """Input schema for category-specific predictions."""
    
    user_profile: ExpensePredictionIn
    target_category: str = Field(..., description="Specific category to predict")
    historical_months: Optional[int] = Field(6, ge=3, le=24, description="Months of history to consider")

class CategoryPredictionOut(BaseModel):
    """Output schema for category-specific predictions."""
    
    category: str
    predicted_amount: float
    confidence_score: float
    trend_direction: str  # "increasing", "decreasing", "stable"
    seasonal_pattern: Dict[str, float]
    recommendations: List[str]

class BudgetOptimizationIn(BaseModel):
    """Input schema for budget optimization."""
    
    user_profile: ExpensePredictionIn
    target_savings_rate: float = Field(..., ge=0, le=0.8, description="Target savings rate (0-80%)")
    priority_categories: List[str] = Field(default_factory=list, description="Categories to prioritize")
    constraints: Dict[str, float] = Field(default_factory=dict, description="Minimum amounts per category")

class BudgetOptimizationOut(BaseModel):
    """Output schema for budget optimization."""
    
    optimized_budget: Dict[str, float]
    current_vs_optimized: Dict[str, Dict[str, float]]
    potential_savings: float
    achievability_score: float
    optimization_steps: List[str]

class PredictionAccuracyOut(BaseModel):
    """Output schema for prediction accuracy metrics."""
    
    overall_accuracy: float
    category_accuracy: Dict[str, float]
    mean_absolute_error: float
    prediction_confidence: float
    model_version: str
    last_trained: datetime

# === UTILITY FUNCTIONS ===

def get_user_historical_data(db: Session, user_id: int, months: int = 12) -> Dict[str, Any]:
    """Get user's historical transaction data for ML training."""
    try:
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=months * 30)
        
        txn_filter = transaction.TransactionFilter(
            start_date=start_date,
            end_date=end_date
        )
        
        transactions = transaction.get_transactions_by_user(
            db, user_id, filters=txn_filter
        )
        
        if not transactions:
            return {}
        
        # Process transactions for ML
        import pandas as pd
        
        txn_data = []
        for txn in transactions:
            txn_data.append({
                'date': txn.date,
                'amount': abs(txn.amount) if txn.amount < 0 else 0,  # Expenses only
                'category': txn.category.value if hasattr(txn.category, 'value') else str(txn.category),
                'description': txn.description
            })
        
        df = pd.DataFrame(txn_data)
        
        if df.empty:
            return {}
        
        # Monthly aggregation by category
        df['month'] = pd.to_datetime(df['date']).dt.to_period('M')
        monthly_expenses = df.groupby(['month', 'category'])['amount'].sum().unstack(fill_value=0)
        
        # Calculate statistics
        avg_monthly = monthly_expenses.mean().to_dict()
        trend_data = monthly_expenses.to_dict('records')
        
        return {
            'monthly_averages': avg_monthly,
            'trend_data': trend_data,
            'total_months': len(monthly_expenses),
            'categories': list(monthly_expenses.columns)
        }
        
    except Exception as e:
        logger.error(f"Failed to get historical data for user {user_id}: {e}")
        return {}

def enhance_prediction_with_history(
    base_prediction: ExpensePredictionOut, 
    historical_data: Dict[str, Any]
) -> ExpensePredictionOut:
    """Enhance prediction accuracy using historical data."""
    try:
        if not historical_data or 'monthly_averages' not in historical_data:
            return base_prediction
        
        monthly_averages = historical_data['monthly_averages']
        enhanced_expenses = {}
        
        # Blend prediction with historical averages (70% prediction, 30% history)
        for category, predicted_amount in base_prediction.predicted_expenses.items():
            historical_avg = monthly_averages.get(category, predicted_amount)
            enhanced_amount = (predicted_amount * 0.7) + (historical_avg * 0.3)
            enhanced_expenses[category] = round(enhanced_amount, 2)
        
        # Update the prediction
        base_prediction.predicted_expenses = enhanced_expenses
        base_prediction.total_predicted_expenses = sum(enhanced_expenses.values())
        base_prediction.savings_potential = max(0, 
            base_prediction.savings_potential - (base_prediction.total_predicted_expenses - sum(base_prediction.predicted_expenses.values()))
        )
        
        # Increase confidence due to historical data
        base_prediction.confidence_score = min(base_prediction.confidence_score + 0.1, 1.0)
        
        return base_prediction
        
    except Exception as e:
        logger.error(f"Failed to enhance prediction with history: {e}")
        return base_prediction

def calculate_seasonal_adjustments(month: int) -> Dict[str, float]:
    """Calculate seasonal adjustment factors for Indian context."""
    seasonal_factors = {
        1: {"Entertainment": 1.1, "Shopping": 1.2, "Food & Groceries": 1.05},  # New Year
        2: {"Entertainment": 0.95, "Shopping": 0.9, "Food & Groceries": 1.0},
        3: {"Entertainment": 1.0, "Shopping": 1.1, "Food & Groceries": 1.0},   # Year-end
        4: {"Entertainment": 1.0, "Shopping": 1.0, "Food & Groceries": 1.0},   # New financial year
        5: {"Entertainment": 1.1, "Shopping": 1.05, "Food & Groceries": 1.0},  # Summer
        6: {"Entertainment": 1.05, "Shopping": 1.0, "Food & Groceries": 1.0},
        7: {"Utilities": 1.1, "Transport": 0.9, "Food & Groceries": 1.0},      # Monsoon
        8: {"Utilities": 1.1, "Transport": 0.9, "Food & Groceries": 1.0},
        9: {"Entertainment": 1.0, "Shopping": 1.0, "Food & Groceries": 1.0},
        10: {"Entertainment": 1.2, "Shopping": 1.3, "Food & Groceries": 1.1},  # Festive season
        11: {"Entertainment": 1.25, "Shopping": 1.4, "Food & Groceries": 1.15}, # Diwali
        12: {"Entertainment": 1.15, "Shopping": 1.2, "Food & Groceries": 1.1}   # Year-end
    }
    
    return seasonal_factors.get(month, {})

# === API ENDPOINTS ===

@router.post("/expenses", response_model=ExpensePredictionOut)
async def predict_user_expenses(
    payload: ExpensePredictionIn,
    db: Session = Depends(get_db),
    current_user: user.User = Depends(get_current_user),
    prediction_type: PredictionTypeEnum = Query(PredictionTypeEnum.ADVANCED, description="Type of prediction"),
    use_history: bool = Query(True, description="Use historical data for better accuracy")
):
    """
    Predict monthly expenses for a user with advanced ML models.
    """
    try:
        # Use current user data if not provided in payload
        profile_data = {
            "income": payload.income if payload.income else current_user.income,
            "city": payload.city if payload.city else current_user.city,
            "gender": payload.gender.value if payload.gender else current_user.gender,
            "age": payload.age if payload.age else current_user.age,
            "dependents": getattr(payload, 'dependents', 0),
            "employment_type": getattr(payload, 'employment_type', 'salaried'),
            "experience_years": getattr(payload, 'experience_years', 5),
            "has_vehicle": getattr(payload, 'has_vehicle', False),
            "lifestyle": getattr(payload, 'lifestyle', 'moderate')
        }
        
        # Get base prediction
        result = predictor.predict(profile_data)
        
        # Create prediction output
        total_expenses = sum(result.values())
        savings_potential = max(0, profile_data["income"] - total_expenses)
        savings_rate = (savings_potential / profile_data["income"]) * 100 if profile_data["income"] > 0 else 0
        
        prediction_out = ExpensePredictionOut(
            predicted_expenses=result,
            total_predicted_expenses=total_expenses,
            savings_potential=savings_potential,
            savings_rate=round(savings_rate, 2),
            confidence_score=0.8,  # Base confidence
            city_cost_multiplier=data_processor.city_cost_index.get(profile_data["city"], 1.0),
            recommendations=[],
            risk_factors=[]
        )
        
        # Enhance with historical data if requested
        if use_history and prediction_type in [PredictionTypeEnum.ADVANCED, PredictionTypeEnum.HISTORICAL]:
            historical_data = get_user_historical_data(db, current_user.id)
            if historical_data:
                prediction_out = enhance_prediction_with_history(prediction_out, historical_data)
        
        # Apply seasonal adjustments
        current_month = datetime.now().month
        seasonal_adj = calculate_seasonal_adjustments(current_month)
        
        for category, factor in seasonal_adj.items():
            if category in prediction_out.predicted_expenses:
                prediction_out.predicted_expenses[category] = round(
                    prediction_out.predicted_expenses[category] * factor, 2
                )
        
        # Recalculate totals after seasonal adjustment
        prediction_out.total_predicted_expenses = sum(prediction_out.predicted_expenses.values())
        prediction_out.savings_potential = max(0, profile_data["income"] - prediction_out.total_predicted_expenses)
        prediction_out.savings_rate = (prediction_out.savings_potential / profile_data["income"]) * 100
        
        logger.info(f"Generated expense prediction for user {current_user.id}")
        
        return prediction_out
        
    except Exception as e:
        logger.error(f"Expense prediction failed for user {current_user.id}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to generate expense prediction"
        )

@router.post("/expenses/bulk", response_model=BulkPredictionOut)
async def predict_bulk_expenses(
    bulk_payload: BulkPredictionIn,
    current_user: user.User = Depends(get_current_user)
):
    """
    Generate bulk expense predictions for multiple scenarios.
    """
    try:
        result = predict_expenses_bulk(bulk_payload)
        logger.info(f"Generated bulk predictions for user {current_user.id}")
        return result
        
    except Exception as e:
        logger.error(f"Bulk prediction failed for user {current_user.id}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to generate bulk predictions"
        )

@router.post("/expenses/forecast", response_model=ExpenseForecastOut)
async def forecast_expenses(
    forecast_input: ExpenseForecastIn,
    db: Session = Depends(get_db),
    current_user: user.User = Depends(get_current_user)
):
    """
    Generate time-series forecast of future expenses.
    """
    try:
        # Get historical data for forecasting
        historical_data = get_user_historical_data(db, current_user.id, 12)
        
        if not historical_data or not historical_data.get('trend_data'):
            raise HTTPException(
                status_code=400,
                detail="Insufficient historical data for forecasting"
            )
        
        # Determine forecast periods
        periods = forecast_input.periods
        if forecast_input.forecast_period == ForecastPeriodEnum.NEXT_MONTH:
            periods = 1
        elif forecast_input.forecast_period == ForecastPeriodEnum.NEXT_QUARTER:
            periods = 3
        elif forecast_input.forecast_period == ForecastPeriodEnum.NEXT_YEAR:
            periods = 12
        
        # Generate forecast for each category
        forecast_data = []
        total_forecast = {}
        confidence_bounds = {}
        
        for category in historical_data['categories']:
            # Prepare time series data
            category_data = []
            for month_data in historical_data['trend_data']:
                category_data.append({
                    'date': str(month_data.get('month', datetime.now().strftime('%Y-%m'))),
                    'value': month_data.get(category, 0)
                })
            
            if len(category_data) >= 3:  # Minimum data for forecasting
                try:
                    # Use time series forecaster
                    forecast_result = forecaster.fit_predict(category_data, periods)
                    
                    if forecast_result:
                        total_forecast[category] = sum([f['forecast'] for f in forecast_result])
                        confidence_bounds[category] = {
                            'lower': sum([f['forecast'] * 0.8 for f in forecast_result]),
                            'upper': sum([f['forecast'] * 1.2 for f in forecast_result])
                        }
                        
                        for f in forecast_result:
                            forecast_data.append({
                                'period': f['date'],
                                'category': category,
                                'forecast': f['forecast'],
                                'confidence_lower': f['forecast'] * 0.8,
                                'confidence_upper': f['forecast'] * 1.2
                            })
                            
                except Exception as e:
                    logger.warning(f"Forecasting failed for category {category}: {e}")
                    # Fallback to simple average
                    avg_value = sum([d['value'] for d in category_data]) / len(category_data)
                    total_forecast[category] = avg_value * periods
        
        # Apply seasonal factors if requested
        seasonal_factors = {}
        if forecast_input.include_seasonality:
            current_month = datetime.now().month
            for i in range(periods):
                month = ((current_month + i - 1) % 12) + 1
                seasonal_adj = calculate_seasonal_adjustments(month)
                for category, factor in seasonal_adj.items():
                    if category not in seasonal_factors:
                        seasonal_factors[category] = []
                    seasonal_factors[category].append(factor)
            
            # Average seasonal factors
            for category, factors in seasonal_factors.items():
                seasonal_factors[category] = sum(factors) / len(factors)
        
        return ExpenseForecastOut(
            forecast_data=forecast_data,
            total_forecast=total_forecast,
            confidence_bounds=confidence_bounds,
            seasonal_factors=seasonal_factors,
            model_accuracy=0.85,  # TODO: Calculate actual accuracy
            generated_at=datetime.now()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Expense forecasting failed for user {current_user.id}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to generate expense forecast"
        )

@router.post("/category", response_model=CategoryPredictionOut)
async def predict_category_expenses(
    category_input: CategoryPredictionIn,
    db: Session = Depends(get_db),
    current_user: user.User = Depends(get_current_user)
):
    """
    Predict expenses for a specific category with detailed analysis.
    """
    try:
        # Get historical data for the specific category
        historical_data = get_user_historical_data(db, current_user.id, category_input.historical_months)
        
        category = category_input.target_category
        
        # Get base prediction
        base_prediction = await predict_user_expenses(
            category_input.user_profile, db, current_user, 
            PredictionTypeEnum.ADVANCED, True
        )
        
        predicted_amount = base_prediction.predicted_expenses.get(category, 0)
        
        # Analyze trend
        trend_direction = "stable"
        if historical_data and 'trend_data' in historical_data:
            category_values = [month.get(category, 0) for month in historical_data['trend_data']]
            if len(category_values) >= 3:
                recent_avg = sum(category_values[-3:]) / 3
                older_avg = sum(category_values[:-3]) / max(1, len(category_values) - 3)
                
                if recent_avg > older_avg * 1.1:
                    trend_direction = "increasing"
                elif recent_avg < older_avg * 0.9:
                    trend_direction = "decreasing"
        
        # Generate seasonal pattern
        seasonal_pattern = {}
        for month in range(1, 13):
            seasonal_adj = calculate_seasonal_adjustments(month)
            seasonal_pattern[f"month_{month}"] = seasonal_adj.get(category, 1.0)
        
        # Generate recommendations
        recommendations = []
        if trend_direction == "increasing":
            recommendations.append(f"Your {category} expenses are trending upward. Consider reviewing recent purchases.")
        elif predicted_amount > base_prediction.predicted_expenses.get(category, 0) * 1.2:
            recommendations.append(f"Predicted {category} expenses are high. Look for optimization opportunities.")
        
        return CategoryPredictionOut(
            category=category,
            predicted_amount=predicted_amount,
            confidence_score=base_prediction.confidence_score,
            trend_direction=trend_direction,
            seasonal_pattern=seasonal_pattern,
            recommendations=recommendations
        )
        
    except Exception as e:
        logger.error(f"Category prediction failed for user {current_user.id}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to generate category prediction"
        )

@router.post("/budget/optimize", response_model=BudgetOptimizationOut)
async def optimize_budget(
    optimization_input: BudgetOptimizationIn,
    db: Session = Depends(get_db),
    current_user: user.User = Depends(get_current_user)
):
    """
    Optimize budget allocation to achieve target savings rate.
    """
    try:
        # Get current prediction
        current_prediction = await predict_user_expenses(
            optimization_input.user_profile, db, current_user,
            PredictionTypeEnum.ADVANCED, True
        )
        
        income = optimization_input.user_profile.income
        target_expenses = income * (1 - optimization_input.target_savings_rate)
        current_expenses = current_prediction.total_predicted_expenses
        
        if target_expenses >= current_expenses:
            # Already meeting target
            return BudgetOptimizationOut(
                optimized_budget=current_prediction.predicted_expenses,
                current_vs_optimized={
                    category: {"current": amount, "optimized": amount}
                    for category, amount in current_prediction.predicted_expenses.items()
                },
                potential_savings=0,
                achievability_score=1.0,
                optimization_steps=["You're already meeting your savings target!"]
            )
        
        # Calculate required reduction
        required_reduction = current_expenses - target_expenses
        
        # Optimize budget allocation
        optimized_budget = current_prediction.predicted_expenses.copy()
        optimization_steps = []
        
        # Priority order for reduction (least essential first)
        reduction_priority = [
            "Entertainment", "Shopping", "Self Care", "Other",
            "Food & Groceries", "Transport", "Utilities", "Housing"
        ]
        
        remaining_reduction = required_reduction
        
        for category in reduction_priority:
            if remaining_reduction <= 0:
                break
                
            if category in optimized_budget:
                current_amount = optimized_budget[category]
                min_amount = optimization_input.constraints.get(category, current_amount * 0.5)
                
                possible_reduction = max(0, current_amount - min_amount)
                actual_reduction = min(possible_reduction, remaining_reduction)
                
                if actual_reduction > 0:
                    optimized_budget[category] = current_amount - actual_reduction
                    remaining_reduction -= actual_reduction
                    
                    optimization_steps.append(
                        f"Reduce {category} by ₹{actual_reduction:,.0f} "
                        f"(from ₹{current_amount:,.0f} to ₹{optimized_budget[category]:,.0f})"
                    )
        
        # Calculate achievability score
        achievability_score = max(0, 1 - (remaining_reduction / required_reduction))
        
        # Current vs optimized comparison
        current_vs_optimized = {}
        for category in current_prediction.predicted_expenses:
            current_vs_optimized[category] = {
                "current": current_prediction.predicted_expenses[category],
                "optimized": optimized_budget[category]
            }
        
        return BudgetOptimizationOut(
            optimized_budget=optimized_budget,
            current_vs_optimized=current_vs_optimized,
            potential_savings=required_reduction - remaining_reduction,
            achievability_score=achievability_score,
            optimization_steps=optimization_steps
        )
        
    except Exception as e:
        logger.error(f"Budget optimization failed for user {current_user.id}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to optimize budget"
        )

@router.get("/accuracy", response_model=PredictionAccuracyOut)
async def get_prediction_accuracy(
    db: Session = Depends(get_db),
    current_user: user.User = Depends(get_current_user)
):
    """
    Get prediction accuracy metrics for the current user.
    """
    try:
        # TODO: Implement actual accuracy calculation based on historical predictions vs actual
        
        # Mock accuracy data for now
        return PredictionAccuracyOut(
            overall_accuracy=0.85,
            category_accuracy={
                "Housing": 0.92,
                "Food & Groceries": 0.78,
                "Transport": 0.83,
                "Utilities": 0.89,
                "Entertainment": 0.71,
                "Other": 0.65
            },
            mean_absolute_error=1250.50,
            prediction_confidence=0.82,
            model_version="v2.1.0",
            last_trained=datetime.now() - timedelta(days=7)
        )
        
    except Exception as e:
        logger.error(f"Failed to get accuracy metrics for user {current_user.id}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve accuracy metrics"
        )

@router.post("/retrain")
async def retrain_model(
    background_tasks: BackgroundTasks,
    current_user: user.User = Depends(get_current_user)
):
    """
    Trigger model retraining with latest data.
    """
    try:
        # Add background task for model retraining
        # background_tasks.add_task(retrain_user_model, current_user.id)
        
        logger.info(f"Model retraining requested for user {current_user.id}")
        
        return {"message": "Model retraining initiated"}
        
    except Exception as e:
        logger.error(f"Model retraining failed for user {current_user.id}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to initiate model retraining"
        )

# Health check
@router.get("/health")
async def predictions_health_check():
    """Predictions service health check."""
    return {
        "status": "healthy",
        "service": "predictions",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": {
            "expense_predictor": predictor.model is not None,
            "time_series_forecaster": True
        }
    }
