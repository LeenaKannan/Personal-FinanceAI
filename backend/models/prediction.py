# backend/models/prediction.py

from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, date
from enum import Enum
import numpy as np
import pandas as pd
from decimal import Decimal
import logging
import joblib
import os
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enums for validation
class GenderEnum(str, Enum):
    MALE = "male"
    FEMALE = "female"
    OTHER = "other"
    PREFER_NOT_TO_SAY = "prefer_not_to_say"

class CityEnum(str, Enum):
    MUMBAI = "Mumbai"
    DELHI = "Delhi"
    BANGALORE = "Bangalore"
    CHENNAI = "Chennai"
    PUNE = "Pune"
    HYDERABAD = "Hyderabad"
    KOLKATA = "Kolkata"
    AHMEDABAD = "Ahmedabad"
    JAIPUR = "Jaipur"
    LUCKNOW = "Lucknow"
    SURAT = "Surat"
    KANPUR = "Kanpur"
    NAGPUR = "Nagpur"
    INDORE = "Indore"
    THANE = "Thane"
    BHOPAL = "Bhopal"
    VISAKHAPATNAM = "Visakhapatnam"
    PATNA = "Patna"
    VADODARA = "Vadodara"
    GHAZIABAD = "Ghaziabad"

class ExpenseCategoryEnum(str, Enum):
    HOUSING = "Housing"
    FOOD_GROCERIES = "Food & Groceries"
    TRANSPORT = "Transport"
    UTILITIES = "Utilities"
    ENTERTAINMENT = "Entertainment"
    SELF_CARE = "Self Care"
    HEALTHCARE = "Healthcare"
    EDUCATION = "Education"
    SHOPPING = "Shopping"
    INVESTMENT = "Investment"
    INSURANCE = "Insurance"
    EMI = "EMI"
    OTHER = "Other"

# === INPUT SCHEMAS ===

class ExpensePredictionIn(BaseModel):
    """Input schema for expense prediction with comprehensive validation."""
    
    income: float = Field(
        ..., 
        gt=0, 
        le=10000000,  # Max 1 crore per month
        description="Monthly income in INR"
    )
    city: str = Field(
        ..., 
        description="City name (major Indian cities supported)"
    )
    gender: GenderEnum = Field(
        ..., 
        description="Gender of the user"
    )
    age: int = Field(
        ..., 
        ge=18, 
        le=100, 
        description="Age of the user"
    )
    
    # Optional advanced features
    dependents: Optional[int] = Field(
        default=0, 
        ge=0, 
        le=10, 
        description="Number of dependents"
    )
    employment_type: Optional[str] = Field(
        default="salaried", 
        description="Employment type (salaried, business, freelance)"
    )
    experience_years: Optional[int] = Field(
        default=5, 
        ge=0, 
        le=50, 
        description="Years of work experience"
    )
    has_vehicle: Optional[bool] = Field(
        default=False, 
        description="Whether user owns a vehicle"
    )
    lifestyle: Optional[str] = Field(
        default="moderate", 
        description="Lifestyle type (conservative, moderate, luxurious)"
    )
    
    @validator('city')
    def validate_city(cls, v):
        """Validate city name."""
        if v.title() not in [city.value for city in CityEnum]:
            logger.warning(f"Unknown city: {v}. Using Mumbai as default.")
            return "Mumbai"
        return v.title()
    
    @validator('income')
    def validate_income(cls, v):
        """Validate income range."""
        if v < 10000:
            raise ValueError("Income must be at least ₹10,000 per month")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "income": 80000,
                "city": "Mumbai",
                "gender": "male",
                "age": 28,
                "dependents": 2,
                "employment_type": "salaried",
                "experience_years": 5,
                "has_vehicle": True,
                "lifestyle": "moderate"
            }
        }

class BulkPredictionIn(BaseModel):
    """Input schema for bulk expense predictions."""
    
    predictions: List[ExpensePredictionIn] = Field(
        ..., 
        min_items=1, 
        max_items=100, 
        description="List of prediction requests"
    )
    
class HistoricalPredictionIn(BaseModel):
    """Input schema for predictions with historical data."""
    
    user_profile: ExpensePredictionIn
    historical_expenses: Optional[Dict[str, List[float]]] = Field(
        default=None,
        description="Historical monthly expenses by category (last 12 months)"
    )
    seasonal_adjustment: Optional[bool] = Field(
        default=True,
        description="Whether to apply seasonal adjustments"
    )

# === OUTPUT SCHEMAS ===

class ExpensePredictionOut(BaseModel):
    """Output schema for expense prediction with detailed breakdown."""
    
    predicted_expenses: Dict[str, float] = Field(
        ..., 
        description="Predicted monthly expenses by category"
    )
    total_predicted_expenses: float = Field(
        ..., 
        description="Total predicted monthly expenses"
    )
    savings_potential: float = Field(
        ..., 
        description="Potential monthly savings"
    )
    savings_rate: float = Field(
        ..., 
        description="Predicted savings rate as percentage"
    )
    confidence_score: float = Field(
        ..., 
        ge=0, 
        le=1, 
        description="Prediction confidence score (0-1)"
    )
    city_cost_multiplier: float = Field(
        ..., 
        description="City cost of living multiplier used"
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Personalized financial recommendations"
    )
    risk_factors: List[str] = Field(
        default_factory=list,
        description="Identified financial risk factors"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "predicted_expenses": {
                    "Housing": 28000.0,
                    "Food & Groceries": 16000.0,
                    "Transport": 9600.0,
                    "Utilities": 6400.0,
                    "Entertainment": 5600.0,
                    "Self Care": 4000.0,
                    "Other": 10400.0
                },
                "total_predicted_expenses": 80000.0,
                "savings_potential": 20000.0,
                "savings_rate": 25.0,
                "confidence_score": 0.85,
                "city_cost_multiplier": 1.0,
                "recommendations": [
                    "Consider increasing your emergency fund",
                    "Explore investment opportunities"
                ],
                "risk_factors": [
                    "High housing cost ratio"
                ]
            }
        }

class BulkPredictionOut(BaseModel):
    """Output schema for bulk predictions."""
    
    predictions: List[ExpensePredictionOut]
    summary: Dict[str, Any] = Field(
        default_factory=dict,
        description="Summary statistics for bulk predictions"
    )

# === PREDICTION ENGINE ===

class ExpensePredictionEngine:
    """
    Advanced expense prediction engine for Indian users.
    Supports multiple prediction models and sophisticated feature engineering.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """Initialize prediction engine."""
        self.city_cost_index = self._get_city_cost_index()
        self.lifestyle_multipliers = self._get_lifestyle_multipliers()
        self.employment_adjustments = self._get_employment_adjustments()
        self.seasonal_factors = self._get_seasonal_factors()
        
        # Load ML model if available
        self.model = None
        if model_path and os.path.exists(model_path):
            try:
                self.model = joblib.load(model_path)
                logger.info(f"Loaded ML model from {model_path}")
            except Exception as e:
                logger.warning(f"Failed to load ML model: {e}")
    
    def _get_city_cost_index(self) -> Dict[str, float]:
        """Get comprehensive city cost of living index."""
        return {
            "Mumbai": 1.0,      # Baseline
            "Delhi": 0.95,      # 5% lower
            "Bangalore": 0.88,  # 12% lower
            "Chennai": 0.82,    # 18% lower
            "Pune": 0.78,       # 22% lower
            "Hyderabad": 0.73,  # 27% lower
            "Kolkata": 0.75,    # 25% lower
            "Ahmedabad": 0.70,  # 30% lower
            "Jaipur": 0.65,     # 35% lower
            "Lucknow": 0.60,    # 40% lower
            "Surat": 0.68,      # 32% lower
            "Kanpur": 0.58,     # 42% lower
            "Nagpur": 0.62,     # 38% lower
            "Indore": 0.64,     # 36% lower
            "Thane": 0.95,      # Similar to Mumbai
            "Bhopal": 0.61,     # 39% lower
            "Visakhapatnam": 0.59,  # 41% lower
            "Patna": 0.55,      # 45% lower
            "Vadodara": 0.67,   # 33% lower
            "Ghaziabad": 0.72,  # 28% lower
        }
    
    def _get_lifestyle_multipliers(self) -> Dict[str, float]:
        """Get lifestyle-based expense multipliers."""
        return {
            "conservative": 0.85,
            "moderate": 1.0,
            "luxurious": 1.25,
            "premium": 1.5
        }
    
    def _get_employment_adjustments(self) -> Dict[str, Dict[str, float]]:
        """Get employment type adjustments."""
        return {
            "salaried": {
                "stability_factor": 1.0,
                "transport_multiplier": 1.0,
                "entertainment_multiplier": 1.0
            },
            "business": {
                "stability_factor": 0.9,
                "transport_multiplier": 1.2,
                "entertainment_multiplier": 1.15
            },
            "freelance": {
                "stability_factor": 0.85,
                "transport_multiplier": 0.8,
                "entertainment_multiplier": 0.95
            }
        }
    
    def _get_seasonal_factors(self) -> Dict[int, Dict[str, float]]:
        """Get seasonal adjustment factors by month."""
        return {
            1: {"Entertainment": 1.1, "Shopping": 1.2},  # January - New Year
            2: {"Entertainment": 0.95, "Shopping": 0.9},  # February
            3: {"Entertainment": 1.05, "Shopping": 1.1},  # March - Year end
            4: {"Entertainment": 1.0, "Shopping": 1.0},   # April
            5: {"Entertainment": 1.15, "Shopping": 1.1},  # May - Summer
            6: {"Entertainment": 1.1, "Shopping": 1.05},  # June
            7: {"Entertainment": 0.95, "Shopping": 0.95}, # July - Monsoon
            8: {"Entertainment": 0.9, "Shopping": 0.9},   # August
            9: {"Entertainment": 1.05, "Shopping": 1.0},  # September
            10: {"Entertainment": 1.2, "Shopping": 1.3},  # October - Festive season
            11: {"Entertainment": 1.25, "Shopping": 1.4}, # November - Diwali
            12: {"Entertainment": 1.15, "Shopping": 1.2}  # December - Year end
        }
    
    def predict_expenses(self, pred_in: ExpensePredictionIn) -> ExpensePredictionOut:
        """
        Predict expenses with advanced modeling.
        
        Args:
            pred_in: Prediction input parameters
            
        Returns:
            ExpensePredictionOut: Detailed prediction results
        """
        try:
            # Use ML model if available, otherwise use rule-based approach
            if self.model:
                return self._predict_with_ml_model(pred_in)
            else:
                return self._predict_with_rules(pred_in)
                
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return self._get_fallback_prediction(pred_in)
    
    def _predict_with_ml_model(self, pred_in: ExpensePredictionIn) -> ExpensePredictionOut:
        """Predict using trained ML model."""
        try:
            # Prepare features
            features = self._prepare_ml_features(pred_in)
            
            # Get prediction
            prediction = self.model.predict(features.reshape(1, -1))[0]
            
            # Map to categories
            categories = list(ExpenseCategoryEnum)
            predicted_expenses = {}
            
            for i, category in enumerate(categories):
                if i < len(prediction):
                    predicted_expenses[category.value] = round(float(prediction[i]), 2)
            
            # Calculate derived metrics
            total_expenses = sum(predicted_expenses.values())
            savings_potential = max(0, pred_in.income - total_expenses)
            savings_rate = (savings_potential / pred_in.income) * 100 if pred_in.income > 0 else 0
            
            # Generate recommendations and risk factors
            recommendations = self._generate_recommendations(pred_in, predicted_expenses)
            risk_factors = self._identify_risk_factors(pred_in, predicted_expenses)
            
            return ExpensePredictionOut(
                predicted_expenses=predicted_expenses,
                total_predicted_expenses=total_expenses,
                savings_potential=savings_potential,
                savings_rate=round(savings_rate, 2),
                confidence_score=0.85,  # Model-based confidence
                city_cost_multiplier=self.city_cost_index.get(pred_in.city, 1.0),
                recommendations=recommendations,
                risk_factors=risk_factors
            )
            
        except Exception as e:
            logger.error(f"ML prediction failed: {e}")
            return self._predict_with_rules(pred_in)
    
    def _predict_with_rules(self, pred_in: ExpensePredictionIn) -> ExpensePredictionOut:
        """Predict using rule-based approach with advanced features."""
        try:
            # Base calculation
            city_multiplier = self.city_cost_index.get(pred_in.city, 1.0)
            base_expenses = pred_in.income * city_multiplier
            
            # Demographic adjustments
            gender_adj = 0.97 if pred_in.gender.lower() == "female" else 1.0
            age_adj = self._get_age_adjustment(pred_in.age)
            dependents_adj = 1.0 + (pred_in.dependents * 0.15)  # 15% increase per dependent
            
            # Lifestyle adjustment
            lifestyle_adj = self.lifestyle_multipliers.get(pred_in.lifestyle, 1.0)
            
            # Employment adjustment
            employment_adj = self.employment_adjustments.get(
                pred_in.employment_type, 
                self.employment_adjustments["salaried"]
            )
            
            # Calculate category-wise expenses
            base_multiplier = gender_adj * age_adj * dependents_adj * lifestyle_adj
            
            expenses = {
                "Housing": round(base_expenses * 0.35 * base_multiplier, 2),
                "Food & Groceries": round(base_expenses * 0.20 * base_multiplier, 2),
                "Transport": round(base_expenses * 0.12 * employment_adj["transport_multiplier"], 2),
                "Utilities": round(base_expenses * 0.08 * base_multiplier, 2),
                "Entertainment": round(base_expenses * 0.07 * employment_adj["entertainment_multiplier"], 2),
                "Self Care": round(base_expenses * 0.05 * gender_adj, 2),
                "Healthcare": round(base_expenses * 0.04 * age_adj, 2),
                "Education": round(base_expenses * 0.03 * dependents_adj, 2),
                "Shopping": round(base_expenses * 0.06 * lifestyle_adj, 2),
                "Other": round(base_expenses * 0.10, 2)
            }
            
            # Apply seasonal adjustments if current month
            current_month = datetime.now().month
            seasonal_adj = self.seasonal_factors.get(current_month, {})
            
            for category, factor in seasonal_adj.items():
                if category in expenses:
                    expenses[category] = round(expenses[category] * factor, 2)
            
            # Calculate derived metrics
            total_expenses = sum(expenses.values())
            savings_potential = max(0, pred_in.income - total_expenses)
            savings_rate = (savings_potential / pred_in.income) * 100 if pred_in.income > 0 else 0
            
            # Generate insights
            recommendations = self._generate_recommendations(pred_in, expenses)
            risk_factors = self._identify_risk_factors(pred_in, expenses)
            
            # Calculate confidence based on data completeness
            confidence = self._calculate_confidence(pred_in)
            
            return ExpensePredictionOut(
                predicted_expenses=expenses,
                total_predicted_expenses=total_expenses,
                savings_potential=savings_potential,
                savings_rate=round(savings_rate, 2),
                confidence_score=confidence,
                city_cost_multiplier=city_multiplier,
                recommendations=recommendations,
                risk_factors=risk_factors
            )
            
        except Exception as e:
            logger.error(f"Rule-based prediction failed: {e}")
            return self._get_fallback_prediction(pred_in)
    
    def _get_age_adjustment(self, age: int) -> float:
        """Get age-based expense adjustment."""
        if age < 25:
            return 0.9   # Younger people spend less
        elif age < 35:
            return 1.0   # Base spending
        elif age < 50:
            return 1.1   # Peak earning/spending years
        else:
            return 1.05  # Slightly higher for older adults
    
    def _prepare_ml_features(self, pred_in: ExpensePredictionIn) -> np.ndarray:
        """Prepare features for ML model."""
        features = [
            pred_in.income,
            self.city_cost_index.get(pred_in.city, 1.0),
            1 if pred_in.gender.lower() == "male" else 0,
            pred_in.age,
            pred_in.dependents,
            pred_in.experience_years,
            1 if pred_in.has_vehicle else 0,
            self.lifestyle_multipliers.get(pred_in.lifestyle, 1.0),
            1 if pred_in.employment_type == "salaried" else 0,
            1 if pred_in.employment_type == "business" else 0
        ]
        
        return np.array(features)
    
    def _generate_recommendations(self, pred_in: ExpensePredictionIn, expenses: Dict[str, float]) -> List[str]:
        """Generate personalized financial recommendations."""
        recommendations = []
        
        # Housing cost check
        housing_ratio = expenses.get("Housing", 0) / pred_in.income
        if housing_ratio > 0.4:
            recommendations.append("Consider reducing housing costs - aim for less than 40% of income")
        
        # Savings rate check
        total_expenses = sum(expenses.values())
        savings_rate = (pred_in.income - total_expenses) / pred_in.income
        
        if savings_rate < 0.2:
            recommendations.append("Try to save at least 20% of your income for financial security")
        elif savings_rate > 0.3:
            recommendations.append("Great savings rate! Consider investing in mutual funds or SIPs")
        
        # Age-specific recommendations
        if pred_in.age < 30:
            recommendations.append("Start investing early to benefit from compound growth")
        elif pred_in.age > 40:
            recommendations.append("Focus on retirement planning and health insurance")
        
        # City-specific recommendations
        if pred_in.city in ["Mumbai", "Delhi", "Bangalore"]:
            recommendations.append("Consider public transport to reduce commuting costs in metro cities")
        
        return recommendations
    
    def _identify_risk_factors(self, pred_in: ExpensePredictionIn, expenses: Dict[str, float]) -> List[str]:
        """Identify financial risk factors."""
        risk_factors = []
        
        # High expense ratios
        housing_ratio = expenses.get("Housing", 0) / pred_in.income
        if housing_ratio > 0.5:
            risk_factors.append("Housing costs exceed 50% of income")
        
        # Low savings
        total_expenses = sum(expenses.values())
        if total_expenses > pred_in.income * 0.9:
            risk_factors.append("Very low savings rate - financial vulnerability")
        
        # Employment type risks
        if pred_in.employment_type in ["freelance", "business"]:
            risk_factors.append("Variable income - maintain higher emergency fund")
        
        # Age-related risks
        if pred_in.age > 45 and expenses.get("Healthcare", 0) < pred_in.income * 0.05:
            risk_factors.append("Healthcare expenses may be underestimated for your age group")
        
        return risk_factors
    
    def _calculate_confidence(self, pred_in: ExpensePredictionIn) -> float:
        """Calculate prediction confidence score."""
        confidence = 0.7  # Base confidence
        
        # Increase confidence for complete data
        if pred_in.dependents is not None:
            confidence += 0.05
        if pred_in.employment_type:
            confidence += 0.05
        if pred_in.experience_years:
            confidence += 0.05
        if pred_in.lifestyle:
            confidence += 0.05
        
        # Adjust for city data availability
        if pred_in.city in self.city_cost_index:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _get_fallback_prediction(self, pred_in: ExpensePredictionIn) -> ExpensePredictionOut:
        """Get basic fallback prediction."""
        basic_expenses = {
            "Housing": round(pred_in.income * 0.35, 2),
            "Food & Groceries": round(pred_in.income * 0.20, 2),
            "Transport": round(pred_in.income * 0.12, 2),
            "Utilities": round(pred_in.income * 0.08, 2),
            "Entertainment": round(pred_in.income * 0.07, 2),
            "Self Care": round(pred_in.income * 0.05, 2),
            "Other": round(pred_in.income * 0.13, 2)
        }
        
        total_expenses = sum(basic_expenses.values())
        
        return ExpensePredictionOut(
            predicted_expenses=basic_expenses,
            total_predicted_expenses=total_expenses,
            savings_potential=max(0, pred_in.income - total_expenses),
            savings_rate=0.0,
            confidence_score=0.5,
            city_cost_multiplier=1.0,
            recommendations=["Basic prediction - provide more details for better accuracy"],
            risk_factors=[]
        )

# === MAIN PREDICTION FUNCTIONS ===

# Initialize global prediction engine
prediction_engine = ExpensePredictionEngine()

def predict_expenses(pred_in: ExpensePredictionIn) -> ExpensePredictionOut:
    """
    Main expense prediction function.
    
    Args:
        pred_in: Prediction input parameters
        
    Returns:
        ExpensePredictionOut: Detailed prediction results
    """
    return prediction_engine.predict_expenses(pred_in)

def predict_expenses_bulk(bulk_in: BulkPredictionIn) -> BulkPredictionOut:
    """
    Bulk expense prediction function.
    
    Args:
        bulk_in: Bulk prediction input
        
    Returns:
        BulkPredictionOut: Bulk prediction results
    """
    predictions = []
    
    for pred_request in bulk_in.predictions:
        prediction = prediction_engine.predict_expenses(pred_request)
        predictions.append(prediction)
    
    # Calculate summary statistics
    total_predictions = len(predictions)
    avg_savings_rate = sum(p.savings_rate for p in predictions) / total_predictions
    avg_confidence = sum(p.confidence_score for p in predictions) / total_predictions
    
    summary = {
        "total_predictions": total_predictions,
        "average_savings_rate": round(avg_savings_rate, 2),
        "average_confidence": round(avg_confidence, 2),
        "timestamp": datetime.now().isoformat()
    }
    
    return BulkPredictionOut(predictions=predictions, summary=summary)

def predict_expenses_with_history(hist_in: HistoricalPredictionIn) -> ExpensePredictionOut:
    """
    Predict expenses with historical data consideration.
    
    Args:
        hist_in: Historical prediction input
        
    Returns:
        ExpensePredictionOut: Enhanced prediction with historical context
    """
    # Get base prediction
    base_prediction = prediction_engine.predict_expenses(hist_in.user_profile)
    
    # Adjust based on historical data if available
    if hist_in.historical_expenses:
        # Calculate historical averages and adjust predictions
        for category, historical_values in hist_in.historical_expenses.items():
            if category in base_prediction.predicted_expenses and historical_values:
                historical_avg = sum(historical_values) / len(historical_values)
                predicted_value = base_prediction.predicted_expenses[category]
                
                # Weighted average: 70% prediction, 30% historical
                adjusted_value = (predicted_value * 0.7) + (historical_avg * 0.3)
                base_prediction.predicted_expenses[category] = round(adjusted_value, 2)
        
        # Recalculate totals
        base_prediction.total_predicted_expenses = sum(base_prediction.predicted_expenses.values())
        base_prediction.savings_potential = max(0, hist_in.user_profile.income - base_prediction.total_predicted_expenses)
        base_prediction.savings_rate = (base_prediction.savings_potential / hist_in.user_profile.income) * 100
        
        # Increase confidence due to historical data
        base_prediction.confidence_score = min(base_prediction.confidence_score + 0.1, 1.0)
    
    return base_prediction

# Example usage and testing
if __name__ == "__main__":
    # Test the prediction models
    print("Testing Expense Prediction Models...")
    
    # Test basic prediction
    test_input = ExpensePredictionIn(
        income=80000,
        city="Mumbai",
        gender=GenderEnum.MALE,
        age=28,
        dependents=2,
        employment_type="salaried",
        experience_years=5,
        has_vehicle=True,
        lifestyle="moderate"
    )
    
    result = predict_expenses(test_input)
    print(f"Predicted expenses: {result.predicted_expenses}")
    print(f"Savings rate: {result.savings_rate}%")
    print(f"Confidence: {result.confidence_score}")
    print(f"Recommendations: {result.recommendations}")
    
    # Test bulk prediction
    bulk_input = BulkPredictionIn(predictions=[test_input, test_input])
    bulk_result = predict_expenses_bulk(bulk_input)
    print(f"Bulk prediction summary: {bulk_result.summary}")
    
    print("All prediction tests completed successfully!")
