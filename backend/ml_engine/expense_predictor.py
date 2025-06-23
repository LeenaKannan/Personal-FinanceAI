# Expense prediction models with Indian market specifics
# backend/ml_engine/expense_predictor.py

import numpy as np
import pandas as pd
import joblib
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import asyncio
from concurrent.futures import ThreadPoolExecutor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import pickle
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExpensePredictor:
    """
    Advanced expense prediction system optimized for Indian market.
    Supports multiple models, caching, and real-time predictions.
    """

    def __init__(self, model_path: str = None, cache_size: int = 1000):
        """
        Initialize the expense predictor.
        
        Args:
            model_path: Path to trained model file
            cache_size: Maximum number of predictions to cache
        """
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.cache = {}
        self.cache_size = cache_size
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Load model if path provided
        if model_path and os.path.exists(model_path):
            try:
                self.load_model(model_path)
            except Exception as e:
                logger.warning(f"Failed to load model: {e}. Using fallback logic.")
        
        # Enhanced cost-of-living index for Indian cities (2024-2025 data)
        self.city_cost_index = {
            # Tier 1 cities
            "Mumbai": 1.00, "Delhi": 0.95, "Bangalore": 0.92, "Chennai": 0.85,
            "Pune": 0.82, "Hyderabad": 0.78, "Kolkata": 0.75, "Ahmedabad": 0.72,
            
            # Tier 2 cities
            "Jaipur": 0.68, "Lucknow": 0.65, "Kanpur": 0.63, "Nagpur": 0.66,
            "Indore": 0.64, "Thane": 0.88, "Bhopal": 0.62, "Visakhapatnam": 0.60,
            "Pimpri-Chinchwad": 0.80, "Patna": 0.58, "Vadodara": 0.70,
            
            # Tier 3 cities
            "Agra": 0.55, "Meerut": 0.54, "Rajkot": 0.58, "Varanasi": 0.52,
            "Aurangabad": 0.59, "Dhanbad": 0.50, "Amritsar": 0.56,
            "Allahabad": 0.53, "Ranchi": 0.57, "Jodhpur": 0.55,
            
            # Default fallback
            "Other": 0.60
        }
        
        # Inflation rates by category (annual %)
        self.inflation_rates = {
            "Housing": 0.06, "Food & Groceries": 0.08, "Transport": 0.07,
            "Utilities": 0.09, "Entertainment": 0.05, "Self Care": 0.06,
            "Healthcare": 0.07, "Education": 0.06, "Clothing": 0.04,
            "Other": 0.06
        }
        
        # Gender-based spending patterns (research-based multipliers)
        self.gender_spending_patterns = {
            "male": {
                "Housing": 1.0, "Food & Groceries": 0.95, "Transport": 1.1,
                "Utilities": 1.0, "Entertainment": 1.2, "Self Care": 0.7,
                "Healthcare": 0.9, "Education": 1.0, "Clothing": 0.8, "Other": 1.0
            },
            "female": {
                "Housing": 1.0, "Food & Groceries": 1.05, "Transport": 0.9,
                "Utilities": 1.0, "Entertainment": 0.9, "Self Care": 1.4,
                "Healthcare": 1.1, "Education": 1.0, "Clothing": 1.3, "Other": 1.0
            }
        }
        
        # Age-based spending adjustments
        self.age_adjustments = {
            "18-25": {"Entertainment": 1.3, "Self Care": 1.2, "Housing": 0.8},
            "26-35": {"Housing": 1.2, "Healthcare": 0.9, "Entertainment": 1.1},
            "36-45": {"Housing": 1.1, "Healthcare": 1.1, "Education": 1.3},
            "46-55": {"Healthcare": 1.3, "Entertainment": 0.9, "Housing": 1.0},
            "55+": {"Healthcare": 1.5, "Entertainment": 0.7, "Transport": 0.8}
        }

    def _get_cache_key(self, profile: Dict) -> str:
        """Generate cache key for profile."""
        key_items = [
            str(profile.get("income", 0)),
            profile.get("city", ""),
            profile.get("gender", ""),
            str(profile.get("age", 0)),
            profile.get("profession", ""),
            str(profile.get("dependents", 0))
        ]
        return "|".join(key_items)

    def _manage_cache(self, key: str, value: Dict):
        """Manage cache size and store new prediction."""
        if len(self.cache) >= self.cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        self.cache[key] = {
            "data": value,
            "timestamp": datetime.now()
        }

    def _get_age_group(self, age: int) -> str:
        """Get age group for spending adjustments."""
        if age < 26:
            return "18-25"
        elif age < 36:
            return "26-35"
        elif age < 46:
            return "36-45"
        elif age < 56:
            return "46-55"
        else:
            return "55+"

    def _apply_inflation(self, expenses: Dict[str, float], months_ahead: int = 0) -> Dict[str, float]:
        """Apply inflation to expense predictions."""
        if months_ahead == 0:
            return expenses
        
        inflated_expenses = {}
        for category, amount in expenses.items():
            rate = self.inflation_rates.get(category, 0.06)
            monthly_rate = rate / 12
            inflated_amount = amount * ((1 + monthly_rate) ** months_ahead)
            inflated_expenses[category] = round(inflated_amount, 2)
        
        return inflated_expenses

    def predict(self, profile: Dict, months_ahead: int = 0) -> Dict[str, float]:
        """
        Predict monthly expenses for a user profile.
        
        Args:
            profile: User profile with income, city, gender, age, etc.
            months_ahead: Number of months ahead for inflation adjustment
            
        Returns:
            Dictionary with predicted expenses by category
        """
        # Check cache first
        cache_key = self._get_cache_key(profile)
        if cache_key in self.cache:
            cached_data = self.cache[cache_key]
            # Use cache if less than 1 hour old
            if (datetime.now() - cached_data["timestamp"]).seconds < 3600:
                logger.info("Using cached prediction")
                return self._apply_inflation(cached_data["data"], months_ahead)

        try:
            # Use ML model if available and trained
            if self.model and self.is_trained:
                prediction = self._ml_predict(profile)
            else:
                prediction = self._rule_based_predict(profile)
            
            # Apply inflation if needed
            if months_ahead > 0:
                prediction = self._apply_inflation(prediction, months_ahead)
            
            # Cache the result
            self._manage_cache(cache_key, prediction)
            
            logger.info(f"Generated prediction for user in {profile.get('city', 'Unknown')}")
            return prediction
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            # Fallback to basic prediction
            return self._basic_fallback_predict(profile)

    def _ml_predict(self, profile: Dict) -> Dict[str, float]:
        """Make prediction using trained ML model."""
        features = self._extract_features(profile)
        features_scaled = self.scaler.transform([features])
        prediction = self.model.predict(features_scaled)[0]
        
        categories = ["Housing", "Food & Groceries", "Transport", "Utilities", 
                     "Entertainment", "Self Care", "Healthcare", "Education", 
                     "Clothing", "Other"]
        
        return dict(zip(categories, prediction))

    def _extract_features(self, profile: Dict) -> List[float]:
        """Extract features for ML model."""
        income = profile.get("income", 0)
        city_index = self.city_cost_index.get(profile.get("city", "Other"), 0.60)
        gender_encoded = 1 if profile.get("gender", "male").lower() == "male" else 0
        age = profile.get("age", 30)
        dependents = profile.get("dependents", 0)
        profession_encoded = self._encode_profession(profile.get("profession", "Other"))
        
        return [income, city_index, gender_encoded, age, dependents, profession_encoded]

    def _encode_profession(self, profession: str) -> float:
        """Encode profession into numerical value."""
        profession_map = {
            "Software Engineer": 1.2, "Doctor": 1.3, "Lawyer": 1.25,
            "Teacher": 0.9, "Government Employee": 1.0, "Business Owner": 1.4,
            "Consultant": 1.15, "Banker": 1.1, "Other": 1.0
        }
        return profession_map.get(profession, 1.0)

    def _rule_based_predict(self, profile: Dict) -> Dict[str, float]:
        """Enhanced rule-based prediction with multiple factors."""
        income = profile.get("income", 0)
        city = profile.get("city", "Mumbai")
        gender = profile.get("gender", "male").lower()
        age = profile.get("age", 30)
        dependents = profile.get("dependents", 0)
        profession = profile.get("profession", "Other")
        
        # Base calculation with city cost adjustment
        city_multiplier = self.city_cost_index.get(city, 0.60)
        base_amount = income * city_multiplier
        
        # Age group adjustments
        age_group = self._get_age_group(age)
        age_adjustments = self.age_adjustments.get(age_group, {})
        
        # Gender-based spending patterns
        gender_patterns = self.gender_spending_patterns.get(gender, 
                                                          self.gender_spending_patterns["male"])
        
        # Profession adjustment
        profession_multiplier = self._encode_profession(profession)
        
        # Dependents adjustment
        dependents_multiplier = 1 + (dependents * 0.15)
        
        # Base category allocations
        base_allocations = {
            "Housing": 0.30, "Food & Groceries": 0.20, "Transport": 0.12,
            "Utilities": 0.08, "Entertainment": 0.08, "Self Care": 0.05,
            "Healthcare": 0.06, "Education": 0.04, "Clothing": 0.04, "Other": 0.03
        }
        
        expenses = {}
        total_allocated = 0
        
        for category, base_pct in base_allocations.items():
            # Apply all adjustments
            amount = base_amount * base_pct
            amount *= gender_patterns.get(category, 1.0)
            amount *= age_adjustments.get(category, 1.0)
            amount *= profession_multiplier if category in ["Entertainment", "Self Care"] else 1.0
            amount *= dependents_multiplier if category in ["Food & Groceries", "Healthcare", "Education"] else 1.0
            
            expenses[category] = round(amount, 2)
            total_allocated += amount
        
        # Ensure total doesn't exceed 85% of income (leaving room for savings)
        max_expenses = income * 0.85
        if total_allocated > max_expenses:
            adjustment_factor = max_expenses / total_allocated
            expenses = {k: round(v * adjustment_factor, 2) for k, v in expenses.items()}
        
        return expenses

    def _basic_fallback_predict(self, profile: Dict) -> Dict[str, float]:
        """Basic fallback prediction for error cases."""
        income = profile.get("income", 50000)
        return {
            "Housing": round(income * 0.30, 2),
            "Food & Groceries": round(income * 0.20, 2),
            "Transport": round(income * 0.12, 2),
            "Utilities": round(income * 0.08, 2),
            "Entertainment": round(income * 0.08, 2),
            "Self Care": round(income * 0.05, 2),
            "Healthcare": round(income * 0.06, 2),
            "Education": round(income * 0.04, 2),
            "Clothing": round(income * 0.04, 2),
            "Other": round(income * 0.03, 2)
        }

    async def predict_async(self, profile: Dict, months_ahead: int = 0) -> Dict[str, float]:
        """Async version of predict method."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self.predict, profile, months_ahead)

    def batch_predict(self, profiles: List[Dict], months_ahead: int = 0) -> List[Dict[str, float]]:
        """Predict expenses for multiple profiles."""
        predictions = []
        for profile in profiles:
            try:
                prediction = self.predict(profile, months_ahead)
                predictions.append(prediction)
            except Exception as e:
                logger.error(f"Batch prediction failed for profile: {e}")
                predictions.append(self._basic_fallback_predict(profile))
        return predictions

    def train_model(self, training_data: pd.DataFrame, target_columns: List[str]) -> Dict[str, float]:
        """
        Train ML model on historical data.
        
        Args:
            training_data: DataFrame with features and target expenses
            target_columns: List of expense category columns
            
        Returns:
            Training metrics
        """
        try:
            # Extract features
            feature_columns = ["income", "city_index", "gender_encoded", "age", "dependents", "profession_encoded"]
            X = training_data[feature_columns].values
            y = training_data[target_columns].values
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train model
            self.model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            self.model.fit(X_train_scaled, y_train)
            
            # Evaluate
            y_pred = self.model.predict(X_test_scaled)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            self.is_trained = True
            
            logger.info(f"Model trained successfully. MAE: {mae:.2f}, R²: {r2:.3f}")
            
            return {"mae": mae, "r2_score": r2, "features": feature_columns}
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            return {"error": str(e)}

    def save_model(self, filepath: str):
        """Save trained model and scaler."""
        if not self.is_trained:
            raise ValueError("No trained model to save")
        
        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "is_trained": self.is_trained,
            "metadata": {
                "trained_at": datetime.now().isoformat(),
                "city_cost_index": self.city_cost_index,
                "inflation_rates": self.inflation_rates
            }
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """Load trained model and scaler."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data["model"]
        self.scaler = model_data["scaler"]
        self.is_trained = model_data["is_trained"]
        
        # Update indices if available
        metadata = model_data.get("metadata", {})
        if "city_cost_index" in metadata:
            self.city_cost_index.update(metadata["city_cost_index"])
        if "inflation_rates" in metadata:
            self.inflation_rates.update(metadata["inflation_rates"])
        
        logger.info(f"Model loaded from {filepath}")

    def get_spending_insights(self, profile: Dict, actual_expenses: Dict[str, float]) -> List[Dict]:
        """
        Compare predicted vs actual expenses and generate insights.
        
        Args:
            profile: User profile
            actual_expenses: Actual spending by category
            
        Returns:
            List of insights and recommendations
        """
        predictions = self.predict(profile)
        insights = []
        
        for category, predicted in predictions.items():
            actual = actual_expenses.get(category, 0)
            if actual == 0:
                continue
                
            variance = (actual - predicted) / predicted
            
            if variance > 0.2:  # 20% over prediction
                insights.append({
                    "type": "warning",
                    "category": category,
                    "message": f"You spent {variance*100:.1f}% more than expected on {category}",
                    "predicted": predicted,
                    "actual": actual,
                    "recommendation": self._get_category_recommendation(category, "overspend")
                })
            elif variance < -0.1:  # 10% under prediction
                insights.append({
                    "type": "positive",
                    "category": category,
                    "message": f"Great! You saved {abs(variance)*100:.1f}% on {category}",
                    "predicted": predicted,
                    "actual": actual,
                    "recommendation": self._get_category_recommendation(category, "underspend")
                })
        
        return insights

    def _get_category_recommendation(self, category: str, spend_type: str) -> str:
        """Get recommendations based on spending patterns."""
        recommendations = {
            "Housing": {
                "overspend": "Consider refinancing or finding a roommate to reduce costs",
                "underspend": "You might want to invest the savings or upgrade your living situation"
            },
            "Food & Groceries": {
                "overspend": "Try meal planning and bulk buying to reduce grocery costs",
                "underspend": "Great budgeting! Consider investing the savings"
            },
            "Transport": {
                "overspend": "Consider public transport or carpooling to reduce costs",
                "underspend": "Excellent! You're managing transport costs well"
            },
            "Entertainment": {
                "overspend": "Look for free or low-cost entertainment options",
                "underspend": "You have room to enjoy more activities within budget"
            }
        }
        
        default_rec = {
            "overspend": "Consider reducing expenses in this category",
            "underspend": "Good job managing this expense category"
        }
        
        return recommendations.get(category, default_rec)[spend_type]

# Example usage and testing
if __name__ == "__main__":
    predictor = ExpensePredictor()
    
    # Test prediction
    test_profile = {
        "income": 80000,
        "city": "Bangalore",
        "gender": "female",
        "age": 28,
        "dependents": 1,
        "profession": "Software Engineer"
    }
    
    prediction = predictor.predict(test_profile)
    print("Expense Prediction:")
    for category, amount in prediction.items():
        print(f"{category}: ₹{amount:,.2f}")
    
    # Test with inflation
    future_prediction = predictor.predict(test_profile, months_ahead=12)
    print("\nPrediction with 12-month inflation:")
    for category, amount in future_prediction.items():
        print(f"{category}: ₹{amount:,.2f}")
