# Expense prediction models
# backend/ml_engine/expense_predictor.py

import numpy as np
import joblib
from typing import Dict

class ExpensePredictor:
    """
    Predicts user expenses using a trained ML model.
    """

    def __init__(self, model_path: str = None):
        # Load a trained model if path is given, else use a mock model.
        if model_path:
            self.model = joblib.load(model_path)
        else:
            self.model = None  # Use rule-based fallback

        # Example cost-of-living index for Indian cities
        self.city_cost_index = {
            "Mumbai": 1.0, "Delhi": 0.95, "Bangalore": 0.88, "Chennai": 0.82,
            "Pune": 0.78, "Hyderabad": 0.73, "Kolkata": 0.75, "Ahmedabad": 0.70,
            "Jaipur": 0.65, "Lucknow": 0.60
        }

    def predict(self, profile: Dict) -> Dict[str, float]:
        """
        Predict monthly expenses by category for a user profile.
        profile: dict with keys income, city, gender, age
        """
        # If a real model is loaded, use it
        if self.model:
            features = np.array([
                profile.get("income", 0),
                self.city_cost_index.get(profile.get("city", "Mumbai"), 1.0),
                1 if profile.get("gender", "male") == "male" else 0,
                profile.get("age", 30)
            ]).reshape(1, -1)
            pred = self.model.predict(features)
            return dict(zip(["Housing", "Food & Groceries", "Transport", "Utilities", "Entertainment", "Self Care", "Other"], pred[0]))

        # Fallback: rule-based logic
        income = profile.get("income", 0)
        city = profile.get("city", "Mumbai")
        gender = profile.get("gender", "male")
        age = profile.get("age", 30)
        base = income * self.city_cost_index.get(city, 1.0)
        gender_adj = 0.97 if gender.lower() == "female" else 1.0
        age_adj = 0.9 if age < 25 else (1.1 if age > 40 else 1.0)
        return {
            "Housing": round(base * 0.35 * gender_adj * age_adj, 2),
            "Food & Groceries": round(base * 0.20 * gender_adj, 2),
            "Transport": round(base * 0.12, 2),
            "Utilities": round(base * 0.08, 2),
            "Entertainment": round(base * 0.07, 2),
            "Self Care": round(base * 0.05, 2),
            "Other": round(base * 0.13, 2),
        }
