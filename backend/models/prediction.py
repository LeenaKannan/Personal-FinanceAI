# backend/models/prediction.py

from pydantic import BaseModel
from typing import Dict

# Input schema for expense prediction
class ExpensePredictionIn(BaseModel):
    income: float
    city: str
    gender: str
    age: int

# Output schema for expense prediction
class ExpensePredictionOut(BaseModel):
    predicted_expenses: Dict[str, float]

# Dummy ML logic (replace with actual model inference)
def predict_expenses(pred_in: ExpensePredictionIn) -> ExpensePredictionOut:
    city_cost_index = {
        "Mumbai": 1.0, "Delhi": 0.95, "Bangalore": 0.88, "Chennai": 0.82,
        "Pune": 0.78, "Hyderabad": 0.73, "Kolkata": 0.75, "Ahmedabad": 0.70,
        "Jaipur": 0.65, "Lucknow": 0.60
    }
    base = pred_in.income * city_cost_index.get(pred_in.city, 1.0)
    gender_adj = 0.97 if pred_in.gender.lower() == "female" else 1.0
    age_adj = 0.9 if pred_in.age < 25 else (1.1 if pred_in.age > 40 else 1.0)
    expenses = {
        "Housing": round(base * 0.35 * gender_adj * age_adj, 2),
        "Food & Groceries": round(base * 0.20 * gender_adj, 2),
        "Transport": round(base * 0.12, 2),
        "Utilities": round(base * 0.08, 2),
        "Entertainment": round(base * 0.07, 2),
        "Self Care": round(base * 0.05, 2),
        "Other": round(base * 0.13, 2),
    }
    return ExpensePredictionOut(predicted_expenses=expenses)
