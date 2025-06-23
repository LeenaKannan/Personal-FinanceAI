# backend/api/predictions.py

from fastapi import APIRouter, Depends, HTTPException
from backend.ml_engine.expense_predictor import ExpensePredictor
from backend.models.prediction import ExpensePredictionIn, ExpensePredictionOut
from backend.api.auth import get_current_user
from backend.models import user

router = APIRouter(prefix="/api/predictions", tags=["predictions"])
predictor = ExpensePredictor()

@router.post("/expenses", response_model=ExpensePredictionOut)
def predict_expenses(
    payload: ExpensePredictionIn,
    current_user: user.User = Depends(get_current_user),
):
    # Optionally, use current_user info if not provided in payload
    profile = {
        "income": payload.income or current_user.income,
        "city": payload.city or current_user.city,
        "gender": payload.gender or current_user.gender,
        "age": payload.age or current_user.age,
    }
    result = predictor.predict(profile)
    return ExpensePredictionOut(predicted_expenses=result)
