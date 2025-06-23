# backend/api/insights.py

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from typing import List

from backend.models import transaction, user
from backend.models.database import get_db
from backend.api.auth import get_current_user
from backend.ml_engine.insights_generator import InsightsGenerator

router = APIRouter(prefix="/api/insights", tags=["insights"])
insights_engine = InsightsGenerator()

@router.get("/", response_model=List[dict])
def get_insights(
    db: Session = Depends(get_db),
    current_user: user.User = Depends(get_current_user),
):
    txns = transaction.get_transactions_by_user(db, current_user.id)
    txns_dicts = [
        {
            "amount": t.amount,
            "category": t.category,
            "date": t.date.isoformat(),
            "description": t.description,
        }
        for t in txns
    ]
    user_data = {
        "income": current_user.income,
        "avg_transport": 4000,  # Example; compute real averages from user's data
    }
    insights = insights_engine.generate(user_data, txns_dicts)
    return insights
