# backend/models/investment.py

from sqlalchemy import Column, Integer, Float, String, ForeignKey, Date
from sqlalchemy.orm import Session
from pydantic import BaseModel
from datetime import date
from backend.models.database import Base

class Investment(Base):
    __tablename__ = "investments"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    date = Column(Date, nullable=False)
    investment_type = Column(String(32), nullable=False)  # e.g., Mutual Fund, Stock, FD
    name = Column(String(64), nullable=False)
    amount = Column(Float, nullable=False)
    current_value = Column(Float, nullable=True)

# Pydantic schemas
class InvestmentBase(BaseModel):
    user_id: int
    date: date
    investment_type: str
    name: str
    amount: float
    current_value: float | None = None

class InvestmentCreate(InvestmentBase):
    pass

class InvestmentOut(InvestmentBase):
    id: int
    class Config:
        orm_mode = True

# CRUD functions
def create_investment(db: Session, inv_in: InvestmentCreate):
    db_inv = Investment(**inv_in.dict())
    db.add(db_inv)
    db.commit()
    db.refresh(db_inv)
    return db_inv

def get_investments_by_user(db: Session, user_id: int):
    return db.query(Investment).filter(Investment.user_id == user_id).order_by(Investment.date.desc()).all()
