# backend/models/transaction.py

from sqlalchemy import Column, Integer, Float, String, ForeignKey, Date
from sqlalchemy.orm import Session, relationship
from pydantic import BaseModel
from datetime import date
from backend.models.database import Base

class Transaction(Base):
    __tablename__ = "transactions"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    date = Column(Date, nullable=False)
    description = Column(String(128), nullable=False)
    category = Column(String(32), nullable=False)
    amount = Column(Float, nullable=False)

# Pydantic schemas
class TransactionBase(BaseModel):
    user_id: int
    date: date
    description: str
    category: str
    amount: float

class TransactionCreate(TransactionBase):
    pass

class TransactionOut(TransactionBase):
    id: int
    class Config:
        orm_mode = True

# CRUD functions
def create_transaction(db: Session, txn_in: TransactionCreate):
    db_txn = Transaction(**txn_in.dict())
    db.add(db_txn)
    db.commit()
    db.refresh(db_txn)
    return db_txn

def get_transactions_by_user(db: Session, user_id: int):
    return db.query(Transaction).filter(Transaction.user_id == user_id).order_by(Transaction.date.desc()).all()
