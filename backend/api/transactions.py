# backend/api/transactions.py

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List

from backend.models import transaction, user
from backend.models.database import get_db
from backend.api.auth import get_current_user
from backend.ml_engine.categorizer import TransactionCategorizer

router = APIRouter(prefix="/api/transactions", tags=["transactions"])
categorizer = TransactionCategorizer()

@router.post("/", response_model=transaction.TransactionOut)
def add_transaction(
    txn_in: transaction.TransactionCreate,
    db: Session = Depends(get_db),
    current_user: user.User = Depends(get_current_user),
):
    # Auto-categorize if not provided
    if not txn_in.category or txn_in.category == "Other":
        txn_in.category = categorizer.categorize(txn_in.description)
    txn_in.user_id = current_user.id
    return transaction.create_transaction(db, txn_in)

@router.get("/", response_model=List[transaction.TransactionOut])
def list_transactions(
    db: Session = Depends(get_db),
    current_user: user.User = Depends(get_current_user),
):
    return transaction.get_transactions_by_user(db, current_user.id)

@router.get("/{txn_id}", response_model=transaction.TransactionOut)
def get_transaction(
    txn_id: int,
    db: Session = Depends(get_db),
    current_user: user.User = Depends(get_current_user),
):
    txn = db.query(transaction.Transaction).filter(transaction.Transaction.id == txn_id).first()
    if not txn or txn.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Transaction not found")
    return txn

@router.delete("/{txn_id}")
def delete_transaction(
    txn_id: int,
    db: Session = Depends(get_db),
    current_user: user.User = Depends(get_current_user),
):
    txn = db.query(transaction.Transaction).filter(transaction.Transaction.id == txn_id).first()
    if not txn or txn.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Transaction not found")
    db.delete(txn)
    db.commit()
    return {"status": "deleted"}
