# backend/app.py

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from backend.models import user, transaction, prediction, investment
from backend.models.database import Base, engine, get_db

# Create all tables
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Personal Finance AI Backend",
    description="AI-powered backend for personal finance management",
    version="1.0.0"
)
from backend.api import auth, transactions, predictions, insights

app.include_router(auth.router)
app.include_router(transactions.router)
app.include_router(predictions.router)
app.include_router(insights.router)

# CORS middleware for frontend-backend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, restrict this!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- User Endpoints ---
@app.post("/users/", response_model=user.UserOut)
def create_user(user_in: user.UserCreate, db: Session = Depends(get_db)):
    db_user = user.get_user_by_email(db, email=user_in.email)
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    return user.create_user(db, user_in)

@app.get("/users/{user_id}", response_model=user.UserOut)
def get_user(user_id: int, db: Session = Depends(get_db)):
    db_user = user.get_user(db, user_id)
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")
    return db_user

# --- Transaction Endpoints ---
@app.post("/transactions/", response_model=transaction.TransactionOut)
def add_transaction(txn_in: transaction.TransactionCreate, db: Session = Depends(get_db)):
    return transaction.create_transaction(db, txn_in)

@app.get("/transactions/{user_id}", response_model=list[transaction.TransactionOut])
def get_transactions(user_id: int, db: Session = Depends(get_db)):
    return transaction.get_transactions_by_user(db, user_id)

# --- Prediction Endpoints ---
@app.post("/predict/expenses", response_model=prediction.ExpensePredictionOut)
def predict_expenses(pred_in: prediction.ExpensePredictionIn):
    # Dummy logic; replace with ML model inference
    return prediction.predict_expenses(pred_in)

# --- Investment Endpoints ---
@app.post("/investments/", response_model=investment.InvestmentOut)
def add_investment(inv_in: investment.InvestmentCreate, db: Session = Depends(get_db)):
    return investment.create_investment(db, inv_in)

@app.get("/investments/{user_id}", response_model=list[investment.InvestmentOut])
def get_investments(user_id: int, db: Session = Depends(get_db)):
    return investment.get_investments_by_user(db, user_id)

# Health check
@app.get("/")
def root():
    return {"status": "ok", "message": "Personal Finance AI Backend is running!"}


