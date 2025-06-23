# backend/app.py

import logging
import os
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, status, BackgroundTasks, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from sqlalchemy import and_, func
import pandas as pd
import jwt
from passlib.context import CryptContext
import uvicorn

# Local imports
from backend.models.database import engine, get_db, Base
from backend.models.user import User, UserCreate, UserOut, create_user, get_user, get_user_by_email
from backend.models.transaction import Transaction, TransactionCreate, TransactionOut, create_transaction, get_transactions_by_user
from backend.models.investment import Investment, InvestmentCreate, InvestmentOut, create_investment, get_investments_by_user
from backend.models.prediction import ExpensePredictionIn, ExpensePredictionOut, predict_expenses

from backend.ml_engine import (
    ExpensePredictor, 
    TransactionCategorizer, 
    InsightsGenerator, 
    MarketAnalyzer, 
    TimeSeriesForecaster
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Security configuration
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

# Initialize ML models
expense_predictor = ExpensePredictor()
transaction_categorizer = TransactionCategorizer()
insights_generator = InsightsGenerator()
market_analyzer = MarketAnalyzer()
time_series_forecaster = TimeSeriesForecaster()

# Database initialization
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database and ML models on startup"""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
        
        # Initialize ML models in background
        logger.info("ML models initialized successfully")
        yield
    except Exception as e:
        logger.error(f"Startup error: {e}")
        raise
    finally:
        logger.info("Application shutdown")

# FastAPI app initialization
app = FastAPI(
    title="Personal Finance AI Assistant",
    description="AI-powered personal finance management for Indian users",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["localhost", "127.0.0.1", "*.yourdomain.com"]
)

# Authentication utilities
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify JWT token and extract user info"""
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: int = payload.get("sub")
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials"
            )
        return user_id
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials"
        )

# Exception handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Global exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }

# Authentication endpoints
@app.post("/auth/register", response_model=UserOut)
async def register_user(user: UserCreate, db: Session = Depends(get_db)):
    """Register new user"""
    try:
        # Check if user exists
        db_user = get_user_by_email(db, email=user.email)
        if db_user:
            raise HTTPException(
                status_code=400,
                detail="Email already registered"
            )
        
        # Create user
        created_user = create_user(db=db, user_in=user)
        logger.info(f"User registered: {created_user.email}")
        return created_user
    
    except Exception as e:
        logger.error(f"Registration error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/auth/login")
async def login_user(email: str, db: Session = Depends(get_db)):
    """Login user (simplified for demo - add password validation in production)"""
    try:
        user = get_user_by_email(db, email=email)
        if not user:
            raise HTTPException(
                status_code=401,
                detail="Invalid credentials"
            )
        
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": str(user.id)}, 
            expires_delta=access_token_expires
        )
        
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "user": UserOut.from_orm(user)
        }
    
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(status_code=401, detail="Invalid credentials")

# User endpoints
@app.get("/users/me", response_model=UserOut)
async def get_current_user(
    current_user_id: int = Depends(verify_token),
    db: Session = Depends(get_db)
):
    """Get current user profile"""
    user = get_user(db, user_id=current_user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

@app.put("/users/me", response_model=UserOut)
async def update_user_profile(
    user_update: UserCreate,
    current_user_id: int = Depends(verify_token),
    db: Session = Depends(get_db)
):
    """Update user profile"""
    try:
        user = get_user(db, user_id=current_user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Update user fields
        for field, value in user_update.dict().items():
            if field != "email":  # Don't allow email changes
                setattr(user, field, value)
        
        db.commit()
        db.refresh(user)
        return user
    
    except Exception as e:
        logger.error(f"Profile update error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

# Transaction endpoints
@app.post("/transactions", response_model=TransactionOut)
async def create_new_transaction(
    transaction: TransactionCreate,
    current_user_id: int = Depends(verify_token),
    db: Session = Depends(get_db)
):
    """Create new transaction"""
    try:
        # Auto-categorize if category is not provided
        if not transaction.category or transaction.category == "Other":
            transaction.category = transaction_categorizer.categorize(transaction.description)
        
        transaction.user_id = current_user_id
        created_transaction = create_transaction(db=db, txn_in=transaction)
        logger.info(f"Transaction created: {created_transaction.id}")
        return created_transaction
    
    except Exception as e:
        logger.error(f"Transaction creation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/transactions", response_model=List[TransactionOut])
async def get_user_transactions(
    skip: int = 0,
    limit: int = 100,
    category: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    current_user_id: int = Depends(verify_token),
    db: Session = Depends(get_db)
):
    """Get user transactions with filtering"""
    try:
        query = db.query(Transaction).filter(Transaction.user_id == current_user_id)
        
        if category:
            query = query.filter(Transaction.category == category)
        
        if start_date:
            query = query.filter(Transaction.date >= start_date)
        
        if end_date:
            query = query.filter(Transaction.date <= end_date)
        
        transactions = query.order_by(Transaction.date.desc()).offset(skip).limit(limit).all()
        return transactions
    
    except Exception as e:
        logger.error(f"Transaction fetch error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/transactions/bulk-upload")
async def bulk_upload_transactions(
    file: UploadFile = File(...),
    current_user_id: int = Depends(verify_token),
    db: Session = Depends(get_db),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Upload transactions from CSV/Excel file"""
    try:
        if not file.filename.endswith(('.csv', '.xlsx', '.xls')):
            raise HTTPException(status_code=400, detail="Only CSV and Excel files are supported")
        
        # Read file content
        contents = await file.read()
        
        # Process in background
        background_tasks.add_task(
            process_transaction_file,
            contents,
            file.filename,
            current_user_id,
            db
        )
        
        return {"message": "File uploaded successfully. Processing in background."}
    
    except Exception as e:
        logger.error(f"Bulk upload error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

async def process_transaction_file(contents: bytes, filename: str, user_id: int, db: Session):
    """Background task to process uploaded transaction file"""
    try:
        # Parse file based on extension
        if filename.endswith('.csv'):
            df = pd.read_csv(pd.io.common.BytesIO(contents))
        else:
            df = pd.read_excel(pd.io.common.BytesIO(contents))
        
        # Expected columns: date, description, amount
        required_columns = ['date', 'description', 'amount']
        if not all(col in df.columns for col in required_columns):
            logger.error(f"Missing required columns: {required_columns}")
            return
        
        # Process transactions
        for _, row in df.iterrows():
            try:
                category = transaction_categorizer.categorize(row['description'])
                transaction = TransactionCreate(
                    user_id=user_id,
                    date=pd.to_datetime(row['date']).date(),
                    description=row['description'],
                    category=category,
                    amount=float(row['amount'])
                )
                create_transaction(db=db, txn_in=transaction)
            except Exception as e:
                logger.error(f"Error processing transaction row: {e}")
                continue
        
        logger.info(f"Processed {len(df)} transactions for user {user_id}")
    
    except Exception as e:
        logger.error(f"File processing error: {e}")

# Expense prediction endpoints
@app.post("/predictions/expenses", response_model=ExpensePredictionOut)
async def predict_user_expenses(
    prediction_input: ExpensePredictionIn,
    current_user_id: int = Depends(verify_token)
):
    """Predict user expenses based on profile"""
    try:
        prediction = expense_predictor.predict(prediction_input.dict())
        return ExpensePredictionOut(predicted_expenses=prediction)
    
    except Exception as e:
        logger.error(f"Expense prediction error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/predictions/expenses/historical")
async def get_expense_forecast(
    months: int = 6,
    current_user_id: int = Depends(verify_token),
    db: Session = Depends(get_db)
):
    """Get expense forecast using time series analysis"""
    try:
        # Get historical transactions
        transactions = get_transactions_by_user(db, user_id=current_user_id)
        
        if not transactions:
            raise HTTPException(status_code=400, detail="No historical data available")
        
        # Group by month and sum expenses
        df = pd.DataFrame([{
            'date': t.date,
            'amount': abs(t.amount) if t.amount < 0 else 0  # Only expenses
        } for t in transactions])
        
        df['date'] = pd.to_datetime(df['date'])
        monthly_expenses = df.groupby(df['date'].dt.to_period('M'))['amount'].sum().reset_index()
        monthly_expenses['date'] = monthly_expenses['date'].dt.to_timestamp()
        
        # Prepare data for forecasting
        history = [
            {"date": row['date'].strftime('%Y-%m-%d'), "value": row['amount']}
            for _, row in monthly_expenses.iterrows()
        ]
        
        # Generate forecast
        if len(history) >= 3:  # Need minimum data points
            forecast = time_series_forecaster.fit_predict(history, periods=months)
            return {"forecast": forecast, "historical": history}
        else:
            raise HTTPException(status_code=400, detail="Insufficient historical data for forecasting")
    
    except Exception as e:
        logger.error(f"Forecast error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

# Analytics and insights endpoints
@app.get("/analytics/insights")
async def get_financial_insights(
    current_user_id: int = Depends(verify_token),
    db: Session = Depends(get_db)
):
    """Get AI-generated financial insights"""
    try:
        user = get_user(db, user_id=current_user_id)
        transactions = get_transactions_by_user(db, user_id=current_user_id)
        
        if not user or not transactions:
            return {"insights": []}
        
        # Convert to dict format for insights generator
        user_data = {
            "income": user.income,
            "city": user.city,
            "gender": user.gender,
            "age": user.age
        }
        
        transaction_data = [
            {
                "date": t.date.isoformat(),
                "category": t.category,
                "amount": t.amount,
                "description": t.description
            }
            for t in transactions
        ]
        
        insights = insights_generator.generate(user_data, transaction_data)
        return {"insights": insights}
    
    except Exception as e:
        logger.error(f"Insights generation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/analytics/spending-breakdown")
async def get_spending_breakdown(
    period: str = "month",  # month, quarter, year
    current_user_id: int = Depends(verify_token),
    db: Session = Depends(get_db)
):
    """Get spending breakdown by category"""
    try:
        # Calculate date range based on period
        end_date = datetime.now().date()
        if period == "month":
            start_date = end_date.replace(day=1)
        elif period == "quarter":
            quarter_start = ((end_date.month - 1) // 3) * 3 + 1
            start_date = end_date.replace(month=quarter_start, day=1)
        else:  # year
            start_date = end_date.replace(month=1, day=1)
        
        # Query transactions
        transactions = db.query(Transaction).filter(
            and_(
                Transaction.user_id == current_user_id,
                Transaction.date >= start_date,
                Transaction.date <= end_date,
                Transaction.amount < 0  # Only expenses
            )
        ).all()
        
        # Group by category
        category_totals = {}
        for t in transactions:
            category = t.category
            amount = abs(t.amount)
            category_totals[category] = category_totals.get(category, 0) + amount
        
        # Calculate percentages
        total_spending = sum(category_totals.values())
        breakdown = [
            {
                "category": category,
                "amount": amount,
                "percentage": round((amount / total_spending) * 100, 2) if total_spending > 0 else 0
            }
            for category, amount in category_totals.items()
        ]
        
        return {
            "period": period,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "total_spending": total_spending,
            "breakdown": sorted(breakdown, key=lambda x: x["amount"], reverse=True)
        }
    
    except Exception as e:
        logger.error(f"Spending breakdown error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

# Investment endpoints
@app.post("/investments", response_model=InvestmentOut)
async def create_new_investment(
    investment: InvestmentCreate,
    current_user_id: int = Depends(verify_token),
    db: Session = Depends(get_db)
):
    """Create new investment record"""
    try:
        investment.user_id = current_user_id
        created_investment = create_investment(db=db, inv_in=investment)
        logger.info(f"Investment created: {created_investment.id}")
        return created_investment
    
    except Exception as e:
        logger.error(f"Investment creation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/investments", response_model=List[InvestmentOut])
async def get_user_investments(
    current_user_id: int = Depends(verify_token),
    db: Session = Depends(get_db)
):
    """Get user investments"""
    try:
        investments = get_investments_by_user(db, user_id=current_user_id)
        return investments
    
    except Exception as e:
        logger.error(f"Investment fetch error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

# Market analysis endpoints
@app.get("/market/stock/{symbol}")
async def get_stock_info(symbol: str):
    """Get stock information"""
    try:
        # Add .NS suffix for NSE stocks if not present
        if not symbol.endswith(('.NS', '.BO')):
            symbol += '.NS'
        
        stock_info = market_analyzer.get_stock_summary(symbol)
        return stock_info
    
    except Exception as e:
        logger.error(f"Stock info error: {e}")
        raise HTTPException(status_code=400, detail=f"Error fetching stock info: {str(e)}")

@app.get("/market/stock/{symbol}/history")
async def get_stock_history(symbol: str, period: str = "1mo"):
    """Get stock price history"""
    try:
        # Add .NS suffix for NSE stocks if not present
        if not symbol.endswith(('.NS', '.BO')):
            symbol += '.NS'
        
        history = market_analyzer.get_historical_prices(symbol, period)
        return {"symbol": symbol, "period": period, "history": history}
    
    except Exception as e:
        logger.error(f"Stock history error: {e}")
        raise HTTPException(status_code=400, detail=f"Error fetching stock history: {str(e)}")

@app.get("/market/recommendations")
async def get_investment_recommendations(
    current_user_id: int = Depends(verify_token),
    db: Session = Depends(get_db)
):
    """Get personalized investment recommendations"""
    try:
        user = get_user(db, user_id=current_user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Simple recommendation logic based on user profile
        recommendations = []
        
        # Age-based recommendations
        if user.age < 30:
            recommendations.append({
                "type": "Equity Mutual Fund",
                "name": "Large Cap Growth Fund",
                "reason": "Young investors can take higher risk for better returns",
                "allocation": "60-70%"
            })
        elif user.age < 50:
            recommendations.append({
                "type": "Balanced Fund",
                "name": "Hybrid Equity Fund",
                "reason": "Balanced approach for middle-aged investors",
                "allocation": "50-60%"
            })
        else:
            recommendations.append({
                "type": "Debt Fund",
                "name": "Corporate Bond Fund",
                "reason": "Lower risk investments for pre-retirement",
                "allocation": "60-70%"
            })
        
        # Income-based recommendations
        if user.income > 100000:  # High income
            recommendations.append({
                "type": "ELSS",
                "name": "Tax Saving Mutual Fund",
                "reason": "Tax benefits with potential for good returns",
                "allocation": "10-15%"
            })
        
        # City-based recommendations (metro vs non-metro)
        metro_cities = ["Mumbai", "Delhi", "Bangalore", "Chennai", "Hyderabad", "Pune"]
        if user.city in metro_cities:
            recommendations.append({
                "type": "Real Estate",
                "name": "REITs",
                "reason": "Real estate exposure without direct property investment",
                "allocation": "5-10%"
            })
        
        return {"recommendations": recommendations}
    
    except Exception as e:
        logger.error(f"Recommendation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

# Utility endpoints
@app.post("/utils/categorize-transaction")
async def categorize_transaction_text(
    description: str,
    current_user_id: int = Depends(verify_token)
):
    """Categorize transaction description"""
    try:
        category = transaction_categorizer.categorize(description)
        return {"description": description, "category": category}
    
    except Exception as e:
        logger.error(f"Categorization error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/utils/cities")
async def get_supported_cities():
    """Get list of supported Indian cities"""
    cities = [
        "Mumbai", "Delhi", "Bangalore", "Chennai", "Pune", "Hyderabad", 
        "Kolkata", "Ahmedabad", "Jaipur", "Lucknow", "Indore", "Bhopal",
        "Nagpur", "Visakhapatnam", "Ludhiana", "Agra", "Nashik", "Vadodara",
        "Rajkot", "Varanasi", "Srinagar", "Aurangabad", "Dhanbad", "Amritsar",
        "Allahabad", "Ranchi", "Howrah", "Coimbatore", "Jabalpur", "Gwalior"
    ]
    return {"cities": sorted(cities)}

# Development endpoints (remove in production)
@app.get("/dev/seed-data")
async def seed_test_data(
    current_user_id: int = Depends(verify_token),
    db: Session = Depends(get_db)
):
    """Seed test data for development"""
    try:
        # Create sample transactions
        sample_transactions = [
            {"description": "Zomato food order", "amount": -450.0, "category": "Food & Groceries"},
            {"description": "Uber ride", "amount": -120.0, "category": "Transport"},
            {"description": "Electricity bill", "amount": -2500.0, "category": "Utilities"},
            {"description": "Salary credit", "amount": 50000.0, "category": "Income"},
            {"description": "Movie ticket", "amount": -300.0, "category": "Entertainment"},
            {"description": "Salon visit", "amount": -800.0, "category": "Self Care"},
        ]
        
        for txn_data in sample_transactions:
            transaction = TransactionCreate(
                user_id=current_user_id,
                date=datetime.now().date(),
                description=txn_data["description"],
                category=txn_data["category"],
                amount=txn_data["amount"]
            )
            create_transaction(db=db, txn_in=transaction)
        
        return {"message": f"Created {len(sample_transactions)} sample transactions"}
    
    except Exception as e:
        logger.error(f"Seed data error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )


