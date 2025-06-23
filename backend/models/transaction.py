# backend/models/transaction.py

from sqlalchemy import Column, Integer, Float, String, ForeignKey, Date, DateTime, Boolean, Text, Enum as SQLEnum
from sqlalchemy.orm import Session, relationship
from sqlalchemy.sql import func
from pydantic import BaseModel, Field, validator
from datetime import date, datetime
from typing import Optional, List, Dict, Any
from enum import Enum
from decimal import Decimal
import logging

from backend.models.database import Base

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enums for transaction types and categories
class TransactionTypeEnum(str, Enum):
    DEBIT = "debit"
    CREDIT = "credit"

class TransactionStatusEnum(str, Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class PaymentMethodEnum(str, Enum):
    CASH = "cash"
    UPI = "upi"
    CARD = "card"
    NET_BANKING = "net_banking"
    CHEQUE = "cheque"
    WALLET = "wallet"
    BANK_TRANSFER = "bank_transfer"

class CategoryEnum(str, Enum):
    HOUSING = "Housing"
    FOOD_GROCERIES = "Food & Groceries"
    TRANSPORT = "Transport"
    UTILITIES = "Utilities"
    ENTERTAINMENT = "Entertainment"
    SELF_CARE = "Self Care"
    HEALTHCARE = "Healthcare"
    EDUCATION = "Education"
    SHOPPING = "Shopping"
    INVESTMENT = "Investment"
    INSURANCE = "Insurance"
    EMI = "EMI"
    INCOME = "Income"
    TRANSFER = "Transfer"
    OTHER = "Other"

# === SQLAlchemy Model ===

class Transaction(Base):
    """
    Comprehensive transaction model for Indian financial data.
    Supports UPI, bank transfers, and various Indian payment methods.
    """
    __tablename__ = "transactions"
    
    # Primary fields
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    
    # Transaction details
    date = Column(Date, nullable=False, index=True)
    description = Column(String(200), nullable=False)
    category = Column(SQLEnum(CategoryEnum), nullable=False, default=CategoryEnum.OTHER)
    amount = Column(Float, nullable=False)  # Positive for credit, negative for debit
    
    # Enhanced fields
    transaction_type = Column(SQLEnum(TransactionTypeEnum), nullable=False)
    payment_method = Column(SQLEnum(PaymentMethodEnum), nullable=True)
    status = Column(SQLEnum(TransactionStatusEnum), nullable=False, default=TransactionStatusEnum.COMPLETED)
    
    # Indian-specific fields
    upi_id = Column(String(100), nullable=True)  # UPI transaction ID
    reference_number = Column(String(50), nullable=True, unique=True)  # Bank reference
    merchant_name = Column(String(100), nullable=True)
    merchant_category = Column(String(50), nullable=True)
    
    # Location and context
    location = Column(String(100), nullable=True)
    city = Column(String(50), nullable=True)
    
    # Financial details
    balance_after = Column(Float, nullable=True)  # Account balance after transaction
    fees = Column(Float, nullable=True, default=0.0)  # Transaction fees
    tax_amount = Column(Float, nullable=True, default=0.0)  # GST or other taxes
    
    # Metadata
    notes = Column(Text, nullable=True)  # User notes
    tags = Column(String(200), nullable=True)  # Comma-separated tags
    is_recurring = Column(Boolean, default=False)
    is_business = Column(Boolean, default=False)
    is_verified = Column(Boolean, default=True)
    
    # Audit fields
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    user = relationship("User", back_populates="transactions")
    
    def __repr__(self):
        return f"<Transaction(id={self.id}, amount={self.amount}, category={self.category})>"

# === Pydantic Schemas ===

class TransactionBase(BaseModel):
    """Base transaction schema with validation."""
    
    user_id: int = Field(..., gt=0, description="User ID")
    date: date = Field(..., description="Transaction date")
    description: str = Field(..., min_length=1, max_length=200, description="Transaction description")
    category: CategoryEnum = Field(default=CategoryEnum.OTHER, description="Transaction category")
    amount: float = Field(..., description="Transaction amount (positive for credit, negative for debit)")
    
    # Optional enhanced fields
    payment_method: Optional[PaymentMethodEnum] = Field(None, description="Payment method used")
    upi_id: Optional[str] = Field(None, max_length=100, description="UPI transaction ID")
    reference_number: Optional[str] = Field(None, max_length=50, description="Bank reference number")
    merchant_name: Optional[str] = Field(None, max_length=100, description="Merchant name")
    location: Optional[str] = Field(None, max_length=100, description="Transaction location")
    city: Optional[str] = Field(None, max_length=50, description="City where transaction occurred")
    fees: Optional[float] = Field(0.0, ge=0, description="Transaction fees")
    tax_amount: Optional[float] = Field(0.0, ge=0, description="Tax amount")
    notes: Optional[str] = Field(None, max_length=500, description="User notes")
    tags: Optional[str] = Field(None, max_length=200, description="Comma-separated tags")
    is_recurring: Optional[bool] = Field(False, description="Is this a recurring transaction")
    is_business: Optional[bool] = Field(False, description="Is this a business transaction")
    
    @validator('date')
    def validate_date(cls, v):
        """Validate transaction date."""
        if v > date.today():
            raise ValueError("Transaction date cannot be in the future")
        return v
    
    @validator('amount')
    def validate_amount(cls, v):
        """Validate transaction amount."""
        if v == 0:
            raise ValueError("Transaction amount cannot be zero")
        if abs(v) > 10000000:  # 1 crore limit
            raise ValueError("Transaction amount exceeds maximum limit")
        return v
    
    @validator('description')
    def validate_description(cls, v):
        """Clean and validate description."""
        if not v or not v.strip():
            raise ValueError("Description cannot be empty")
        return v.strip()
    
    @validator('upi_id')
    def validate_upi_id(cls, v):
        """Validate UPI ID format."""
        if v and not v.isalnum():
            # Basic UPI ID validation
            if '@' not in v:
                raise ValueError("Invalid UPI ID format")
        return v

class TransactionCreate(TransactionBase):
    """Schema for creating new transactions."""
    pass

class TransactionUpdate(BaseModel):
    """Schema for updating existing transactions."""
    
    description: Optional[str] = Field(None, min_length=1, max_length=200)
    category: Optional[CategoryEnum] = None
    payment_method: Optional[PaymentMethodEnum] = None
    merchant_name: Optional[str] = Field(None, max_length=100)
    location: Optional[str] = Field(None, max_length=100)
    city: Optional[str] = Field(None, max_length=50)
    notes: Optional[str] = Field(None, max_length=500)
    tags: Optional[str] = Field(None, max_length=200)
    is_recurring: Optional[bool] = None
    is_business: Optional[bool] = None

class TransactionOut(TransactionBase):
    """Schema for transaction output."""
    
    id: int
    transaction_type: TransactionTypeEnum
    status: TransactionStatusEnum
    balance_after: Optional[float]
    is_verified: bool
    created_at: datetime
    updated_at: Optional[datetime]
    
    class Config:
        orm_mode = True
        schema_extra = {
            "example": {
                "id": 1,
                "user_id": 1,
                "date": "2025-06-23",
                "description": "ZOMATO ONLINE PAYMENT",
                "category": "Food & Groceries",
                "amount": -850.0,
                "transaction_type": "debit",
                "payment_method": "upi",
                "status": "completed",
                "upi_id": "zomato@paytm",
                "merchant_name": "Zomato",
                "city": "Mumbai",
                "is_verified": True
            }
        }

class TransactionSummary(BaseModel):
    """Schema for transaction summary statistics."""
    
    total_transactions: int
    total_income: float
    total_expenses: float
    net_amount: float
    category_breakdown: Dict[str, float]
    payment_method_breakdown: Dict[str, int]
    average_transaction_size: float
    largest_expense: float
    largest_income: float

class TransactionFilter(BaseModel):
    """Schema for filtering transactions."""
    
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    category: Optional[CategoryEnum] = None
    payment_method: Optional[PaymentMethodEnum] = None
    min_amount: Optional[float] = None
    max_amount: Optional[float] = None
    transaction_type: Optional[TransactionTypeEnum] = None
    city: Optional[str] = None
    is_business: Optional[bool] = None
    search_term: Optional[str] = None

# === CRUD Functions ===

def create_transaction(db: Session, txn_in: TransactionCreate) -> Transaction:
    """
    Create a new transaction record in the database.
    
    Args:
        db: Database session
        txn_in: Transaction creation data
        
    Returns:
        Transaction: Created transaction object
    """
    try:
        # Determine transaction type based on amount
        transaction_type = TransactionTypeEnum.CREDIT if txn_in.amount > 0 else TransactionTypeEnum.DEBIT
        
        # Create transaction object
        db_txn = Transaction(
            **txn_in.dict(exclude={'transaction_type'}),
            transaction_type=transaction_type
        )
        
        db.add(db_txn)
        db.commit()
        db.refresh(db_txn)
        
        logger.info(f"Created transaction {db_txn.id} for user {db_txn.user_id}")
        return db_txn
        
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to create transaction: {str(e)}")
        raise

def get_transaction(db: Session, transaction_id: int) -> Optional[Transaction]:
    """Get a single transaction by ID."""
    return db.query(Transaction).filter(Transaction.id == transaction_id).first()

def get_transactions_by_user(
    db: Session, 
    user_id: int, 
    skip: int = 0, 
    limit: int = 100,
    filters: Optional[TransactionFilter] = None
) -> List[Transaction]:
    """
    Retrieve transactions for a user with optional filtering.
    
    Args:
        db: Database session
        user_id: User ID
        skip: Number of records to skip
        limit: Maximum number of records to return
        filters: Optional filters to apply
        
    Returns:
        List[Transaction]: List of transactions
    """
    try:
        query = db.query(Transaction).filter(Transaction.user_id == user_id)
        
        # Apply filters if provided
        if filters:
            if filters.start_date:
                query = query.filter(Transaction.date >= filters.start_date)
            if filters.end_date:
                query = query.filter(Transaction.date <= filters.end_date)
            if filters.category:
                query = query.filter(Transaction.category == filters.category)
            if filters.payment_method:
                query = query.filter(Transaction.payment_method == filters.payment_method)
            if filters.min_amount:
                query = query.filter(Transaction.amount >= filters.min_amount)
            if filters.max_amount:
                query = query.filter(Transaction.amount <= filters.max_amount)
            if filters.transaction_type:
                query = query.filter(Transaction.transaction_type == filters.transaction_type)
            if filters.city:
                query = query.filter(Transaction.city.ilike(f"%{filters.city}%"))
            if filters.is_business is not None:
                query = query.filter(Transaction.is_business == filters.is_business)
            if filters.search_term:
                search = f"%{filters.search_term}%"
                query = query.filter(
                    Transaction.description.ilike(search) |
                    Transaction.merchant_name.ilike(search) |
                    Transaction.notes.ilike(search)
                )
        
        return query.order_by(Transaction.date.desc()).offset(skip).limit(limit).all()
        
    except Exception as e:
        logger.error(f"Failed to retrieve transactions: {str(e)}")
        return []

def update_transaction(db: Session, transaction_id: int, txn_update: TransactionUpdate) -> Optional[Transaction]:
    """
    Update an existing transaction.
    
    Args:
        db: Database session
        transaction_id: Transaction ID to update
        txn_update: Update data
        
    Returns:
        Transaction: Updated transaction or None if not found
    """
    try:
        db_txn = db.query(Transaction).filter(Transaction.id == transaction_id).first()
        
        if not db_txn:
            return None
        
        # Update fields
        update_data = txn_update.dict(exclude_unset=True)
        for field, value in update_data.items():
            setattr(db_txn, field, value)
        
        db.commit()
        db.refresh(db_txn)
        
        logger.info(f"Updated transaction {transaction_id}")
        return db_txn
        
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to update transaction {transaction_id}: {str(e)}")
        raise

def delete_transaction(db: Session, transaction_id: int) -> bool:
    """
    Delete a transaction.
    
    Args:
        db: Database session
        transaction_id: Transaction ID to delete
        
    Returns:
        bool: True if deleted, False if not found
    """
    try:
        db_txn = db.query(Transaction).filter(Transaction.id == transaction_id).first()
        
        if not db_txn:
            return False
        
        db.delete(db_txn)
        db.commit()
        
        logger.info(f"Deleted transaction {transaction_id}")
        return True
        
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to delete transaction {transaction_id}: {str(e)}")
        raise

def get_transaction_summary(db: Session, user_id: int, filters: Optional[TransactionFilter] = None) -> TransactionSummary:
    """
    Get transaction summary statistics for a user.
    
    Args:
        db: Database session
        user_id: User ID
        filters: Optional filters to apply
        
    Returns:
        TransactionSummary: Summary statistics
    """
    try:
        transactions = get_transactions_by_user(db, user_id, limit=10000, filters=filters)
        
        if not transactions:
            return TransactionSummary(
                total_transactions=0,
                total_income=0.0,
                total_expenses=0.0,
                net_amount=0.0,
                category_breakdown={},
                payment_method_breakdown={},
                average_transaction_size=0.0,
                largest_expense=0.0,
                largest_income=0.0
            )
        
        # Calculate statistics
        total_transactions = len(transactions)
        income_transactions = [t for t in transactions if t.amount > 0]
        expense_transactions = [t for t in transactions if t.amount < 0]
        
        total_income = sum(t.amount for t in income_transactions)
        total_expenses = abs(sum(t.amount for t in expense_transactions))
        net_amount = total_income - total_expenses
        
        # Category breakdown
        category_breakdown = {}
        for txn in expense_transactions:
            category = txn.category.value
            category_breakdown[category] = category_breakdown.get(category, 0) + abs(txn.amount)
        
        # Payment method breakdown
        payment_method_breakdown = {}
        for txn in transactions:
            if txn.payment_method:
                method = txn.payment_method.value
                payment_method_breakdown[method] = payment_method_breakdown.get(method, 0) + 1
        
        # Other statistics
        all_amounts = [abs(t.amount) for t in transactions]
        average_transaction_size = sum(all_amounts) / len(all_amounts) if all_amounts else 0
        largest_expense = max([abs(t.amount) for t in expense_transactions], default=0)
        largest_income = max([t.amount for t in income_transactions], default=0)
        
        return TransactionSummary(
            total_transactions=total_transactions,
            total_income=total_income,
            total_expenses=total_expenses,
            net_amount=net_amount,
            category_breakdown=category_breakdown,
            payment_method_breakdown=payment_method_breakdown,
            average_transaction_size=round(average_transaction_size, 2),
            largest_expense=largest_expense,
            largest_income=largest_income
        )
        
    except Exception as e:
        logger.error(f"Failed to generate transaction summary: {str(e)}")
        raise

def bulk_create_transactions(db: Session, transactions: List[TransactionCreate]) -> List[Transaction]:
    """
    Create multiple transactions in bulk.
    
    Args:
        db: Database session
        transactions: List of transaction creation data
        
    Returns:
        List[Transaction]: List of created transactions
    """
    try:
        db_transactions = []
        
        for txn_in in transactions:
            transaction_type = TransactionTypeEnum.CREDIT if txn_in.amount > 0 else TransactionTypeEnum.DEBIT
            
            db_txn = Transaction(
                **txn_in.dict(exclude={'transaction_type'}),
                transaction_type=transaction_type
            )
            db_transactions.append(db_txn)
        
        db.add_all(db_transactions)
        db.commit()
        
        for txn in db_transactions:
            db.refresh(txn)
        
        logger.info(f"Created {len(db_transactions)} transactions in bulk")
        return db_transactions
        
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to create bulk transactions: {str(e)}")
        raise

# === Utility Functions ===

def categorize_transaction_auto(description: str, merchant_name: Optional[str] = None) -> CategoryEnum:
    """
    Auto-categorize transaction based on description and merchant.
    
    Args:
        description: Transaction description
        merchant_name: Optional merchant name
        
    Returns:
        CategoryEnum: Predicted category
    """
    text = f"{description} {merchant_name or ''}".upper()
    
    # Food & Groceries
    if any(keyword in text for keyword in ['ZOMATO', 'SWIGGY', 'GROFER', 'BIGBASKET', 'RESTAURANT', 'FOOD']):
        return CategoryEnum.FOOD_GROCERIES
    
    # Transport
    if any(keyword in text for keyword in ['OLA', 'UBER', 'METRO', 'BUS', 'PETROL', 'FUEL']):
        return CategoryEnum.TRANSPORT
    
    # Utilities
    if any(keyword in text for keyword in ['ELECTRICITY', 'WATER', 'GAS', 'INTERNET', 'MOBILE']):
        return CategoryEnum.UTILITIES
    
    # Entertainment
    if any(keyword in text for keyword in ['NETFLIX', 'PRIME', 'MOVIE', 'BOOKMYSHOW']):
        return CategoryEnum.ENTERTAINMENT
    
    # Shopping
    if any(keyword in text for keyword in ['AMAZON', 'FLIPKART', 'MYNTRA', 'SHOPPING']):
        return CategoryEnum.SHOPPING
    
    # Housing
    if any(keyword in text for keyword in ['RENT', 'MAINTENANCE', 'SOCIETY']):
        return CategoryEnum.HOUSING
    
    # Income
    if any(keyword in text for keyword in ['SALARY', 'CREDIT', 'REFUND']):
        return CategoryEnum.INCOME
    
    return CategoryEnum.OTHER

# Example usage and testing
if __name__ == "__main__":
    print("Testing Transaction model...")
    
    # Test transaction creation
    sample_transaction = TransactionCreate(
        user_id=1,
        date=date.today(),
        description="ZOMATO ONLINE PAYMENT",
        category=CategoryEnum.FOOD_GROCERIES,
        amount=-850.0,
        payment_method=PaymentMethodEnum.UPI,
        upi_id="zomato@paytm",
        merchant_name="Zomato",
        city="Mumbai"
    )
    
    print(f"Sample transaction: {sample_transaction}")
    
    # Test auto-categorization
    category = categorize_transaction_auto("ZOMATO ONLINE PAYMENT", "Zomato")
    print(f"Auto-categorized as: {category}")
    
    print("Transaction model tests completed successfully!")
