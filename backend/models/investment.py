# backend/models/investment.py

from sqlalchemy import Column, Integer, Float, String, ForeignKey, Date, DateTime, Boolean, JSON
from sqlalchemy.orm import Session, relationship
from pydantic import BaseModel, validator
from datetime import date, datetime
from typing import Optional, Dict, Any, List
from enum import Enum
import logging

from backend.models.database import Base

logger = logging.getLogger(__name__)

class InvestmentType(str, Enum):
    MUTUAL_FUND = "Mutual Fund"
    STOCK = "Stock"
    FIXED_DEPOSIT = "Fixed Deposit"
    PPF = "PPF"
    EPF = "EPF"
    NSC = "NSC"
    BOND = "Bond"
    GOLD = "Gold"
    REAL_ESTATE = "Real Estate"
    CRYPTO = "Cryptocurrency"
    SIP = "SIP"
    ELSS = "ELSS"

class RiskLevel(str, Enum):
    LOW = "Low"
    MODERATE = "Moderate"
    HIGH = "High"
    VERY_HIGH = "Very High"

class Investment(Base):
    __tablename__ = "investments"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    date = Column(Date, nullable=False)
    investment_type = Column(String(32), nullable=False)
    name = Column(String(128), nullable=False)
    symbol = Column(String(32), nullable=True)  # For stocks/mutual funds
    amount = Column(Float, nullable=False)
    units = Column(Float, nullable=True)  # For mutual funds/stocks
    purchase_price = Column(Float, nullable=False)
    current_value = Column(Float, nullable=True)
    current_price = Column(Float, nullable=True)
    last_updated = Column(DateTime, default=datetime.utcnow)
    maturity_date = Column(Date, nullable=True)
    risk_level = Column(String(16), nullable=True)
    is_active = Column(Boolean, default=True)
    metadata = Column(JSON, nullable=True)  # Additional investment details
    
    # Relationship to user (assuming User model exists)
    # user = relationship("User", back_populates="investments")

# Pydantic schemas
class InvestmentBase(BaseModel):
    user_id: int
    date: date
    investment_type: InvestmentType
    name: str
    symbol: Optional[str] = None
    amount: float
    units: Optional[float] = None
    purchase_price: float
    current_value: Optional[float] = None
    current_price: Optional[float] = None
    maturity_date: Optional[date] = None
    risk_level: Optional[RiskLevel] = None
    metadata: Optional[Dict[str, Any]] = None

    @validator('amount', 'purchase_price')
    def validate_positive_amounts(cls, v):
        if v <= 0:
            raise ValueError('Amount and purchase price must be positive')
        return v

    @validator('current_value', 'current_price', 'units')
    def validate_optional_positive(cls, v):
        if v is not None and v < 0:
            raise ValueError('Values must be non-negative')
        return v

class InvestmentCreate(InvestmentBase):
    pass

class InvestmentUpdate(BaseModel):
    name: Optional[str] = None
    current_value: Optional[float] = None
    current_price: Optional[float] = None
    is_active: Optional[bool] = None
    metadata: Optional[Dict[str, Any]] = None

class InvestmentOut(InvestmentBase):
    id: int
    last_updated: datetime
    is_active: bool
    
    # Calculated fields
    total_return: Optional[float] = None
    return_percentage: Optional[float] = None
    
    class Config:
        orm_mode = True

class PortfolioSummary(BaseModel):
    total_invested: float
    current_value: float
    total_return: float
    return_percentage: float
    investments_by_type: Dict[str, float]
    risk_distribution: Dict[str, float]
    top_performers: List[Dict[str, Any]]
    underperformers: List[Dict[str, Any]]

# CRUD functions
def create_investment(db: Session, inv_in: InvestmentCreate) -> Investment:
    """Create a new investment record."""
    try:
        # Calculate units if not provided
        if inv_in.units is None and inv_in.purchase_price > 0:
            calculated_units = inv_in.amount / inv_in.purchase_price
            inv_dict = inv_in.dict()
            inv_dict['units'] = calculated_units
        else:
            inv_dict = inv_in.dict()
        
        # Set initial current values
        if inv_dict.get('current_value') is None:
            inv_dict['current_value'] = inv_dict['amount']
        if inv_dict.get('current_price') is None:
            inv_dict['current_price'] = inv_dict['purchase_price']
            
        db_inv = Investment(**inv_dict)
        db.add(db_inv)
        db.commit()
        db.refresh(db_inv)
        logger.info(f"Created investment {db_inv.id} for user {db_inv.user_id}")
        return db_inv
    except Exception as e:
        logger.error(f"Error creating investment: {str(e)}")
        db.rollback()
        raise

def get_investment(db: Session, investment_id: int) -> Optional[Investment]:
    """Get investment by ID."""
    return db.query(Investment).filter(Investment.id == investment_id).first()

def get_investments_by_user(db: Session, user_id: int, 
                          investment_type: Optional[str] = None,
                          is_active: bool = True) -> List[Investment]:
    """Get all investments for a user with optional filtering."""
    query = db.query(Investment).filter(Investment.user_id == user_id)
    
    if investment_type:
        query = query.filter(Investment.investment_type == investment_type)
    
    if is_active is not None:
        query = query.filter(Investment.is_active == is_active)
    
    return query.order_by(Investment.date.desc()).all()

def update_investment(db: Session, investment_id: int, 
                     inv_update: InvestmentUpdate) -> Optional[Investment]:
    """Update an existing investment."""
    try:
        db_inv = db.query(Investment).filter(Investment.id == investment_id).first()
        if not db_inv:
            return None
        
        update_data = inv_update.dict(exclude_unset=True)
        for field, value in update_data.items():
            setattr(db_inv, field, value)
        
        db_inv.last_updated = datetime.utcnow()
        db.commit()
        db.refresh(db_inv)
        logger.info(f"Updated investment {investment_id}")
        return db_inv
    except Exception as e:
        logger.error(f"Error updating investment {investment_id}: {str(e)}")
        db.rollback()
        raise

def delete_investment(db: Session, investment_id: int) -> bool:
    """Soft delete an investment (mark as inactive)."""
    try:
        db_inv = db.query(Investment).filter(Investment.id == investment_id).first()
        if not db_inv:
            return False
        
        db_inv.is_active = False
        db_inv.last_updated = datetime.utcnow()
        db.commit()
        logger.info(f"Deleted investment {investment_id}")
        return True
    except Exception as e:
        logger.error(f"Error deleting investment {investment_id}: {str(e)}")
        db.rollback()
        raise

def get_portfolio_summary(db: Session, user_id: int) -> PortfolioSummary:
    """Generate comprehensive portfolio summary for a user."""
    try:
        investments = get_investments_by_user(db, user_id, is_active=True)
        
        if not investments:
            return PortfolioSummary(
                total_invested=0,
                current_value=0,
                total_return=0,
                return_percentage=0,
                investments_by_type={},
                risk_distribution={},
                top_performers=[],
                underperformers=[]
            )
        
        total_invested = sum(inv.amount for inv in investments)
        current_value = sum(inv.current_value or inv.amount for inv in investments)
        total_return = current_value - total_invested
        return_percentage = (total_return / total_invested * 100) if total_invested > 0 else 0
        
        # Group by investment type
        investments_by_type = {}
        for inv in investments:
            inv_type = inv.investment_type
            investments_by_type[inv_type] = investments_by_type.get(inv_type, 0) + (inv.current_value or inv.amount)
        
        # Risk distribution
        risk_distribution = {}
        for inv in investments:
            risk = inv.risk_level or "Unknown"
            risk_distribution[risk] = risk_distribution.get(risk, 0) + (inv.current_value or inv.amount)
        
        # Performance analysis
        performance_data = []
        for inv in investments:
            inv_return = (inv.current_value or inv.amount) - inv.amount
            inv_return_pct = (inv_return / inv.amount * 100) if inv.amount > 0 else 0
            performance_data.append({
                "id": inv.id,
                "name": inv.name,
                "type": inv.investment_type,
                "return": inv_return,
                "return_percentage": inv_return_pct,
                "current_value": inv.current_value or inv.amount
            })
        
        # Sort by performance
        performance_data.sort(key=lambda x: x["return_percentage"], reverse=True)
        top_performers = performance_data[:3] if len(performance_data) >= 3 else performance_data
        underperformers = performance_data[-3:] if len(performance_data) >= 3 else []
        
        return PortfolioSummary(
            total_invested=total_invested,
            current_value=current_value,
            total_return=total_return,
            return_percentage=return_percentage,
            investments_by_type=investments_by_type,
            risk_distribution=risk_distribution,
            top_performers=top_performers,
            underperformers=underperformers
        )
    except Exception as e:
        logger.error(f"Error generating portfolio summary for user {user_id}: {str(e)}")
        raise

def update_investment_values(db: Session, investment_id: int, 
                           current_price: float, current_value: Optional[float] = None) -> Optional[Investment]:
    """Update investment with latest market values."""
    try:
        db_inv = db.query(Investment).filter(Investment.id == investment_id).first()
        if not db_inv:
            return None
        
        db_inv.current_price = current_price
        db_inv.last_updated = datetime.utcnow()
        
        # Calculate current value if not provided
        if current_value is None and db_inv.units:
            db_inv.current_value = db_inv.units * current_price
        elif current_value is not None:
            db_inv.current_value = current_value
        
        db.commit()
        db.refresh(db_inv)
        return db_inv
    except Exception as e:
        logger.error(f"Error updating investment values for {investment_id}: {str(e)}")
        db.rollback()
        raise

def get_investments_by_type(db: Session, user_id: int, investment_type: str) -> List[Investment]:
    """Get investments filtered by type."""
    return db.query(Investment).filter(
        Investment.user_id == user_id,
        Investment.investment_type == investment_type,
        Investment.is_active == True
    ).order_by(Investment.date.desc()).all()

def get_investment_performance(db: Session, investment_id: int) -> Optional[Dict[str, Any]]:
    """Calculate detailed performance metrics for an investment."""
    try:
        inv = db.query(Investment).filter(Investment.id == investment_id).first()
        if not inv:
            return None
        
        current_val = inv.current_value or inv.amount
        total_return = current_val - inv.amount
        return_percentage = (total_return / inv.amount * 100) if inv.amount > 0 else 0
        
        # Calculate annualized return
        days_held = (datetime.now().date() - inv.date).days
        if days_held > 0:
            annualized_return = ((current_val / inv.amount) ** (365 / days_held) - 1) * 100
        else:
            annualized_return = 0
        
        return {
            "investment_id": investment_id,
            "total_return": total_return,
            "return_percentage": return_percentage,
            "annualized_return": annualized_return,
            "current_value": current_val,
            "invested_amount": inv.amount,
            "days_held": days_held
        }
    except Exception as e:
        logger.error(f"Error calculating performance for investment {investment_id}: {str(e)}")
        return None
