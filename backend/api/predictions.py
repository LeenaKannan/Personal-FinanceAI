# backend/api/insights.py

from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
from datetime import datetime, date, timedelta
from enum import Enum
import logging

from backend.models import transaction, user
from backend.models.database import get_db
from backend.api.auth import get_current_user
from backend.ml_engine.insights_generator import InsightsGenerator
from backend.utils.data_processor import DataProcessor
from pydantic import BaseModel, Field

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/insights", tags=["insights"])

# Initialize engines
insights_engine = InsightsGenerator()
data_processor = DataProcessor()

# === ENUMS ===

class InsightTypeEnum(str, Enum):
    TIP = "tip"
    WARNING = "warning"
    OPPORTUNITY = "opportunity"
    ACHIEVEMENT = "achievement"
    ALERT = "alert"

class InsightCategoryEnum(str, Enum):
    SPENDING = "spending"
    SAVING = "saving"
    INVESTMENT = "investment"
    BUDGET = "budget"
    GOAL = "goal"
    SECURITY = "security"

class TimeRangeEnum(str, Enum):
    WEEK = "week"
    MONTH = "month"
    QUARTER = "quarter"
    YEAR = "year"
    ALL = "all"

# === PYDANTIC SCHEMAS ===

class InsightOut(BaseModel):
    """Schema for individual insight output."""
    
    id: str = Field(..., description="Unique insight identifier")
    type: InsightTypeEnum = Field(..., description="Type of insight")
    category: InsightCategoryEnum = Field(..., description="Category of insight")
    title: str = Field(..., description="Insight title")
    message: str = Field(..., description="Detailed insight message")
    action: Optional[str] = Field(None, description="Recommended action")
    priority: int = Field(..., ge=1, le=5, description="Priority level (1=low, 5=high)")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score")
    impact: str = Field(..., description="Expected impact (low/medium/high)")
    created_at: datetime = Field(..., description="When insight was generated")
    expires_at: Optional[datetime] = Field(None, description="When insight expires")
    
    # Financial metrics
    amount_involved: Optional[float] = Field(None, description="Amount related to insight")
    percentage_change: Optional[float] = Field(None, description="Percentage change if applicable")
    
    # Metadata
    tags: List[str] = Field(default_factory=list, description="Insight tags")
    related_categories: List[str] = Field(default_factory=list, description="Related expense categories")
    
    class Config:
        schema_extra = {
            "example": {
                "id": "insight_001",
                "type": "warning",
                "category": "spending",
                "title": "High Food Spending Alert",
                "message": "Your food expenses increased by 30% this month compared to your average.",
                "action": "Consider cooking at home more often or exploring budget-friendly restaurants.",
                "priority": 4,
                "confidence": 0.85,
                "impact": "medium",
                "amount_involved": 5000.0,
                "percentage_change": 30.0,
                "tags": ["food", "overspending"],
                "related_categories": ["Food & Groceries"]
            }
        }

class InsightsSummary(BaseModel):
    """Schema for insights summary."""
    
    total_insights: int
    high_priority_count: int
    categories_covered: List[str]
    potential_savings: float
    last_generated: datetime
    next_update: datetime

class InsightsFilter(BaseModel):
    """Schema for filtering insights."""
    
    insight_type: Optional[InsightTypeEnum] = None
    category: Optional[InsightCategoryEnum] = None
    min_priority: Optional[int] = Field(None, ge=1, le=5)
    time_range: Optional[TimeRangeEnum] = TimeRangeEnum.MONTH
    include_expired: Optional[bool] = False

# === UTILITY FUNCTIONS ===

def calculate_user_metrics(db: Session, user_id: int, time_range: TimeRangeEnum) -> Dict[str, Any]:
    """Calculate comprehensive user financial metrics."""
    try:
        # Get date range
        end_date = datetime.now().date()
        if time_range == TimeRangeEnum.WEEK:
            start_date = end_date - timedelta(days=7)
        elif time_range == TimeRangeEnum.MONTH:
            start_date = end_date - timedelta(days=30)
        elif time_range == TimeRangeEnum.QUARTER:
            start_date = end_date - timedelta(days=90)
        elif time_range == TimeRangeEnum.YEAR:
            start_date = end_date - timedelta(days=365)
        else:  # ALL
            start_date = end_date - timedelta(days=1095)  # 3 years
        
        # Get transactions
        txn_filter = transaction.TransactionFilter(
            start_date=start_date,
            end_date=end_date
        )
        
        transactions = transaction.get_transactions_by_user(
            db, user_id, filters=txn_filter
        )
        
        if not transactions:
            return {}
        
        # Calculate metrics
        total_income = sum(t.amount for t in transactions if t.amount > 0)
        total_expenses = abs(sum(t.amount for t in transactions if t.amount < 0))
        net_savings = total_income - total_expenses
        savings_rate = (net_savings / total_income * 100) if total_income > 0 else 0
        
        # Category breakdown
        category_expenses = {}
        for txn in transactions:
            if txn.amount < 0:  # Expenses only
                category = txn.category.value if hasattr(txn.category, 'value') else str(txn.category)
                category_expenses[category] = category_expenses.get(category, 0) + abs(txn.amount)
        
        # Calculate averages for comparison
        days_in_period = (end_date - start_date).days
        daily_expenses = total_expenses / days_in_period if days_in_period > 0 else 0
        monthly_expenses = daily_expenses * 30
        
        # Previous period comparison
        prev_start = start_date - timedelta(days=days_in_period)
        prev_filter = transaction.TransactionFilter(
            start_date=prev_start,
            end_date=start_date
        )
        
        prev_transactions = transaction.get_transactions_by_user(
            db, user_id, filters=prev_filter
        )
        
        prev_expenses = abs(sum(t.amount for t in prev_transactions if t.amount < 0))
        expense_change = ((total_expenses - prev_expenses) / prev_expenses * 100) if prev_expenses > 0 else 0
        
        return {
            'total_income': total_income,
            'total_expenses': total_expenses,
            'net_savings': net_savings,
            'savings_rate': savings_rate,
            'category_expenses': category_expenses,
            'monthly_expenses': monthly_expenses,
            'expense_change': expense_change,
            'transaction_count': len(transactions),
            'avg_transaction_size': total_expenses / len([t for t in transactions if t.amount < 0]) if transactions else 0,
            'time_range': time_range,
            'period_days': days_in_period
        }
        
    except Exception as e:
        logger.error(f"Failed to calculate user metrics: {e}")
        return {}

def generate_spending_insights(user_data: Dict[str, Any], metrics: Dict[str, Any]) -> List[InsightOut]:
    """Generate spending-related insights."""
    insights = []
    
    try:
        # High spending categories
        category_expenses = metrics.get('category_expenses', {})
        total_expenses = metrics.get('total_expenses', 0)
        
        for category, amount in category_expenses.items():
            if total_expenses > 0:
                percentage = (amount / total_expenses) * 100
                
                # Category-specific thresholds
                thresholds = {
                    'Housing': 40,
                    'Food & Groceries': 25,
                    'Transport': 20,
                    'Entertainment': 15,
                    'Shopping': 15
                }
                
                threshold = thresholds.get(category, 10)
                
                if percentage > threshold:
                    insights.append(InsightOut(
                        id=f"spending_{category.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}",
                        type=InsightTypeEnum.WARNING,
                        category=InsightCategoryEnum.SPENDING,
                        title=f"High {category} Spending",
                        message=f"You spent {percentage:.1f}% of your budget on {category}, which is above the recommended {threshold}%.",
                        action=f"Consider reviewing your {category.lower()} expenses and finding areas to optimize.",
                        priority=4 if percentage > threshold * 1.5 else 3,
                        confidence=0.9,
                        impact="medium" if percentage > threshold * 1.2 else "low",
                        created_at=datetime.now(),
                        amount_involved=amount,
                        percentage_change=percentage - threshold,
                        tags=[category.lower(), "overspending"],
                        related_categories=[category]
                    ))
        
        # Expense trend analysis
        expense_change = metrics.get('expense_change', 0)
        if abs(expense_change) > 20:
            insight_type = InsightTypeEnum.WARNING if expense_change > 0 else InsightTypeEnum.ACHIEVEMENT
            title = f"Expenses {'Increased' if expense_change > 0 else 'Decreased'} Significantly"
            message = f"Your expenses {'increased' if expense_change > 0 else 'decreased'} by {abs(expense_change):.1f}% compared to the previous period."
            
            insights.append(InsightOut(
                id=f"expense_trend_{datetime.now().strftime('%Y%m%d')}",
                type=insight_type,
                category=InsightCategoryEnum.SPENDING,
                title=title,
                message=message,
                action="Review your recent transactions to understand this change." if expense_change > 0 else "Great job on reducing expenses!",
                priority=4 if expense_change > 30 else 3,
                confidence=0.85,
                impact="high" if abs(expense_change) > 30 else "medium",
                created_at=datetime.now(),
                percentage_change=expense_change,
                tags=["trend", "expenses"]
            ))
        
    except Exception as e:
        logger.error(f"Failed to generate spending insights: {e}")
    
    return insights

def generate_savings_insights(user_data: Dict[str, Any], metrics: Dict[str, Any]) -> List[InsightOut]:
    """Generate savings-related insights."""
    insights = []
    
    try:
        savings_rate = metrics.get('savings_rate', 0)
        income = user_data.get('income', 0)
        
        # Savings rate analysis
        if savings_rate < 10:
            insights.append(InsightOut(
                id=f"low_savings_{datetime.now().strftime('%Y%m%d')}",
                type=InsightTypeEnum.WARNING,
                category=InsightCategoryEnum.SAVING,
                title="Low Savings Rate",
                message=f"Your current savings rate is {savings_rate:.1f}%, which is below the recommended 20%.",
                action="Try the 50/30/20 rule: 50% needs, 30% wants, 20% savings.",
                priority=5,
                confidence=0.95,
                impact="high",
                created_at=datetime.now(),
                percentage_change=savings_rate - 20,
                tags=["savings", "financial_health"]
            ))
        elif savings_rate > 30:
            insights.append(InsightOut(
                id=f"excellent_savings_{datetime.now().strftime('%Y%m%d')}",
                type=InsightTypeEnum.ACHIEVEMENT,
                category=InsightCategoryEnum.SAVING,
                title="Excellent Savings Rate!",
                message=f"Congratulations! Your savings rate of {savings_rate:.1f}% is excellent.",
                action="Consider investing your surplus savings for better returns.",
                priority=2,
                confidence=0.95,
                impact="low",
                created_at=datetime.now(),
                percentage_change=savings_rate - 20,
                tags=["savings", "achievement"]
            ))
        
        # Emergency fund suggestion
        monthly_expenses = metrics.get('monthly_expenses', 0)
        if monthly_expenses > 0:
            emergency_fund_needed = monthly_expenses * 6  # 6 months
            current_savings = metrics.get('net_savings', 0) * 12  # Annualized
            
            if current_savings < emergency_fund_needed:
                insights.append(InsightOut(
                    id=f"emergency_fund_{datetime.now().strftime('%Y%m%d')}",
                    type=InsightTypeEnum.TIP,
                    category=InsightCategoryEnum.SAVING,
                    title="Build Emergency Fund",
                    message=f"Consider building an emergency fund of ₹{emergency_fund_needed:,.0f} (6 months of expenses).",
                    action="Start with a goal of saving ₹5,000 per month towards your emergency fund.",
                    priority=3,
                    confidence=0.9,
                    impact="medium",
                    created_at=datetime.now(),
                    amount_involved=emergency_fund_needed - current_savings,
                    tags=["emergency_fund", "financial_security"]
                ))
        
    except Exception as e:
        logger.error(f"Failed to generate savings insights: {e}")
    
    return insights

def generate_investment_insights(user_data: Dict[str, Any], metrics: Dict[str, Any]) -> List[InsightOut]:
    """Generate investment-related insights."""
    insights = []
    
    try:
        age = user_data.get('age', 30)
        income = user_data.get('income', 0)
        savings_rate = metrics.get('savings_rate', 0)
        
        # Age-based investment advice
        if age < 30 and savings_rate > 20:
            insights.append(InsightOut(
                id=f"young_investor_{datetime.now().strftime('%Y%m%d')}",
                type=InsightTypeEnum.OPPORTUNITY,
                category=InsightCategoryEnum.INVESTMENT,
                title="Start Investing Early",
                message="You're young with good savings! This is the perfect time to start investing for long-term wealth creation.",
                action="Consider starting SIPs in equity mutual funds or index funds.",
                priority=3,
                confidence=0.8,
                impact="high",
                created_at=datetime.now(),
                tags=["investment", "young_investor", "equity"]
            ))
        
        elif age > 40:
            insights.append(InsightOut(
                id=f"retirement_planning_{datetime.now().strftime('%Y%m%d')}",
                type=InsightTypeEnum.TIP,
                category=InsightCategoryEnum.INVESTMENT,
                title="Focus on Retirement Planning",
                message="Consider increasing allocation to retirement-focused investments like PPF, NPS, and balanced funds.",
                action="Review your retirement corpus goal and adjust investments accordingly.",
                priority=4,
                confidence=0.85,
                impact="high",
                created_at=datetime.now(),
                tags=["retirement", "investment", "planning"]
            ))
        
        # Tax saving suggestions
        if income > 500000:  # Above tax exemption limit
            insights.append(InsightOut(
                id=f"tax_saving_{datetime.now().strftime('%Y%m%d')}",
                type=InsightTypeEnum.TIP,
                category=InsightCategoryEnum.INVESTMENT,
                title="Tax Saving Opportunity",
                message="Maximize tax savings through 80C investments like ELSS, PPF, or life insurance.",
                action="Invest up to ₹1.5 lakh in tax-saving instruments to save up to ₹46,800 in taxes.",
                priority=3,
                confidence=0.9,
                impact="medium",
                created_at=datetime.now(),
                amount_involved=46800,  # Max tax saving
                tags=["tax_saving", "80c", "investment"]
            ))
        
    except Exception as e:
        logger.error(f"Failed to generate investment insights: {e}")
    
    return insights

def generate_goal_insights(user_data: Dict[str, Any], metrics: Dict[str, Any]) -> List[InsightOut]:
    """Generate goal-related insights."""
    insights = []
    
    try:
        age = user_data.get('age', 30)
        income = user_data.get('income', 0)
        city = user_data.get('city', 'Mumbai')
        
        # Home buying goal
        if age < 35 and income > 50000:
            home_price = data_processor.adjust_for_city_cost(5000000, city)  # Base 50L home
            down_payment = home_price * 0.2
            
            insights.append(InsightOut(
                id=f"home_buying_{datetime.now().strftime('%Y%m%d')}",
                type=InsightTypeEnum.OPPORTUNITY,
                category=InsightCategoryEnum.GOAL,
                title="Plan for Home Purchase",
                message=f"Based on your income and city, consider planning for a home purchase. Estimated down payment needed: ₹{down_payment:,.0f}",
                action="Start a dedicated savings plan for home down payment.",
                priority=2,
                confidence=0.7,
                impact="medium",
                created_at=datetime.now(),
                amount_involved=down_payment,
                tags=["home_buying", "goal", "real_estate"]
            ))
        
        # Education fund for children (if applicable)
        # This would need user profile data about dependents
        
    except Exception as e:
        logger.error(f"Failed to generate goal insights: {e}")
    
    return insights

# === API ENDPOINTS ===

@router.get("/", response_model=List[InsightOut])
async def get_insights(
    db: Session = Depends(get_db),
    current_user: user.User = Depends(get_current_user),
    time_range: TimeRangeEnum = Query(TimeRangeEnum.MONTH, description="Time range for analysis"),
    limit: int = Query(20, ge=1, le=100, description="Maximum number of insights"),
    insight_type: Optional[InsightTypeEnum] = Query(None, description="Filter by insight type"),
    category: Optional[InsightCategoryEnum] = Query(None, description="Filter by category"),
    min_priority: Optional[int] = Query(None, ge=1, le=5, description="Minimum priority level")
):
    """
    Get personalized financial insights for the current user.
    """
    try:
        # Calculate user metrics
        metrics = calculate_user_metrics(db, current_user.id, time_range)
        
        if not metrics:
            return []
        
        # Prepare user data
        user_data = {
            'income': current_user.income,
            'age': current_user.age,
            'city': current_user.city,
            'gender': current_user.gender
        }
        
        # Generate insights from different categories
        all_insights = []
        
        # Spending insights
        all_insights.extend(generate_spending_insights(user_data, metrics))
        
        # Savings insights
        all_insights.extend(generate_savings_insights(user_data, metrics))
        
        # Investment insights
        all_insights.extend(generate_investment_insights(user_data, metrics))
        
        # Goal insights
        all_insights.extend(generate_goal_insights(user_data, metrics))
        
        # Apply filters
        filtered_insights = all_insights
        
        if insight_type:
            filtered_insights = [i for i in filtered_insights if i.type == insight_type]
        
        if category:
            filtered_insights = [i for i in filtered_insights if i.category == category]
        
        if min_priority:
            filtered_insights = [i for i in filtered_insights if i.priority >= min_priority]
        
        # Sort by priority (descending) and limit
        filtered_insights.sort(key=lambda x: x.priority, reverse=True)
        
        logger.info(f"Generated {len(filtered_insights)} insights for user {current_user.id}")
        
        return filtered_insights[:limit]
        
    except Exception as e:
        logger.error(f"Failed to get insights for user {current_user.id}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to generate insights"
        )

@router.get("/summary", response_model=InsightsSummary)
async def get_insights_summary(
    db: Session = Depends(get_db),
    current_user: user.User = Depends(get_current_user),
    time_range: TimeRangeEnum = Query(TimeRangeEnum.MONTH, description="Time range for analysis")
):
    """
    Get summary of insights for the current user.
    """
    try:
        # Get all insights
        insights = await get_insights(db, current_user, time_range, limit=100)
        
        # Calculate summary
        high_priority_count = len([i for i in insights if i.priority >= 4])
        categories_covered = list(set([i.category.value for i in insights]))
        
        # Calculate potential savings
        potential_savings = sum([
            i.amount_involved for i in insights 
            if i.amount_involved and i.type in [InsightTypeEnum.WARNING, InsightTypeEnum.OPPORTUNITY]
        ])
        
        return InsightsSummary(
            total_insights=len(insights),
            high_priority_count=high_priority_count,
            categories_covered=categories_covered,
            potential_savings=potential_savings,
            last_generated=datetime.now(),
            next_update=datetime.now() + timedelta(days=1)
        )
        
    except Exception as e:
        logger.error(f"Failed to get insights summary for user {current_user.id}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to generate insights summary"
        )

@router.post("/refresh")
async def refresh_insights(
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: user.User = Depends(get_current_user)
):
    """
    Refresh insights for the current user.
    """
    try:
        # Add background task to regenerate insights
        # background_tasks.add_task(regenerate_user_insights, current_user.id)
        
        logger.info(f"Insights refresh requested for user {current_user.id}")
        
        return {"message": "Insights refresh initiated"}
        
    except Exception as e:
        logger.error(f"Failed to refresh insights for user {current_user.id}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to refresh insights"
        )

@router.get("/categories", response_model=List[str])
async def get_insight_categories():
    """
    Get available insight categories.
    """
    return [category.value for category in InsightCategoryEnum]

@router.get("/types", response_model=List[str])
async def get_insight_types():
    """
    Get available insight types.
    """
    return [insight_type.value for insight_type in InsightTypeEnum]

# Health check
@router.get("/health")
async def insights_health_check():
    """Insights service health check."""
    return {
        "status": "healthy",
        "service": "insights",
        "timestamp": datetime.now().isoformat()
    }
