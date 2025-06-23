# backend/models/user.py

import logging
from datetime import datetime, timezone
from typing import Optional, List
from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Enum as SQLEnum
from sqlalchemy.orm import Session, relationship
from sqlalchemy.exc import IntegrityError
from pydantic import BaseModel, EmailStr, validator, Field
from passlib.context import CryptContext
import enum
from backend.models.database import Base

logger = logging.getLogger(__name__)

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class GenderEnum(str, enum.Enum):
    MALE = "male"
    FEMALE = "female"
    OTHER = "other"
    PREFER_NOT_TO_SAY = "prefer_not_to_say"

class EmploymentTypeEnum(str, enum.Enum):
    SALARIED = "salaried"
    SELF_EMPLOYED = "self_employed"
    FREELANCER = "freelancer"
    BUSINESS_OWNER = "business_owner"
    STUDENT = "student"
    UNEMPLOYED = "unemployed"
    RETIRED = "retired"

# Indian cities with cost of living data
INDIAN_CITIES = [
    "Mumbai", "Delhi", "Bangalore", "Chennai", "Pune", "Hyderabad", 
    "Kolkata", "Ahmedabad", "Jaipur", "Lucknow", "Kochi", "Chandigarh",
    "Bhubaneswar", "Indore", "Nagpur", "Coimbatore", "Visakhapatnam"
]

# SQLAlchemy model
class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False, index=True)
    email = Column(String(120), unique=True, index=True, nullable=False)
    hashed_password = Column(String(128), nullable=False)
    phone = Column(String(15), unique=True, index=True)
    
    # Demographics
    city = Column(String(50), nullable=False)
    state = Column(String(50))
    gender = Column(SQLEnum(GenderEnum), nullable=False)
    age = Column(Integer, nullable=False)
    
    # Financial profile
    income = Column(Float, nullable=False)
    employment_type = Column(SQLEnum(EmploymentTypeEnum), default=EmploymentTypeEnum.SALARIED)
    financial_goals = Column(String(500))  # JSON string
    risk_tolerance = Column(String(20), default="moderate")  # conservative, moderate, aggressive
    
    # Account management
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), default=datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=datetime.now(timezone.utc), onupdate=datetime.now(timezone.utc))
    last_login = Column(DateTime(timezone=True))
    
    # Relationships
    transactions = relationship("Transaction", back_populates="user", cascade="all, delete-orphan")
    investments = relationship("Investment", back_populates="user", cascade="all, delete-orphan")
    budgets = relationship("Budget", back_populates="user", cascade="all, delete-orphan")

    def verify_password(self, password: str) -> bool:
        """Verify user password"""
        return pwd_context.verify(password, self.hashed_password)
    
    def set_password(self, password: str):
        """Set user password hash"""
        self.hashed_password = pwd_context.hash(password)
    
    def update_last_login(self):
        """Update last login timestamp"""
        self.last_login = datetime.now(timezone.utc)

# Pydantic schemas
class UserBase(BaseModel):
    name: str = Field(..., min_length=2, max_length=100)
    email: EmailStr
    phone: Optional[str] = Field(None, regex=r'^\+?91?[6-9]\d{9}$')
    city: str
    state: Optional[str] = None
    gender: GenderEnum
    age: int = Field(..., ge=18, le=100)
    income: float = Field(..., gt=0)
    employment_type: EmploymentTypeEnum = EmploymentTypeEnum.SALARIED
    financial_goals: Optional[str] = None
    risk_tolerance: str = Field("moderate", regex=r'^(conservative|moderate|aggressive)$')
    
    @validator('city')
    def validate_city(cls, v):
        if v not in INDIAN_CITIES:
            raise ValueError(f'City must be one of: {", ".join(INDIAN_CITIES)}')
        return v
    
    @validator('income')
    def validate_income(cls, v):
        if v < 10000:  # Minimum monthly income
            raise ValueError('Monthly income should be at least ₹10,000')
        return v

class UserCreate(UserBase):
    password: str = Field(..., min_length=8)
    confirm_password: str
    
    @validator('confirm_password')
    def passwords_match(cls, v, values):
        if 'password' in values and v != values['password']:
            raise ValueError('Passwords do not match')
        return v
    
    @validator('password')
    def validate_password(cls, v):
        if not any(c.isupper() for c in v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not any(c.islower() for c in v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not any(c.isdigit() for c in v):
            raise ValueError('Password must contain at least one digit')
        return v

class UserUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=2, max_length=100)
    phone: Optional[str] = Field(None, regex=r'^\+?91?[6-9]\d{9}$')
    city: Optional[str] = None
    state: Optional[str] = None
    income: Optional[float] = Field(None, gt=0)
    employment_type: Optional[EmploymentTypeEnum] = None
    financial_goals: Optional[str] = None
    risk_tolerance: Optional[str] = Field(None, regex=r'^(conservative|moderate|aggressive)$')
    
    @validator('city')
    def validate_city(cls, v):
        if v and v not in INDIAN_CITIES:
            raise ValueError(f'City must be one of: {", ".join(INDIAN_CITIES)}')
        return v

class UserOut(UserBase):
    id: int
    is_active: bool
    is_verified: bool
    created_at: datetime
    updated_at: datetime
    last_login: Optional[datetime] = None
    
    class Config:
        from_attributes = True

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserProfile(BaseModel):
    """Simplified user profile for public display"""
    id: int
    name: str
    city: str
    age: int
    employment_type: EmploymentTypeEnum
    created_at: datetime
    
    class Config:
        from_attributes = True

# CRUD functions
class UserCRUD:
    @staticmethod
    def get_user(db: Session, user_id: int) -> Optional[User]:
        """Get user by ID"""
        try:
            return db.query(User).filter(User.id == user_id, User.is_active == True).first()
        except Exception as e:
            logger.error(f"Error fetching user {user_id}: {e}")
            return None
    
    @staticmethod
    def get_user_by_email(db: Session, email: str) -> Optional[User]:
        """Get user by email"""
        try:
            return db.query(User).filter(User.email == email, User.is_active == True).first()
        except Exception as e:
            logger.error(f"Error fetching user by email {email}: {e}")
            return None
    
    @staticmethod
    def get_user_by_phone(db: Session, phone: str) -> Optional[User]:
        """Get user by phone"""
        try:
            return db.query(User).filter(User.phone == phone, User.is_active == True).first()
        except Exception as e:
            logger.error(f"Error fetching user by phone {phone}: {e}")
            return None
    
    @staticmethod
    def create_user(db: Session, user_in: UserCreate) -> Optional[User]:
        """Create new user"""
        try:
            # Check if user exists
            if UserCRUD.get_user_by_email(db, user_in.email):
                raise ValueError("User with this email already exists")
            
            if user_in.phone and UserCRUD.get_user_by_phone(db, user_in.phone):
                raise ValueError("User with this phone number already exists")
            
            # Create user
            user_data = user_in.dict(exclude={'password', 'confirm_password'})
            db_user = User(**user_data)
            db_user.set_password(user_in.password)
            
            db.add(db_user)
            db.commit()
            db.refresh(db_user)
            
            logger.info(f"User created successfully: {db_user.id}")
            return db_user
            
        except IntegrityError as e:
            db.rollback()
            logger.error(f"Integrity error creating user: {e}")
            raise ValueError("User with this email or phone already exists")
        except Exception as e:
            db.rollback()
            logger.error(f"Error creating user: {e}")
            raise
    
    @staticmethod
    def update_user(db: Session, user_id: int, user_update: UserUpdate) -> Optional[User]:
        """Update user profile"""
        try:
            db_user = UserCRUD.get_user(db, user_id)
            if not db_user:
                return None
            
            update_data = user_update.dict(exclude_unset=True)
            for field, value in update_data.items():
                setattr(db_user, field, value)
            
            db_user.updated_at = datetime.now(timezone.utc)
            db.commit()
            db.refresh(db_user)
            
            logger.info(f"User updated successfully: {user_id}")
            return db_user
            
        except Exception as e:
            db.rollback()
            logger.error(f"Error updating user {user_id}: {e}")
            raise
    
    @staticmethod
    def deactivate_user(db: Session, user_id: int) -> bool:
        """Deactivate user account"""
        try:
            db_user = UserCRUD.get_user(db, user_id)
            if not db_user:
                return False
            
            db_user.is_active = False
            db_user.updated_at = datetime.now(timezone.utc)
            db.commit()
            
            logger.info(f"User deactivated: {user_id}")
            return True
            
        except Exception as e:
            db.rollback()
            logger.error(f"Error deactivating user {user_id}: {e}")
            return False
    
    @staticmethod
    def authenticate_user(db: Session, email: str, password: str) -> Optional[User]:
        """Authenticate user login"""
        try:
            user = UserCRUD.get_user_by_email(db, email)
            if not user or not user.verify_password(password):
                return None
            
            user.update_last_login()
            db.commit()
            return user
            
        except Exception as e:
            logger.error(f"Error authenticating user {email}: {e}")
            return None
    
    @staticmethod
    def get_users_by_city(db: Session, city: str, limit: int = 100) -> List[User]:
        """Get users by city (for analytics)"""
        try:
            return db.query(User).filter(
                User.city == city, 
                User.is_active == True
            ).limit(limit).all()
        except Exception as e:
            logger.error(f"Error fetching users by city {city}: {e}")
            return []

# Utility functions
def get_city_cost_index() -> dict:
    """Get cost of living index for Indian cities"""
    return {
        "Mumbai": 1.0, "Delhi": 0.95, "Bangalore": 0.88, "Chennai": 0.82,
        "Pune": 0.78, "Hyderabad": 0.73, "Kolkata": 0.75, "Ahmedabad": 0.70,
        "Jaipur": 0.65, "Lucknow": 0.60, "Kochi": 0.72, "Chandigarh": 0.68,
        "Bhubaneswar": 0.62, "Indore": 0.58, "Nagpur": 0.55, "Coimbatore": 0.67,
        "Visakhapatnam": 0.64
    }
