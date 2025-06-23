# backend/api/auth.py

import os
import secrets
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import logging

from fastapi import APIRouter, Depends, HTTPException, status, Request, BackgroundTasks
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from passlib.context import CryptContext
from jose import JWTError, jwt
from pydantic import BaseModel, EmailStr, Field, validator
import redis
from email_validator import validate_email, EmailNotValidError

from backend.models import user
from backend.models.database import get_db
from backend.utils.validators import Validators
from backend.utils.encryption import EncryptionUtils

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
SECRET_KEY = os.getenv("SECRET_KEY", "your-super-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "1440"))  # 24 hours
REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "30"))  # 30 days
RESET_TOKEN_EXPIRE_MINUTES = int(os.getenv("RESET_TOKEN_EXPIRE_MINUTES", "15"))  # 15 minutes

# Rate limiting configuration
MAX_LOGIN_ATTEMPTS = 5
LOGIN_LOCKOUT_DURATION = 3600  # 1 hour in seconds
MAX_REGISTRATION_ATTEMPTS = 3
REGISTRATION_LOCKOUT_DURATION = 1800  # 30 minutes

# Initialize components
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/token")
encryption_utils = EncryptionUtils()
validators = Validators()

# Redis for rate limiting and session management
try:
    redis_client = redis.Redis(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", "6379")),
        db=int(os.getenv("REDIS_DB", "0")),
        decode_responses=True
    )
except Exception as e:
    logger.warning(f"Redis connection failed: {e}. Rate limiting disabled.")
    redis_client = None

router = APIRouter(prefix="/api/auth", tags=["authentication"])

# === PYDANTIC SCHEMAS ===

class UserRegistration(BaseModel):
    """User registration schema with comprehensive validation."""
    
    name: str = Field(..., min_length=2, max_length=64, description="Full name")
    email: EmailStr = Field(..., description="Email address")
    password: str = Field(..., min_length=8, max_length=128, description="Password")
    city: str = Field(..., description="City of residence")
    gender: str = Field(..., description="Gender")
    income: float = Field(..., gt=0, description="Monthly income in INR")
    age: int = Field(..., ge=18, le=100, description="Age")
    phone: Optional[str] = Field(None, description="Phone number")
    accept_terms: bool = Field(..., description="Terms and conditions acceptance")
    
    @validator('name')
    def validate_name(cls, v):
        if not v.strip():
            raise ValueError("Name cannot be empty")
        # Remove extra spaces and validate
        name = ' '.join(v.split())
        if len(name) < 2:
            raise ValueError("Name must be at least 2 characters")
        return name
    
    @validator('email')
    def validate_email_format(cls, v):
        try:
            validate_email(v)
        except EmailNotValidError:
            raise ValueError("Invalid email format")
        return v.lower()
    
    @validator('password')
    def validate_password_strength(cls, v):
        validation_result = validators.validate_password(v)
        if not validation_result['valid']:
            raise ValueError(f"Password validation failed: {', '.join(validation_result['errors'])}")
        return v
    
    @validator('city')
    def validate_city_name(cls, v):
        if not validators.validate_city(v):
            raise ValueError("Invalid city name")
        return v.title()
    
    @validator('gender')
    def validate_gender_value(cls, v):
        if not validators.validate_gender(v):
            raise ValueError("Invalid gender value")
        return v.lower()
    
    @validator('income')
    def validate_income_range(cls, v):
        if not validators.validate_income(v):
            raise ValueError("Income must be between ₹10,000 and ₹50,00,000 per month")
        return v
    
    @validator('age')
    def validate_age_range(cls, v):
        if not validators.validate_age(v):
            raise ValueError("Age must be between 18 and 100")
        return v
    
    @validator('phone')
    def validate_phone_number(cls, v):
        if v and not validators.validate_phone(v):
            raise ValueError("Invalid Indian phone number format")
        return v
    
    @validator('accept_terms')
    def validate_terms_acceptance(cls, v):
        if not v:
            raise ValueError("You must accept the terms and conditions")
        return v

class UserLogin(BaseModel):
    """User login schema."""
    
    email: EmailStr = Field(..., description="Email address")
    password: str = Field(..., description="Password")
    remember_me: Optional[bool] = Field(False, description="Remember login for extended period")

class TokenResponse(BaseModel):
    """Token response schema."""
    
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    user_info: Dict[str, Any]

class PasswordResetRequest(BaseModel):
    """Password reset request schema."""
    
    email: EmailStr = Field(..., description="Email address")

class PasswordReset(BaseModel):
    """Password reset schema."""
    
    token: str = Field(..., description="Reset token")
    new_password: str = Field(..., min_length=8, max_length=128, description="New password")
    
    @validator('new_password')
    def validate_new_password(cls, v):
        validation_result = validators.validate_password(v)
        if not validation_result['valid']:
            raise ValueError(f"Password validation failed: {', '.join(validation_result['errors'])}")
        return v

class PasswordChange(BaseModel):
    """Password change schema."""
    
    current_password: str = Field(..., description="Current password")
    new_password: str = Field(..., min_length=8, max_length=128, description="New password")
    
    @validator('new_password')
    def validate_new_password(cls, v):
        validation_result = validators.validate_password(v)
        if not validation_result['valid']:
            raise ValueError(f"Password validation failed: {', '.join(validation_result['errors'])}")
        return v

# === UTILITY FUNCTIONS ===

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    try:
        return pwd_context.verify(plain_password, hashed_password)
    except Exception as e:
        logger.error(f"Password verification failed: {e}")
        return False

def get_password_hash(password: str) -> str:
    """Generate password hash."""
    try:
        return pwd_context.hash(password)
    except Exception as e:
        logger.error(f"Password hashing failed: {e}")
        raise HTTPException(status_code=500, detail="Password processing failed")

def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token."""
    try:
        to_encode = data.copy()
        expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
        to_encode.update({
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "access"
        })
        return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    except Exception as e:
        logger.error(f"Token creation failed: {e}")
        raise HTTPException(status_code=500, detail="Token generation failed")

def create_refresh_token(data: Dict[str, Any]) -> str:
    """Create JWT refresh token."""
    try:
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
        to_encode.update({
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "refresh"
        })
        return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    except Exception as e:
        logger.error(f"Refresh token creation failed: {e}")
        raise HTTPException(status_code=500, detail="Refresh token generation failed")

def create_reset_token(email: str) -> str:
    """Create password reset token."""
    try:
        data = {
            "sub": email,
            "exp": datetime.utcnow() + timedelta(minutes=RESET_TOKEN_EXPIRE_MINUTES),
            "type": "reset"
        }
        return jwt.encode(data, SECRET_KEY, algorithm=ALGORITHM)
    except Exception as e:
        logger.error(f"Reset token creation failed: {e}")
        raise HTTPException(status_code=500, detail="Reset token generation failed")

def verify_token(token: str, token_type: str = "access") -> Dict[str, Any]:
    """Verify and decode JWT token."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        
        if payload.get("type") != token_type:
            raise JWTError("Invalid token type")
        
        return payload
    except JWTError as e:
        logger.warning(f"Token verification failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )

# === RATE LIMITING ===

def check_rate_limit(key: str, max_attempts: int, window_seconds: int) -> bool:
    """Check if rate limit is exceeded."""
    if not redis_client:
        return True  # Allow if Redis is not available
    
    try:
        current = redis_client.get(key)
        if current is None:
            redis_client.setex(key, window_seconds, 1)
            return True
        elif int(current) < max_attempts:
            redis_client.incr(key)
            return True
        else:
            return False
    except Exception as e:
        logger.error(f"Rate limiting check failed: {e}")
        return True  # Allow on error

def get_client_ip(request: Request) -> str:
    """Get client IP address."""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host

# === AUTHENTICATION DEPENDENCIES ===

async def get_current_user(
    token: str = Depends(oauth2_scheme), 
    db: Session = Depends(get_db)
) -> user.User:
    """Get current authenticated user."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = verify_token(token, "access")
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except HTTPException:
        raise credentials_exception
    
    db_user = user.get_user_by_email(db, email=email)
    if db_user is None:
        raise credentials_exception
    
    # Check if user is active
    if not getattr(db_user, 'is_active', True):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is deactivated"
        )
    
    return db_user

async def get_current_active_user(
    current_user: user.User = Depends(get_current_user)
) -> user.User:
    """Get current active user (additional check)."""
    if not getattr(current_user, 'is_active', True):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is deactivated"
        )
    return current_user

# === API ENDPOINTS ===

@router.post("/register", response_model=user.UserOut, status_code=status.HTTP_201_CREATED)
async def register_user(
    user_data: UserRegistration,
    request: Request,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Register a new user with comprehensive validation and security checks.
    """
    client_ip = get_client_ip(request)
    rate_limit_key = f"register:{client_ip}"
    
    # Check rate limiting
    if not check_rate_limit(rate_limit_key, MAX_REGISTRATION_ATTEMPTS, REGISTRATION_LOCKOUT_DURATION):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many registration attempts. Please try again later."
        )
    
    try:
        # Check if user already exists
        existing_user = user.get_user_by_email(db, user_data.email)
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
        
        # Hash password
        hashed_password = get_password_hash(user_data.password)
        
        # Create user data
        user_create_data = user.UserCreate(
            name=user_data.name,
            email=user_data.email,
            password=hashed_password,
            city=user_data.city,
            gender=user_data.gender,
            income=user_data.income,
            age=user_data.age,
            phone=user_data.phone
        )
        
        # Create user
        new_user = user.create_user(db, user_create_data)
        
        # Log successful registration
        logger.info(f"New user registered: {new_user.email} from IP: {client_ip}")
        
        # TODO: Send welcome email in background
        # background_tasks.add_task(send_welcome_email, new_user.email, new_user.name)
        
        return new_user
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Registration failed for {user_data.email}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed. Please try again."
        )

@router.post("/token", response_model=TokenResponse)
async def login_user(
    form_data: OAuth2PasswordRequestForm = Depends(),
    request: Request = None,
    db: Session = Depends(get_db)
):
    """
    Authenticate user and return access and refresh tokens.
    """
    client_ip = get_client_ip(request) if request else "unknown"
    rate_limit_key = f"login:{client_ip}:{form_data.username}"
    
    # Check rate limiting
    if not check_rate_limit(rate_limit_key, MAX_LOGIN_ATTEMPTS, LOGIN_LOCKOUT_DURATION):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many login attempts. Please try again later."
        )
    
    try:
        # Get user by email
        db_user = user.get_user_by_email(db, form_data.username)
        if not db_user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password"
            )
        
        # Verify password
        if not verify_password(form_data.password, db_user.password):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password"
            )
        
        # Check if user is active
        if not getattr(db_user, 'is_active', True):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="User account is deactivated"
            )
        
        # Create tokens
        token_data = {"sub": db_user.email, "user_id": db_user.id}
        
        # Determine token expiry based on remember_me
        remember_me = getattr(form_data, 'remember_me', False)
        access_token_expires = timedelta(
            minutes=ACCESS_TOKEN_EXPIRE_MINUTES * (7 if remember_me else 1)
        )
        
        access_token = create_access_token(token_data, access_token_expires)
        refresh_token = create_refresh_token(token_data)
        
        # Update last login
        # TODO: Update user's last_login timestamp
        
        # Log successful login
        logger.info(f"User logged in: {db_user.email} from IP: {client_ip}")
        
        # Prepare user info for response
        user_info = {
            "id": db_user.id,
            "name": db_user.name,
            "email": db_user.email,
            "city": db_user.city
        }
        
        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=int(access_token_expires.total_seconds()),
            user_info=user_info
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login failed for {form_data.username}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed. Please try again."
        )

@router.post("/refresh", response_model=TokenResponse)
async def refresh_access_token(
    refresh_token: str,
    db: Session = Depends(get_db)
):
    """
    Refresh access token using refresh token.
    """
    try:
        # Verify refresh token
        payload = verify_token(refresh_token, "refresh")
        email = payload.get("sub")
        
        if not email:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token"
            )
        
        # Get user
        db_user = user.get_user_by_email(db, email)
        if not db_user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found"
            )
        
        # Create new tokens
        token_data = {"sub": db_user.email, "user_id": db_user.id}
        new_access_token = create_access_token(token_data)
        new_refresh_token = create_refresh_token(token_data)
        
        user_info = {
            "id": db_user.id,
            "name": db_user.name,
            "email": db_user.email,
            "city": db_user.city
        }
        
        return TokenResponse(
            access_token=new_access_token,
            refresh_token=new_refresh_token,
            expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            user_info=user_info
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Token refresh failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Token refresh failed"
        )

@router.get("/me", response_model=user.UserOut)
async def get_current_user_info(
    current_user: user.User = Depends(get_current_active_user)
):
    """
    Get current user information.
    """
    return current_user

@router.post("/logout")
async def logout_user(
    current_user: user.User = Depends(get_current_active_user)
):
    """
    Logout user (invalidate token on client side).
    """
    # TODO: Implement token blacklisting if needed
    logger.info(f"User logged out: {current_user.email}")
    return {"message": "Successfully logged out"}

@router.post("/forgot-password")
async def forgot_password(
    reset_request: PasswordResetRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Request password reset.
    """
    try:
        # Check if user exists
        db_user = user.get_user_by_email(db, reset_request.email)
        if not db_user:
            # Don't reveal if email exists or not
            return {"message": "If the email exists, a reset link has been sent"}
        
        # Create reset token
        reset_token = create_reset_token(reset_request.email)
        
        # TODO: Send reset email in background
        # background_tasks.add_task(send_password_reset_email, reset_request.email, reset_token)
        
        logger.info(f"Password reset requested for: {reset_request.email}")
        
        return {"message": "If the email exists, a reset link has been sent"}
        
    except Exception as e:
        logger.error(f"Password reset request failed: {e}")
        return {"message": "If the email exists, a reset link has been sent"}

@router.post("/reset-password")
async def reset_password(
    reset_data: PasswordReset,
    db: Session = Depends(get_db)
):
    """
    Reset password using reset token.
    """
    try:
        # Verify reset token
        payload = verify_token(reset_data.token, "reset")
        email = payload.get("sub")
        
        if not email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid or expired reset token"
            )
        
        # Get user
        db_user = user.get_user_by_email(db, email)
        if not db_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User not found"
            )
        
        # Hash new password
        new_password_hash = get_password_hash(reset_data.new_password)
        
        # Update password
        user.update_user_password(db, db_user.id, new_password_hash)
        
        logger.info(f"Password reset completed for: {email}")
        
        return {"message": "Password reset successful"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Password reset failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Password reset failed"
        )

@router.post("/change-password")
async def change_password(
    password_data: PasswordChange,
    current_user: user.User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Change user password.
    """
    try:
        # Verify current password
        if not verify_password(password_data.current_password, current_user.password):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Current password is incorrect"
            )
        
        # Hash new password
        new_password_hash = get_password_hash(password_data.new_password)
        
        # Update password
        user.update_user_password(db, current_user.id, new_password_hash)
        
        logger.info(f"Password changed for user: {current_user.email}")
        
        return {"message": "Password changed successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Password change failed for user {current_user.email}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Password change failed"
        )

@router.delete("/delete-account")
async def delete_account(
    current_user: user.User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Delete user account (soft delete).
    """
    try:
        # TODO: Implement soft delete
        # user.soft_delete_user(db, current_user.id)
        
        logger.info(f"Account deletion requested for: {current_user.email}")
        
        return {"message": "Account deletion request processed"}
        
    except Exception as e:
        logger.error(f"Account deletion failed for user {current_user.email}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Account deletion failed"
        )

# Health check endpoint
@router.get("/health")
async def auth_health_check():
    """Authentication service health check."""
    return {
        "status": "healthy",
        "service": "authentication",
        "timestamp": datetime.utcnow().isoformat(),
        "redis_connected": redis_client is not None
    }
