# Authentication routes
# backend/api/auth.py

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta

from backend.models import user
from backend.models.database import get_db

SECRET_KEY = "super-secret-key"  # Use env var in production!
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/token")

router = APIRouter(prefix="/api/auth", tags=["auth"])

def verify_password(plain, hashed):
    return pwd_context.verify(plain, hashed)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    db_user = user.get_user_by_email(db, email=email)
    if db_user is None:
        raise credentials_exception
    return db_user

@router.post("/register", response_model=user.UserOut)
def register(user_in: user.UserCreate, db: Session = Depends(get_db)):
    db_user = user.get_user_by_email(db, user_in.email)
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    hashed_pw = get_password_hash(user_in.password)
    user_in_data = user_in.dict()
    user_in_data["password"] = hashed_pw
    return user.create_user(db, user.UserCreate(**user_in_data))

@router.post("/token")
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    db_user = user.get_user_by_email(db, form_data.username)
    if not db_user or not verify_password(form_data.password, db_user.password):
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    access_token = create_access_token(data={"sub": db_user.email}, expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    return {"access_token": access_token, "token_type": "bearer"}

@router.get("/me", response_model=user.UserOut)
def read_users_me(current_user: user.User = Depends(get_current_user)):
    return current_user
