# backend/models/user.py

from sqlalchemy import Column, Integer, String, Float
from sqlalchemy.orm import Session
from pydantic import BaseModel, EmailStr
from backend.models.database import Base

# SQLAlchemy model
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(64), nullable=False)
    email = Column(String(120), unique=True, index=True, nullable=False)
    city = Column(String(32), nullable=False)
    gender = Column(String(16), nullable=False)
    income = Column(Float, nullable=False)
    age = Column(Integer, nullable=False)

# Pydantic schemas
class UserBase(BaseModel):
    name: str
    email: EmailStr
    city: str
    gender: str
    income: float
    age: int

class UserCreate(UserBase):
    pass

class UserOut(UserBase):
    id: int
    class Config:
        orm_mode = True

# CRUD functions
def get_user(db: Session, user_id: int):
    return db.query(User).filter(User.id == user_id).first()

def get_user_by_email(db: Session, email: str):
    return db.query(User).filter(User.email == email).first()

def create_user(db: Session, user_in: UserCreate):
    db_user = User(**user_in.dict())
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user
