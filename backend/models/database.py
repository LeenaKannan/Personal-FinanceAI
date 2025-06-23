# backend/models/database.py

import logging
from contextlib import contextmanager
from sqlalchemy import create_engine, event
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
import os
from typing import Generator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./finance_ai.db")

# SQLite specific configuration for development
if DATABASE_URL.startswith("sqlite"):
    engine = create_engine(
        DATABASE_URL,
        connect_args={
            "check_same_thread": False,
            "timeout": 30,
        },
        poolclass=StaticPool,
        echo=os.getenv("DEBUG") == "true"
    )
else:
    # PostgreSQL/MySQL configuration for production
    engine = create_engine(
        DATABASE_URL,
        pool_size=20,
        max_overflow=30,
        pool_recycle=3600,
        pool_pre_ping=True,
        echo=os.getenv("DEBUG") == "true"
    )

# Enable foreign key constraints for SQLite
@event.listens_for(engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    if DATABASE_URL.startswith("sqlite"):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class DatabaseManager:
    """Manages database connections and transactions"""
    
    @staticmethod
    def get_db() -> Generator[Session, None, None]:
        """Dependency for FastAPI to get database session"""
        db = SessionLocal()
        try:
            yield db
        except Exception as e:
            logger.error(f"Database error: {e}")
            db.rollback()
            raise
        finally:
            db.close()
    
    @staticmethod
    @contextmanager
    def get_db_context():
        """Context manager for database operations"""
        db = SessionLocal()
        try:
            yield db
            db.commit()
        except Exception as e:
            logger.error(f"Database transaction error: {e}")
            db.rollback()
            raise
        finally:
            db.close()
    
    @staticmethod
    def create_tables():
        """Create all database tables"""
        try:
            Base.metadata.create_all(bind=engine)
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Error creating tables: {e}")
            raise
    
    @staticmethod
    def drop_tables():
        """Drop all database tables (use with caution!)"""
        try:
            Base.metadata.drop_all(bind=engine)
            logger.info("Database tables dropped successfully")
        except Exception as e:
            logger.error(f"Error dropping tables: {e}")
            raise

# Convenience function for backward compatibility
def get_db():
    return DatabaseManager.get_db()

# Health check function
def check_db_health() -> bool:
    """Check if database is accessible"""
    try:
        with DatabaseManager.get_db_context() as db:
            db.execute("SELECT 1")
        return True
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return False
