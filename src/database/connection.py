"""
Database Connection Management for Sentinel-AI

Provides SQLAlchemy engine, session management, and database initialization.
Supports both PostgreSQL and SQLite (fallback).
"""

import os
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool

from .models import Base


# Get database URL from environment or use SQLite fallback
DATABASE_URL = os.getenv(
    "DATABASE_URL", 
    "sqlite:///./sentinel_ai.db"
)

# Configure engine based on database type
if DATABASE_URL.startswith("sqlite"):
    # SQLite-specific configuration
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        echo=False  # Set to True for SQL debugging
    )
    
    # Enable foreign key support for SQLite
    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()
else:
    # PostgreSQL configuration
    engine = create_engine(
        DATABASE_URL,
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True,
        echo=False
    )

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db() -> None:
    """
    Initialize the database by creating all tables.
    Safe to call multiple times - only creates tables that don't exist.
    """
    Base.metadata.create_all(bind=engine)
    print("✅ Database initialized successfully!")


def drop_db() -> None:
    """
    Drop all tables. USE WITH CAUTION!
    Only for development/testing purposes.
    """
    Base.metadata.drop_all(bind=engine)
    print("⚠️ All database tables dropped!")


@contextmanager
def get_session() -> Generator[Session, None, None]:
    """
    Context manager for database sessions.
    Automatically handles commit/rollback and session cleanup.
    
    Usage:
        with get_session() as session:
            audit = ContentAudit(...)
            session.add(audit)
    """
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_db() -> Generator[Session, None, None]:
    """
    Dependency injection for FastAPI-style usage.
    Yields a database session for use in route handlers.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class DatabaseManager:
    """
    High-level database operations manager.
    Provides convenient methods for common database operations.
    """
    
    @staticmethod
    def health_check() -> bool:
        """Check if database connection is healthy."""
        try:
            with get_session() as session:
                session.execute("SELECT 1")
            return True
        except Exception as e:
            print(f"❌ Database health check failed: {e}")
            return False
    
    @staticmethod
    def get_table_counts() -> dict:
        """Get row counts for all tables."""
        from .models import ContentAudit, PolicyViolation, HumanReview, KPIMetrics
        
        with get_session() as session:
            return {
                "content_audits": session.query(ContentAudit).count(),
                "policy_violations": session.query(PolicyViolation).count(),
                "human_reviews": session.query(HumanReview).count(),
                "kpi_metrics": session.query(KPIMetrics).count()
            }
    
    @staticmethod
    def clear_all_data() -> None:
        """Clear all data while preserving table structure."""
        from .models import ContentAudit, PolicyViolation, HumanReview, KPIMetrics
        
        with get_session() as session:
            session.query(PolicyViolation).delete()
            session.query(HumanReview).delete()
            session.query(ContentAudit).delete()
            session.query(KPIMetrics).delete()
        print("🗑️ All data cleared!")


# Auto-initialize database on import (for development convenience)
if os.getenv("AUTO_INIT_DB", "true").lower() == "true":
    try:
        init_db()
    except Exception as e:
        print(f"⚠️ Auto-init database failed: {e}")
