"""
AuraDiff Database — SQLAlchemy + SQLite
Tables: users, comparison_history
"""

import os
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, Text, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

DB_PATH = os.getenv("DATABASE_URL", "sqlite:///auradiff.db")

engine = create_engine(DB_PATH, connect_args={"check_same_thread": False} if "sqlite" in DB_PATH else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    history = relationship("ComparisonHistory", back_populates="user")


class ComparisonHistory(Base):
    __tablename__ = "comparison_history"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    comparison_type = Column(String(20), nullable=False)  # "pairwise" or "batch"
    filenames = Column(Text, nullable=False)               # JSON list of filenames
    total_score = Column(Float, nullable=True)
    risk_level = Column(String(20), nullable=True)
    result_json = Column(Text, nullable=False)              # Full result as JSON
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="history")


def init_db():
    """Create all tables if they don't exist."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """FastAPI dependency — yields a DB session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
