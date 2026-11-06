
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.core.config import settings


# PostgreSQL Engine
engine = create_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True,
)


# Database Session
SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
)


# Base class for SQLAlchemy models
class Base(DeclarativeBase):
    pass


# Database dependency for FastAPI
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
