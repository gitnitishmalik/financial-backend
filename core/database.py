from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import Column, String, DateTime, Text, JSON, Integer, Float
from datetime import datetime
import uuid

from core.config import settings

engine = create_async_engine(settings.DATABASE_URL, echo=settings.DEBUG)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String, unique=True, nullable=False)
    name = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)


class Document(Base):
    __tablename__ = "documents"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, nullable=False)
    filename = Column(String, nullable=False)
    original_name = Column(String)
    file_path = Column(String)
    file_size = Column(Integer)
    status = Column(String, default="pending")  # pending, processing, done, error
    created_at = Column(DateTime, default=datetime.utcnow)


class Analysis(Base):
    __tablename__ = "analyses"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    document_ids = Column(JSON)  # list of doc ids
    user_id = Column(String)
    query = Column(Text)
    result = Column(JSON)
    risk_score = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)


class Alert(Base):
    __tablename__ = "alerts"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String)
    ticker = Column(String)
    condition = Column(String)   # above / below
    threshold = Column(Float)
    active = Column(Integer, default=1)
    notify_email = Column(String, nullable=True)
    last_triggered_at = Column(DateTime, nullable=True)
    last_price = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class ChatMessage(Base):
    """Persisted chat turns so conversations survive process restarts."""
    __tablename__ = "chat_messages"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String, index=True)
    user_id = Column(String, index=True, nullable=True)
    role = Column(String)         # 'user' | 'assistant'
    content = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)


class Memory(Base):
    """Cross-session user facts ('user holds long TSLA', etc.)."""
    __tablename__ = "memories"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, index=True)
    key = Column(String)
    content = Column(Text)
    importance = Column(Integer, default=5)  # 1-10
    created_at = Column(DateTime, default=datetime.utcnow)


class Notification(Base):
    """Persisted alert firings. Streamed to the frontend via SSE."""
    __tablename__ = "notifications"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, index=True)
    alert_id = Column(String, index=True)
    ticker = Column(String)
    condition = Column(String)
    threshold = Column(Float)
    price = Column(Float)
    message = Column(Text)
    read = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)


async def get_db():
    async with AsyncSessionLocal() as session:
        yield session


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await conn.run_sync(_ensure_columns)


# ── lightweight SQLite column-backfill ───────────────────────────
# `Base.metadata.create_all` won't add new columns to pre-existing tables.
# For schema additions we walk each ORM model and ALTER any missing column in.
# Good enough for the demo's SQLite path; Postgres prod should use real migrations.

_TYPE_SQL = {
    Integer: "INTEGER",
    String:  "TEXT",
    Text:    "TEXT",
    Float:   "REAL",
    DateTime: "DATETIME",
    JSON:    "JSON",
}


def _ensure_columns(sync_conn) -> None:
    from sqlalchemy import inspect
    inspector = inspect(sync_conn)
    for table_name, table in Base.metadata.tables.items():
        if not inspector.has_table(table_name):
            continue
        existing = {c["name"] for c in inspector.get_columns(table_name)}
        for column in table.columns:
            if column.name in existing:
                continue
            col_type = _TYPE_SQL.get(type(column.type), "TEXT")
            null_clause = "" if column.nullable else " NOT NULL"
            default = ""
            if column.default is not None and getattr(column.default, "is_scalar", False):
                default = f" DEFAULT {column.default.arg!r}"
            stmt = (
                f"ALTER TABLE {table_name} ADD COLUMN {column.name} {col_type}{null_clause}{default}"
            )
            try:
                sync_conn.exec_driver_sql(stmt)
            except Exception:
                # Some dialects reject DEFAULT on ADD COLUMN — fall back to a no-default ALTER.
                try:
                    sync_conn.exec_driver_sql(
                        f"ALTER TABLE {table_name} ADD COLUMN {column.name} {col_type}"
                    )
                except Exception:
                    pass
