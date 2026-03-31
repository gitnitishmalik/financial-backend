from pydantic_settings import BaseSettings
from typing import Optional
import sys


class Settings(BaseSettings):
    # ── App ────────────────────────────────────────────────────────
    APP_NAME: str = "Finance Analyst"
    DEBUG: bool = False
    SECRET_KEY: str = "finanalyzer-secret-change-in-production"

    # ── Database (SQLite default — zero setup for local/demo) ─────
    DATABASE_URL: str = "sqlite+aiosqlite:///./finanalyzer.db"

    # ── Redis (only needed if you enable Celery task queue) ───────
    REDIS_URL: str = "redis://localhost:6379"

    # ── Groq AI ───────────────────────────────────────────────────
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "llama-3.1-8b-instant"

    # ── Market data (Yahoo Finance is used — no key needed) ───────
    ALPHA_VANTAGE_KEY: Optional[str] = None
    POLYGON_API_KEY: Optional[str] = None

    # ── Email alerts (optional — fill in to enable) ───────────────
    SMTP_HOST: str = "smtp.gmail.com"
    SMTP_PORT: int = 587
    SMTP_USER: Optional[str] = None
    SMTP_PASS: Optional[str] = None

    # ── File upload limits ────────────────────────────────────────
    MAX_FILE_SIZE_MB: int = 50
    UPLOAD_DIR: str = "uploads"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


settings = Settings()


# ── Startup validation ────────────────────────────────────────────
def validate_settings():
    errors = []
    if not settings.GROQ_API_KEY or settings.GROQ_API_KEY == "your-groq-api-key-here":
        errors.append(
            "GROQ_API_KEY is not set.\n"
            "  1. Go to https://console.groq.com\n"
            "  2. Sign up free → API Keys → Create key\n"
            "  3. Add to backend/.env:  GROQ_API_KEY=gsk_..."
        )
    if errors:
        print("\n" + "=" * 60)
        print("  FinAnalyzer — Configuration Error")
        print("=" * 60)
        for e in errors:
            print(f"\n  {e}")
        print("\n" + "=" * 60 + "\n")
        # ── Raise instead of sys.exit so Render shows the full
        #    traceback in logs rather than a silent port-bind failure
        raise RuntimeError(
            "FinAnalyzer startup failed due to missing configuration. "
            "See details above."
        )