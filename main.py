import asyncio
import uuid
import os
import shutil
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import uvicorn

from api.routes import router
from core.config import settings, validate_settings
from core.database import init_db

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

validate_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    print(f"\n  FinAnalyzer Pro is running")
    print(f"  LLM  : Groq / {settings.GROQ_MODEL}")
    print(f"  DB   : {settings.DATABASE_URL.split('///')[0]}")
    print(f"  Docs : http://localhost:8000/docs\n")
    yield
    shutil.rmtree(UPLOAD_DIR, ignore_errors=True)
    UPLOAD_DIR.mkdir(exist_ok=True)


def get_allowed_origins() -> list[str]:
    origins = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ]

    frontend_url = os.getenv("FRONTEND_URL", "").strip()
    if frontend_url:
        origins.append(frontend_url)

    extra_origins = os.getenv("EXTRA_ORIGINS", "").strip()
    if extra_origins:
        for origin in extra_origins.split(","):
            origin = origin.strip()
            if origin:
                origins.append(origin)

    return origins


app = FastAPI(
    title="FinAnalyzer Pro API",
    description="AI-powered financial document analysis — powered by Groq",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_allowed_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")


@app.get("/")
async def root():
    return {
        "app": "FinAnalyzer Pro API",
        "version": "2.0.0",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "version": "2.0.0",
        "model": settings.GROQ_MODEL,
        "groq_configured": bool(settings.GROQ_API_KEY),
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)