import uuid
import asyncio
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends, BackgroundTasks
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel

from core.config import settings
from core.database import get_db, Document, Analysis, Alert
from services.ai_service import AnalysisService
from services.market_service import MarketService
from services.chat_service import ChatService

router = APIRouter()

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

analysis_service = AnalysisService()
market_service = MarketService()
chat_service = ChatService()


# ─────────────────────────── Documents ───────────────────────────

@router.post("/documents/upload")
async def upload_documents(
    files: List[UploadFile] = File(...),
    user_id: str = Form(default="demo-user"),
    db: AsyncSession = Depends(get_db),
):
    uploaded = []
    for file in files:
        if not file.filename.lower().endswith((".pdf", ".xlsx", ".csv")):
            raise HTTPException(400, f"{file.filename}: only PDF, XLSX, CSV allowed")

        doc_id = str(uuid.uuid4())
        safe_name = f"{doc_id}_{file.filename}"
        file_path = UPLOAD_DIR / safe_name

        content = await file.read()
        if len(content) > settings.MAX_FILE_SIZE_MB * 1024 * 1024:
            raise HTTPException(413, f"{file.filename} exceeds {settings.MAX_FILE_SIZE_MB}MB limit")

        file_path.write_bytes(content)

        doc = Document(
            id=doc_id,
            user_id=user_id,
            filename=safe_name,
            original_name=file.filename,
            file_path=str(file_path),
            file_size=len(content),
            status="ready",
        )
        db.add(doc)
        uploaded.append({"id": doc_id, "name": file.filename, "size": len(content)})

    await db.commit()
    return {"uploaded": uploaded, "count": len(uploaded)}


@router.get("/documents")
async def list_documents(
    user_id: str = "demo-user",
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Document).where(Document.user_id == user_id).order_by(Document.created_at.desc())
    )
    docs = result.scalars().all()
    return {"documents": [
        {"id": d.id, "name": d.original_name, "size": d.file_size,
         "status": d.status, "created_at": str(d.created_at)}
        for d in docs
    ]}


@router.delete("/documents/{doc_id}")
async def delete_document(doc_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Document).where(Document.id == doc_id))
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(404, "Document not found")
    if doc.file_path:
        Path(doc.file_path).unlink(missing_ok=True)
    await db.delete(doc)
    await db.commit()
    return {"deleted": doc_id}


# ─────────────────────────── Analysis ───────────────────────────

class AnalysisRequest(BaseModel):
    document_ids: List[str]
    query: str = "Provide a comprehensive investment analysis"
    user_id: str = "demo-user"


@router.post("/analyze")
async def run_analysis(req: AnalysisRequest, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Document).where(Document.id.in_(req.document_ids))
    )
    docs = result.scalars().all()
    if not docs:
        raise HTTPException(404, "No documents found")

    file_paths = [d.file_path for d in docs if d.file_path]
    analysis_result = await analysis_service.analyze(file_paths, req.query)

    record = Analysis(
        document_ids=req.document_ids,
        user_id=req.user_id,
        query=req.query,
        result=analysis_result,
        risk_score=analysis_result.get("risk_score", 5.0),
    )
    db.add(record)
    await db.commit()

    return {"analysis_id": record.id, **analysis_result}


@router.get("/analyses")
async def list_analyses(user_id: str = "demo-user", db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Analysis).where(Analysis.user_id == user_id).order_by(Analysis.created_at.desc()).limit(20)
    )
    analyses = result.scalars().all()
    return {"analyses": [
        {"id": a.id, "query": a.query, "risk_score": a.risk_score,
         "created_at": str(a.created_at), "result": a.result}
        for a in analyses
    ]}


# ─────────────────────────── Chat ───────────────────────────

class ChatRequest(BaseModel):
    message: str
    document_ids: List[str] = []
    session_id: str = "default"


@router.post("/chat")
async def chat_with_docs(req: ChatRequest, db: AsyncSession = Depends(get_db)):
    file_paths = []
    if req.document_ids:
        result = await db.execute(select(Document).where(Document.id.in_(req.document_ids)))
        docs = result.scalars().all()
        file_paths = [d.file_path for d in docs if d.file_path]

    async def stream():
        async for chunk in chat_service.stream_response(req.message, file_paths, req.session_id):
            yield f"data: {chunk}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")


# ─────────────────────────── Market Data ───────────────────────────

@router.get("/market/quote/{ticker}")
async def get_quote(ticker: str):
    data = await market_service.get_quote(ticker.upper())
    return data


@router.get("/market/search")
async def search_ticker(q: str):
    results = await market_service.search(q)
    return {"results": results}


@router.get("/market/history/{ticker}")
async def get_history(ticker: str, period: str = "1mo"):
    data = await market_service.get_history(ticker.upper(), period)
    return data


@router.get("/market/news/{ticker}")
async def get_news(ticker: str):
    news = await market_service.get_news(ticker.upper())
    return {"news": news}


# ─────────────────────────── Alerts ───────────────────────────

class AlertCreate(BaseModel):
    ticker: str
    condition: str  # "above" or "below"
    threshold: float
    user_id: str = "demo-user"


@router.post("/alerts")
async def create_alert(alert: AlertCreate, db: AsyncSession = Depends(get_db)):
    record = Alert(
        user_id=alert.user_id,
        ticker=alert.ticker.upper(),
        condition=alert.condition,
        threshold=alert.threshold,
    )
    db.add(record)
    await db.commit()
    return {"id": record.id, "ticker": record.ticker, "condition": record.condition, "threshold": record.threshold}


@router.get("/alerts")
async def list_alerts(user_id: str = "demo-user", db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Alert).where(Alert.user_id == user_id, Alert.active == 1))
    alerts = result.scalars().all()
    return {"alerts": [
        {"id": a.id, "ticker": a.ticker, "condition": a.condition, "threshold": a.threshold}
        for a in alerts
    ]}


@router.delete("/alerts/{alert_id}")
async def delete_alert(alert_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Alert).where(Alert.id == alert_id))
    alert = result.scalar_one_or_none()
    if not alert:
        raise HTTPException(404, "Alert not found")
    alert.active = 0
    await db.commit()
    return {"deleted": alert_id}
