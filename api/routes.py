import uuid
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel

from core.config import settings
from core.database import (
    get_db, Document, Analysis, Alert, Notification,
    ChatMessage, Memory, AsyncSessionLocal,
)
from services.analysis_service import AnalysisService   # ← was ai_service (bug fix)
from services.market_service import MarketService
from services.chat_service import ChatService

router = APIRouter()

UPLOAD_DIR = Path(settings.UPLOAD_DIR)
UPLOAD_DIR.mkdir(exist_ok=True)

# One instance per process — all share the same RAG embedder
analysis_service = AnalysisService()
market_service   = MarketService()
chat_service     = ChatService()


# ─────────────────────────── Documents ───────────────────────────

@router.post("/documents/upload")
async def upload_documents(
    files: List[UploadFile] = File(...),
    user_id: str = Form(default="demo-user"),
    db: AsyncSession = Depends(get_db),
):
    uploaded = []
    for file in files:
        if not file.filename.lower().endswith((".pdf", ".xlsx", ".xls", ".csv")):
            raise HTTPException(400, f"{file.filename}: only PDF, XLSX, XLS, CSV allowed")

        doc_id   = str(uuid.uuid4())
        safe_name = f"{doc_id}_{file.filename}"
        file_path = UPLOAD_DIR / safe_name

        content = await file.read()
        if len(content) > settings.MAX_FILE_SIZE_MB * 1024 * 1024:
            raise HTTPException(413, f"{file.filename} exceeds {settings.MAX_FILE_SIZE_MB} MB limit")

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
        select(Document)
        .where(Document.user_id == user_id)
        .order_by(Document.created_at.desc())
    )
    docs = result.scalars().all()
    return {"documents": [
        {
            "id": d.id,
            "name": d.original_name,
            "size": d.file_size,
            "status": d.status,
            "created_at": str(d.created_at),
        }
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


# ─────────────────────────── Analysis ────────────────────────────

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

    file_paths     = [d.file_path for d in docs if d.file_path]
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
async def list_analyses(
    user_id: str = "demo-user",
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Analysis)
        .where(Analysis.user_id == user_id)
        .order_by(Analysis.created_at.desc())
        .limit(20)
    )
    analyses = result.scalars().all()
    return {"analyses": [
        {
            "id": a.id,
            "query": a.query,
            "risk_score": a.risk_score,
            "created_at": str(a.created_at),
            "result": a.result,
        }
        for a in analyses
    ]}


# ─────────────────────────── Chat ────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    document_ids: List[str] = []
    session_id: str = "default"
    user_id: str = "demo-user"


@router.post("/chat")
async def chat_with_docs(req: ChatRequest, db: AsyncSession = Depends(get_db)):
    file_paths: List[str] = []
    if req.document_ids:
        result = await db.execute(
            select(Document).where(Document.id.in_(req.document_ids))
        )
        docs = result.scalars().all()
        file_paths = [d.file_path for d in docs if d.file_path]

    async def stream():
        async for chunk in chat_service.stream_response(
            req.message, file_paths, req.session_id, user_id=req.user_id,
        ):
            yield f"data: {chunk}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")


@router.get("/chat/history/{session_id}")
async def chat_history(session_id: str, db: AsyncSession = Depends(get_db)):
    """Return all persisted turns for a session — used to rehydrate the FE."""
    q = (
        select(ChatMessage)
        .where(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.created_at.asc())
    )
    rows = (await db.execute(q)).scalars().all()
    return {"messages": [
        {"role": r.role, "content": r.content, "created_at": str(r.created_at)}
        for r in rows
    ]}


@router.delete("/chat/history/{session_id}")
async def clear_chat_history(session_id: str, db: AsyncSession = Depends(get_db)):
    from sqlalchemy import delete
    await db.execute(delete(ChatMessage).where(ChatMessage.session_id == session_id))
    await db.commit()
    return {"cleared": session_id}


# ─────────────────────────── Memory ──────────────────────────────

class MemoryCreate(BaseModel):
    user_id: str = "demo-user"
    key: str
    content: str
    importance: int = 5


@router.post("/memories")
async def create_memory(mem: MemoryCreate, db: AsyncSession = Depends(get_db)):
    record = Memory(
        user_id=mem.user_id,
        key=mem.key,
        content=mem.content,
        importance=max(1, min(10, mem.importance)),
    )
    db.add(record)
    await db.commit()
    return {"id": record.id, "key": record.key, "importance": record.importance}


@router.get("/memories")
async def list_memories(user_id: str = "demo-user", db: AsyncSession = Depends(get_db)):
    q = (
        select(Memory)
        .where(Memory.user_id == user_id)
        .order_by(Memory.importance.desc(), Memory.created_at.desc())
    )
    rows = (await db.execute(q)).scalars().all()
    return {"memories": [
        {"id": m.id, "key": m.key, "content": m.content,
         "importance": m.importance, "created_at": str(m.created_at)}
        for m in rows
    ]}


@router.delete("/memories/{mem_id}")
async def delete_memory(mem_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Memory).where(Memory.id == mem_id))
    mem = result.scalar_one_or_none()
    if not mem:
        raise HTTPException(404, "Memory not found")
    await db.delete(mem)
    await db.commit()
    return {"deleted": mem_id}


# ─────────────────────────── Market Data ─────────────────────────

@router.get("/market/quote/{ticker}")
async def get_quote(ticker: str):
    return await market_service.get_quote(ticker.upper())


@router.get("/market/search")
async def search_ticker(q: str):
    results = await market_service.search(q)
    return {"results": results}


@router.get("/market/history/{ticker}")
async def get_history(ticker: str, period: str = "1mo"):
    return await market_service.get_history(ticker.upper(), period)


@router.get("/market/news/{ticker}")
async def get_news(ticker: str):
    news = await market_service.get_news(ticker.upper())
    return {"news": news}


# ─────────────────────────── Alerts ──────────────────────────────

class AlertCreate(BaseModel):
    ticker: str
    condition: str      # "above" or "below"
    threshold: float
    user_id: str = "demo-user"
    notify_email: Optional[str] = None


@router.post("/alerts")
async def create_alert(alert: AlertCreate, db: AsyncSession = Depends(get_db)):
    record = Alert(
        user_id=alert.user_id,
        ticker=alert.ticker.upper(),
        condition=alert.condition,
        threshold=alert.threshold,
        notify_email=alert.notify_email,
    )
    db.add(record)
    await db.commit()
    return {
        "id": record.id,
        "ticker": record.ticker,
        "condition": record.condition,
        "threshold": record.threshold,
        "notify_email": record.notify_email,
    }


@router.get("/alerts")
async def list_alerts(
    user_id: str = "demo-user",
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Alert).where(Alert.user_id == user_id, Alert.active == 1)
    )
    alerts = result.scalars().all()
    return {"alerts": [
        {
            "id": a.id, "ticker": a.ticker, "condition": a.condition,
            "threshold": a.threshold, "notify_email": a.notify_email,
            "last_price": a.last_price,
            "last_triggered_at": str(a.last_triggered_at) if a.last_triggered_at else None,
        }
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


# ─────────────────────────── Notifications ───────────────────────────

@router.get("/notifications")
async def list_notifications(
    user_id: str = "demo-user",
    unread_only: bool = False,
    db: AsyncSession = Depends(get_db),
):
    q = select(Notification).where(Notification.user_id == user_id)
    if unread_only:
        q = q.where(Notification.read == 0)
    q = q.order_by(Notification.created_at.desc()).limit(50)
    result = await db.execute(q)
    notifs = result.scalars().all()
    return {"notifications": [
        {
            "id": n.id, "alert_id": n.alert_id, "ticker": n.ticker,
            "condition": n.condition, "threshold": n.threshold,
            "price": n.price, "message": n.message, "read": bool(n.read),
            "created_at": str(n.created_at),
        }
        for n in notifs
    ]}


@router.post("/notifications/{notif_id}/read")
async def mark_notification_read(notif_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Notification).where(Notification.id == notif_id))
    notif = result.scalar_one_or_none()
    if not notif:
        raise HTTPException(404, "Notification not found")
    notif.read = 1
    await db.commit()
    return {"id": notif.id, "read": True}


@router.get("/notifications/stream")
async def notifications_stream(user_id: str = "demo-user"):
    """SSE stream — pushes new Notification rows for this user as they appear.

    3-second DB poll so it works without Redis pub/sub. The Celery alert
    evaluator writes Notification rows; this endpoint sees them on the next
    tick and pushes them down the wire.
    """
    import asyncio as _asyncio
    import json as _json
    from datetime import datetime as _dt

    async def _stream():
        last_seen: Optional[_dt] = _dt.utcnow()
        yield f"data: {_json.dumps({'event': 'open', 'user_id': user_id})}\n\n"
        while True:
            async with AsyncSessionLocal() as session:
                q = (
                    select(Notification)
                    .where(Notification.user_id == user_id, Notification.created_at > last_seen)
                    .order_by(Notification.created_at.asc())
                )
                result = await session.execute(q)
                new = result.scalars().all()
            for n in new:
                last_seen = n.created_at
                payload = {
                    "event": "alert",
                    "id": n.id, "alert_id": n.alert_id,
                    "ticker": n.ticker, "condition": n.condition,
                    "threshold": n.threshold, "price": n.price,
                    "message": n.message,
                    "created_at": str(n.created_at),
                }
                yield f"data: {_json.dumps(payload)}\n\n"
            yield ": keepalive\n\n"
            await _asyncio.sleep(3)

    return StreamingResponse(_stream(), media_type="text/event-stream")