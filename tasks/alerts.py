"""Background alert evaluator.

Runs every 5 minutes via celery beat. For each active alert:
  1. Fetch the live price for the ticker.
  2. If the condition is met AND we haven't already triggered in the last
     `COOLDOWN_MIN` minutes, persist a Notification + (optionally) email
     the user.

Notifications land in the `notifications` table, which the SSE endpoint
`/api/v1/notifications/stream` reads from.
"""

import asyncio
import logging
import smtplib
from datetime import datetime, timedelta
from email.message import EmailMessage
from typing import Dict, List

from sqlalchemy import select

from celery_app import celery_app
from core.config import settings
from core.database import AsyncSessionLocal, Alert, Notification
from services.market_service import MarketService


log = logging.getLogger(__name__)

# Don't re-trigger the same alert within this window
COOLDOWN_MIN = 60


def _condition_met(condition: str, price: float, threshold: float) -> bool:
    condition = (condition or "").lower()
    if condition == "above":
        return price >= threshold
    if condition == "below":
        return price <= threshold
    return False


async def _evaluate_once() -> Dict:
    market = MarketService()
    triggered: List[Dict] = []
    skipped = 0

    async with AsyncSessionLocal() as session:
        result = await session.execute(select(Alert).where(Alert.active == 1))
        alerts = result.scalars().all()

        tickers = sorted({a.ticker for a in alerts if a.ticker})
        if not tickers:
            return {"triggered": 0, "skipped": 0, "checked": 0}
        quotes = await market.get_multiple_quotes(tickers)
        price_map = {q.get("ticker"): q.get("price") for q in quotes}

        now = datetime.utcnow()
        cooldown = timedelta(minutes=COOLDOWN_MIN)

        for alert in alerts:
            price = price_map.get(alert.ticker)
            if price is None:
                skipped += 1
                continue
            if not _condition_met(alert.condition, price, alert.threshold):
                alert.last_price = price
                continue
            if alert.last_triggered_at and (now - alert.last_triggered_at) < cooldown:
                continue

            message = (
                f"{alert.ticker} crossed {alert.condition} {alert.threshold:.2f} "
                f"— current price {price:.2f}."
            )
            notif = Notification(
                user_id=alert.user_id,
                alert_id=alert.id,
                ticker=alert.ticker,
                condition=alert.condition,
                threshold=alert.threshold,
                price=price,
                message=message,
            )
            session.add(notif)
            alert.last_triggered_at = now
            alert.last_price = price

            if alert.notify_email and settings.SMTP_USER and settings.SMTP_PASS:
                try:
                    _send_email(alert.notify_email, message)
                except Exception as e:
                    log.warning("alert email failed: %s", e)

            triggered.append({
                "alert_id": alert.id,
                "ticker": alert.ticker,
                "price": price,
                "threshold": alert.threshold,
                "condition": alert.condition,
            })

        await session.commit()

    return {"triggered": len(triggered), "skipped": skipped, "checked": len(alerts), "fires": triggered}


def _send_email(to: str, body: str) -> None:
    msg = EmailMessage()
    msg["Subject"] = "FinAnalyzer Pro · Price Alert"
    msg["From"] = settings.SMTP_USER
    msg["To"] = to
    msg.set_content(body)
    with smtplib.SMTP(settings.SMTP_HOST, settings.SMTP_PORT) as s:
        s.starttls()
        s.login(settings.SMTP_USER, settings.SMTP_PASS)
        s.send_message(msg)


@celery_app.task(name="tasks.alerts.evaluate_alerts")
def evaluate_alerts() -> Dict:
    """Sync entry point Celery actually calls. Bridges into the async core."""
    try:
        return asyncio.run(_evaluate_once())
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(_evaluate_once())
        finally:
            loop.close()
