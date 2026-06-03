"""Celery app + beat schedule.

Run the worker with:
    celery -A celery_app worker --loglevel=info
And the scheduler with:
    celery -A celery_app beat --loglevel=info
"""

from celery import Celery

from core.config import settings


celery_app = Celery(
    "finanalyzer",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
    include=["tasks.alerts"],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
)

# ── Beat schedule ─────────────────────────────────────────────────
celery_app.conf.beat_schedule = {
    "evaluate-alerts-every-5-min": {
        "task": "tasks.alerts.evaluate_alerts",
        "schedule": 300.0,  # every 5 minutes
    },
}
