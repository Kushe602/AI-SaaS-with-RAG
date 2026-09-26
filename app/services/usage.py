"""Usage metering and plan-limit enforcement.

This is where a real product would integrate Stripe: map a subscription to
``User.plan`` via webhooks, then the limits below gate access. The metering itself
(documents stored, questions per day) is implemented and enforced here.
"""
import datetime as dt

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models import Document, UsageRecord, User


def plan_limits(plan: str) -> dict[str, int]:
    if plan == "pro":
        return {
            "max_documents": settings.pro_max_documents,
            "daily_questions": settings.pro_daily_questions,
        }
    return {
        "max_documents": settings.free_max_documents,
        "daily_questions": settings.free_daily_questions,
    }


async def document_count(db: AsyncSession, owner_id: str) -> int:
    result = await db.execute(
        select(func.count(Document.id)).where(Document.owner_id == owner_id)
    )
    return int(result.scalar_one())


async def questions_today(db: AsyncSession, owner_id: str) -> int:
    start = dt.datetime.now(dt.UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    result = await db.execute(
        select(func.count(UsageRecord.id)).where(
            UsageRecord.owner_id == owner_id,
            UsageRecord.kind == "question",
            UsageRecord.created_at >= start,
        )
    )
    return int(result.scalar_one())


async def can_upload(db: AsyncSession, user: User) -> bool:
    return await document_count(db, user.id) < plan_limits(user.plan)["max_documents"]


async def can_ask(db: AsyncSession, user: User) -> bool:
    return await questions_today(db, user.id) < plan_limits(user.plan)["daily_questions"]


async def record_question(
    db: AsyncSession, owner_id: str, tokens_in: int = 0, tokens_out: int = 0
) -> None:
    db.add(
        UsageRecord(
            owner_id=owner_id, kind="question", tokens_in=tokens_in, tokens_out=tokens_out
        )
    )
    await db.commit()
