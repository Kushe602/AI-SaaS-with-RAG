"""HTML page routes (landing, dashboard, chat)."""
import json

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.dependencies import get_optional_user
from app.models import Conversation, Document, Message
from app.services import usage
from app.web import templates

router = APIRouter(tags=["pages"])


@router.get("/", response_class=HTMLResponse)
async def landing(request: Request, user=Depends(get_optional_user)):
    if user:
        return RedirectResponse("/app", status_code=303)
    return templates.TemplateResponse(request, "landing.html")


@router.get("/app", response_class=HTMLResponse)
async def dashboard(
    request: Request,
    user=Depends(get_optional_user),
    db: AsyncSession = Depends(get_db),
):
    if not user:
        return RedirectResponse("/login", status_code=303)

    documents = (
        (await db.execute(
            select(Document)
            .where(Document.owner_id == user.id)
            .order_by(Document.created_at.desc())
        )).scalars().all()
    )
    conversations = (
        (await db.execute(
            select(Conversation)
            .where(Conversation.owner_id == user.id)
            .order_by(Conversation.created_at.desc())
        )).scalars().all()
    )
    stats = await _usage_stats(db, user)
    return templates.TemplateResponse(
        request,
        "dashboard.html",
        {
            "user": user,
            "documents": documents,
            "conversations": conversations,
            "stats": stats,
        },
    )


@router.get("/chat/{conversation_id}", response_class=HTMLResponse)
async def chat_page(
    conversation_id: str,
    request: Request,
    user=Depends(get_optional_user),
    db: AsyncSession = Depends(get_db),
):
    if not user:
        return RedirectResponse("/login", status_code=303)

    conversation = (
        await db.execute(select(Conversation).where(Conversation.id == conversation_id))
    ).scalar_one_or_none()
    if not conversation or conversation.owner_id != user.id:
        return RedirectResponse("/app", status_code=303)

    messages = (
        (await db.execute(
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at)
        )).scalars().all()
    )
    conversations = (
        (await db.execute(
            select(Conversation)
            .where(Conversation.owner_id == user.id)
            .order_by(Conversation.created_at.desc())
        )).scalars().all()
    )

    # Resolve the conversation's document scope for the header badge.
    scope_ids = json.loads(conversation.document_ids or "[]")
    if scope_ids:
        names = (
            await db.execute(
                select(Document.filename).where(
                    Document.owner_id == user.id, Document.id.in_(scope_ids)
                )
            )
        ).scalars().all()
        scope_label = f"{len(names)} document" + ("" if len(names) == 1 else "s")
    else:
        scope_label = "All documents"

    return templates.TemplateResponse(
        request,
        "chat.html",
        {
            "user": user,
            "conversation": conversation,
            "messages": messages,
            "conversations": conversations,
            "scope_label": scope_label,
        },
    )


async def _usage_stats(db: AsyncSession, user) -> dict:
    limits = usage.plan_limits(user.plan)
    return {
        "plan": user.plan,
        "documents_used": await usage.document_count(db, user.id),
        "documents_limit": limits["max_documents"],
        "questions_used": await usage.questions_today(db, user.id),
        "questions_limit": limits["daily_questions"],
    }
