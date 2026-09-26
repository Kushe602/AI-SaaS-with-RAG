"""Chat routes: create conversations, ask questions, stream grounded answers (SSE)."""
import json
from html import escape

from fastapi import APIRouter, Depends, Form, Request, Response, status
from fastapi.responses import HTMLResponse, StreamingResponse
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import SessionLocal, get_db
from app.dependencies import get_current_user
from app.models import Conversation, Document, Message
from app.services import llm, retrieval, usage
from app.web import templates

router = APIRouter(prefix="/chat", tags=["chat"])


def _sse(event: str, data: str) -> str:
    """Format one Server-Sent Event. Data is JSON-encoded so it is always single-line-safe."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _render_sources(citations: list[dict]) -> str:
    if not citations:
        return ""
    items = "".join(
        f'<li><span class="font-medium text-slate-600">[{c["n"]}] {escape(c["filename"])}</span> '
        f'<span class="text-slate-400">— {escape(c["snippet"])}…</span></li>'
        for c in citations
    )
    return (
        '<div class="mt-2 border-t border-slate-100 pt-2 text-xs">'
        '<p class="mb-1 font-semibold uppercase tracking-wide text-slate-400">Sources</p>'
        f'<ol class="space-y-1">{items}</ol></div>'
    )


@router.post("/new")
async def new_conversation(user=Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    conversation = Conversation(owner_id=user.id, title="New chat")
    db.add(conversation)
    await db.commit()
    await db.refresh(conversation)
    response = Response(status_code=status.HTTP_204_NO_CONTENT)
    response.headers["HX-Redirect"] = f"/chat/{conversation.id}"
    return response


@router.post("/{conversation_id}/ask", response_class=HTMLResponse)
async def ask(
    conversation_id: str,
    request: Request,
    content: str = Form(...),
    user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    conversation = (
        await db.execute(select(Conversation).where(Conversation.id == conversation_id))
    ).scalar_one_or_none()
    if not conversation or conversation.owner_id != user.id:
        return HTMLResponse("", status_code=404)

    content = content.strip()
    if not content:
        return HTMLResponse("")

    if not await usage.can_ask(db, user):
        return templates.TemplateResponse(
            request,
            "partials/limit.html",
            {"limit": usage.plan_limits(user.plan)["daily_questions"]},
        )

    existing = await db.execute(
        select(func.count(Message.id)).where(Message.conversation_id == conversation_id)
    )
    if existing.scalar_one() == 0:
        conversation.title = content[:60]

    user_message = Message(conversation_id=conversation_id, role="user", content=content)
    db.add(user_message)
    await db.commit()

    assistant_message = Message(conversation_id=conversation_id, role="assistant", content="")
    db.add(assistant_message)
    await db.commit()
    await db.refresh(assistant_message)

    return templates.TemplateResponse(
        request,
        "partials/exchange.html",
        {
            "conversation_id": conversation_id,
            "user_content": content,
            "assistant_id": assistant_message.id,
        },
    )


@router.get("/{conversation_id}/stream/{message_id}")
async def stream(conversation_id: str, message_id: str, user=Depends(get_current_user)):
    async def generate():
        async with SessionLocal() as db:
            assistant = (
                await db.execute(select(Message).where(Message.id == message_id))
            ).scalar_one_or_none()
            conversation = (
                await db.execute(select(Conversation).where(Conversation.id == conversation_id))
            ).scalar_one_or_none()
            if (
                not assistant
                or not conversation
                or conversation.owner_id != user.id
                or assistant.conversation_id != conversation_id
            ):
                yield _sse("done", "1")
                return

            question_row = (
                (await db.execute(
                    select(Message)
                    .where(Message.conversation_id == conversation_id, Message.role == "user")
                    .order_by(Message.created_at.desc())
                )).scalars().first()
            )
            question = question_row.content if question_row else ""

            scored = await retrieval.search(db, user.id, question, settings.max_context_chunks)
            chunks = [chunk for chunk, _ in scored]

            filenames: dict[str, str] = {}
            doc_ids = list({c.document_id for c in chunks})
            if doc_ids:
                rows = (
                    await db.execute(select(Document).where(Document.id.in_(doc_ids)))
                ).scalars().all()
                filenames = {d.id: d.filename for d in rows}

            pieces: list[str] = []
            async for token in llm.stream_answer(question, chunks):
                pieces.append(token)
                yield _sse("token", token)

            citations = [
                {
                    "n": i + 1,
                    "filename": filenames.get(c.document_id, "document"),
                    "snippet": c.content[:200],
                }
                for i, c in enumerate(chunks)
            ]
            sources_html = _render_sources(citations)
            if sources_html:
                yield _sse("sources", sources_html)

            assistant.content = "".join(pieces)
            assistant.citations = json.dumps(citations)
            await db.commit()
            await usage.record_question(db, user.id)
            yield _sse("done", "1")

    headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    return StreamingResponse(generate(), media_type="text/event-stream", headers=headers)
