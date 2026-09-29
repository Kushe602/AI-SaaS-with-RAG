"""Chat routes: create conversations, ask questions, stream grounded answers (SSE)."""
import json

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


@router.post("/new")
async def new_conversation(
    document_ids: list[str] = Form(default=[]),
    user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    # Keep only ids that are real documents owned by this user; empty = search all.
    scope: list[str] = []
    if document_ids:
        owned = (
            await db.execute(
                select(Document.id).where(
                    Document.owner_id == user.id, Document.id.in_(document_ids)
                )
            )
        ).scalars().all()
        scope = list(owned)

    conversation = Conversation(
        owner_id=user.id, title="New chat", document_ids=json.dumps(scope)
    )
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

            # Load the full conversation so we can carry prior turns as context.
            messages = (
                (await db.execute(
                    select(Message)
                    .where(Message.conversation_id == conversation_id)
                    .order_by(Message.created_at)
                )).scalars().all()
            )
            user_messages = [m for m in messages if m.role == "user"]
            question = user_messages[-1].content if user_messages else ""
            cutoff = user_messages[-1].created_at if user_messages else None
            history = [
                (m.role, m.content)
                for m in messages
                if cutoff is not None and m.created_at < cutoff and m.content.strip()
            ]
            prior_questions = [content for role, content in history if role == "user"]

            # Scope retrieval to the conversation's chosen documents (empty = all),
            # and fold recent turns into the query so follow-ups resolve.
            scope = json.loads(conversation.document_ids or "[]")
            search_query = retrieval.condense_query(prior_questions, question)
            scored = await retrieval.search(
                db,
                user.id,
                search_query,
                settings.max_context_chunks,
                document_ids=scope or None,
            )
            chunks = [chunk for chunk, _ in scored]

            filenames: dict[str, str] = {}
            doc_ids = list({c.document_id for c in chunks})
            if doc_ids:
                rows = (
                    await db.execute(select(Document).where(Document.id.in_(doc_ids)))
                ).scalars().all()
                filenames = {d.id: d.filename for d in rows}

            pieces: list[str] = []
            async for token in llm.stream_answer(question, chunks, history=history):
                pieces.append(token)
                yield _sse("token", token)

            citations = [
                {
                    "n": i + 1,
                    "filename": filenames.get(c.document_id, "document"),
                    "snippet": " ".join(c.content.split())[:200],
                    "passage": " ".join(c.content.split())[:1200],
                }
                for i, c in enumerate(chunks)
            ]
            if citations:
                sources_html = templates.env.get_template("partials/sources.html").render(
                    citations=citations, mid=message_id
                )
                yield _sse("sources", sources_html)

            assistant.content = "".join(pieces)
            assistant.citations = json.dumps(citations)
            await db.commit()
            await usage.record_question(db, user.id)
            yield _sse("done", "1")

    headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    return StreamingResponse(generate(), media_type="text/event-stream", headers=headers)
