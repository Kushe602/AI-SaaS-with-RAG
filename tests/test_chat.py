import re

from sqlalchemy import select

from app.database import SessionLocal
from app.models import Message
from app.services import llm


async def _fake_stream(question, chunks):
    for token in ["Indexing ", "uses ", "B-trees ", "[1]"]:
        yield token


async def test_full_chat_flow(client, monkeypatch):
    monkeypatch.setattr(llm, "stream_answer", _fake_stream)

    await client.post("/register", data={"email": "chat@example.com", "password": "password123"})
    files = {"file": ("db.txt", b"Database indexing improves query performance.", "text/plain")}
    await client.post("/documents/upload", files=files)

    new_convo = await client.post("/chat/new")
    conversation_id = new_convo.headers["HX-Redirect"].split("/chat/")[1]

    ask = await client.post(
        f"/chat/{conversation_id}/ask", data={"content": "How does indexing work?"}
    )
    assert ask.status_code == 200
    match = re.search(r'data-mid="([0-9a-f]+)"', ask.text)
    assert match, "assistant message id not found in exchange partial"
    message_id = match.group(1)

    stream = await client.get(f"/chat/{conversation_id}/stream/{message_id}")
    assert stream.status_code == 200
    assert "event: token" in stream.text
    assert "B-trees" in stream.text
    assert "event: done" in stream.text

    async with SessionLocal() as db:
        assistant = (
            await db.execute(select(Message).where(Message.id == message_id))
        ).scalar_one()
        assert "B-trees" in assistant.content


async def test_ask_requires_auth(client):
    response = await client.post("/chat/does-not-exist/ask", data={"content": "hi"})
    assert response.status_code == 401
