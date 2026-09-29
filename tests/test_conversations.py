"""End-to-end tests for per-conversation document scoping and multi-turn follow-ups.

Both run under the keyless demo LLM (USE_FAKE_LLM) so they are deterministic and
need no API key or network.
"""
import json
import re

from sqlalchemy import select

from app.database import SessionLocal
from app.models import Conversation, Document
from app.services import llm


async def _register(client, email):
    await client.post("/register", data={"email": email, "password": "password123"})


async def _upload(client, filename, text):
    await client.post(
        "/documents/upload", files={"file": (filename, text, "text/plain")}
    )


def _assistant_id(ask_html):
    match = re.search(r'data-mid="([0-9a-f]+)"', ask_html)
    assert match, "assistant message id not found in exchange partial"
    return match.group(1)


async def test_new_conversation_stores_selected_scope(client):
    await _register(client, "scope1@example.com")
    await _upload(client, "cats.txt", b"Cats are small furry animals that purr.")
    await _upload(client, "db.txt", b"Database indexing improves query performance.")

    async with SessionLocal() as db:
        docs = (await db.execute(select(Document))).scalars().all()
    db_doc = next(d for d in docs if d.filename == "db.txt")

    new = await client.post("/chat/new", data={"document_ids": [db_doc.id]})
    conversation_id = new.headers["HX-Redirect"].split("/chat/")[1]

    async with SessionLocal() as db:
        convo = (
            await db.execute(select(Conversation).where(Conversation.id == conversation_id))
        ).scalar_one()
        assert json.loads(convo.document_ids) == [db_doc.id]


async def test_new_conversation_defaults_to_all_documents(client):
    await _register(client, "scope2@example.com")
    await _upload(client, "cats.txt", b"Cats are small furry animals that purr.")

    new = await client.post("/chat/new")  # no document_ids -> search all
    conversation_id = new.headers["HX-Redirect"].split("/chat/")[1]

    async with SessionLocal() as db:
        convo = (
            await db.execute(select(Conversation).where(Conversation.id == conversation_id))
        ).scalar_one()
        assert json.loads(convo.document_ids) == []


async def test_scoped_conversation_only_retrieves_in_scope(client, monkeypatch):
    monkeypatch.setattr(llm.settings, "use_fake_llm", True)
    await _register(client, "scope3@example.com")
    await _upload(client, "cats.txt", b"Cats are small furry animals that purr and chase mice.")
    await _upload(client, "db.txt", b"Database indexing improves query performance with B-trees.")

    async with SessionLocal() as db:
        docs = (await db.execute(select(Document))).scalars().all()
    db_doc = next(d for d in docs if d.filename == "db.txt")

    new = await client.post("/chat/new", data={"document_ids": [db_doc.id]})
    conversation_id = new.headers["HX-Redirect"].split("/chat/")[1]

    # Ask about the OTHER document's topic; scoping must keep it out of the answer.
    ask = await client.post(
        f"/chat/{conversation_id}/ask", data={"content": "Tell me about furry animals"}
    )
    message_id = _assistant_id(ask.text)
    stream = await client.get(f"/chat/{conversation_id}/stream/{message_id}")

    assert stream.status_code == 200
    assert "db.txt" in stream.text  # the in-scope document is cited
    assert "cats.txt" not in stream.text  # the out-of-scope document never appears


async def test_multi_turn_followup_carries_prior_question(client, monkeypatch):
    monkeypatch.setattr(llm.settings, "use_fake_llm", True)
    await _register(client, "multiturn@example.com")
    await _upload(
        client,
        "space.txt",
        b"The Apollo program landed astronauts on the Moon. The Saturn V rocket gave the thrust.",
    )

    new = await client.post("/chat/new")
    conversation_id = new.headers["HX-Redirect"].split("/chat/")[1]

    # Turn 1: use a distinctive word ("zebra") that is absent from the document
    # and from the follow-up, so its later reappearance can only come from history.
    ask1 = await client.post(
        f"/chat/{conversation_id}/ask", data={"content": "Explain the zebra topic"}
    )
    stream1 = await client.get(f"/chat/{conversation_id}/stream/{_assistant_id(ask1.text)}")
    assert "event: done" in stream1.text
    assert "Following" not in stream1.text  # first turn has no prior context

    # Turn 2: a follow-up. The demo answer acknowledges the earlier question.
    ask2 = await client.post(
        f"/chat/{conversation_id}/ask", data={"content": "Which rocket was used?"}
    )
    stream2 = await client.get(f"/chat/{conversation_id}/stream/{_assistant_id(ask2.text)}")
    assert "Following" in stream2.text  # multi-turn acknowledgement
    assert "zebra" in stream2.text  # prior question was carried into this turn
