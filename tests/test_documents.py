"""Document management UI: viewing a document's chunks and cascading deletion."""
from sqlalchemy import func, select

from app.database import SessionLocal
from app.models import Chunk, Document
from app.services.ingestion import ingest_document


async def _register(client, email="docs@example.com"):
    await client.post("/register", data={"email": email, "password": "password123"})


async def test_view_document_chunks(client):
    await _register(client)
    content = b"The mitochondria is the powerhouse of the cell. " * 40
    await client.post("/documents/upload", files={"file": ("bio.txt", content, "text/plain")})

    async with SessionLocal() as db:
        document = (await db.execute(select(Document))).scalars().one()

    response = await client.get(f"/documents/{document.id}/chunks")
    assert response.status_code == 200
    assert "bio.txt" in response.text
    assert "Chunk 1" in response.text
    assert "mitochondria" in response.text


async def test_view_chunks_rejects_other_users_document(client):
    await _register(client, "owner@example.com")
    await client.post(
        "/documents/upload",
        files={"file": ("secret.txt", b"private notes", "text/plain")},
    )
    async with SessionLocal() as db:
        document = (await db.execute(select(Document))).scalars().one()

    # A different, freshly-authenticated user must not read those chunks.
    client.cookies.clear()
    await _register(client, "intruder@example.com")
    response = await client.get(f"/documents/{document.id}/chunks")
    assert response.status_code == 404


async def test_delete_document_cascades_to_chunks(client):
    await _register(client, "deleter@example.com")
    await client.post(
        "/documents/upload",
        files={"file": ("gone.txt", b"word " * 400, "text/plain")},
    )

    async with SessionLocal() as db:
        document = (await db.execute(select(Document))).scalars().one()
        assert document.num_chunks > 0

    response = await client.post(f"/documents/{document.id}/delete")
    assert response.status_code == 200
    assert "gone.txt" not in response.text

    async with SessionLocal() as db:
        assert (await db.execute(select(func.count(Document.id)))).scalar_one() == 0
        # Cascade removed the orphaned chunks too.
        assert (await db.execute(select(func.count(Chunk.id)))).scalar_one() == 0


async def test_delete_document_service_cascade():
    async with SessionLocal() as db:
        from app.models import User

        user = User(email="svc-del@example.com", hashed_password="x")
        db.add(user)
        await db.commit()
        document = await ingest_document(db, user.id, "d.txt", "text/plain", b"word " * 300)
        assert (
            await db.execute(select(func.count(Chunk.id)).where(Chunk.document_id == document.id))
        ).scalar_one() > 0

        await db.delete(document)
        await db.commit()
        assert (
            await db.execute(select(func.count(Chunk.id)).where(Chunk.document_id == document.id))
        ).scalar_one() == 0
