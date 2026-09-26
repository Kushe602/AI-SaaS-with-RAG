from app.database import SessionLocal
from app.models import User
from app.services import retrieval
from app.services.ingestion import ingest_document


async def test_search_ranks_relevant_chunk_first():
    async with SessionLocal() as db:
        user = User(email="ret@example.com", hashed_password="x")
        db.add(user)
        await db.commit()

        await ingest_document(
            db, user.id, "cats.txt", "text/plain",
            b"Cats are small furry animals that purr and chase mice.",
        )
        await ingest_document(
            db, user.id, "db.txt", "text/plain",
            b"Database indexing improves query performance using B-tree structures.",
        )

        results = await retrieval.search(db, user.id, "how does database indexing work", top_k=2)
        assert results
        top_chunk, score = results[0]
        assert "indexing" in top_chunk.content.lower()
        assert score > 0


async def test_search_returns_empty_without_documents():
    async with SessionLocal() as db:
        user = User(email="empty@example.com", hashed_password="x")
        db.add(user)
        await db.commit()
        assert await retrieval.search(db, user.id, "anything", top_k=5) == []
