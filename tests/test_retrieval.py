import numpy as np

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


async def test_search_scoped_to_document_ids():
    async with SessionLocal() as db:
        user = User(email="scope-unit@example.com", hashed_password="x")
        db.add(user)
        await db.commit()

        await ingest_document(
            db, user.id, "cats.txt", "text/plain",
            b"Cats are small furry animals that purr and chase mice.",
        )
        db_doc = await ingest_document(
            db, user.id, "db.txt", "text/plain",
            b"Database indexing improves query performance using B-tree structures.",
        )

        # Even a cats-flavoured query must only return chunks from the scoped doc.
        results = await retrieval.search(
            db, user.id, "furry animals that purr", top_k=5, document_ids=[db_doc.id]
        )
        assert results
        assert all(chunk.document_id == db_doc.id for chunk, _ in results)


def test_condense_query_folds_recent_turns():
    condensed = retrieval.condense_query(
        ["What is a B-tree?", "How is it balanced?"], "and its height?"
    )
    assert "B-tree" in condensed
    assert "height" in condensed


def test_keyword_scores_reward_term_overlap():
    scores = retrieval._keyword_scores(
        "database indexing", ["database indexing is fast", "cats are furry"]
    )
    assert scores[0] > scores[1]


def test_mmr_select_prefers_diverse_results():
    relevance = np.array([1.0, 0.9, 0.8], dtype=np.float32)
    # Items 0 and 1 are near-duplicates; item 2 is distinct.
    sim = np.array([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    picked = retrieval._mmr_select(relevance, sim, top_k=2, lam=0.5)
    assert picked == [0, 2]


def test_minmax_handles_all_equal_scores():
    out = retrieval._minmax(np.array([0.3, 0.3, 0.3], dtype=np.float32))
    assert np.allclose(out, 1.0)

