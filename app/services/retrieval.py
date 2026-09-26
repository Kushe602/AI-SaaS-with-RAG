"""Vector retrieval over a user's document chunks.

This reference implementation loads the user's chunks and ranks them by cosine
similarity in-process with NumPy — zero external services, so it runs anywhere and
is trivial to test. For large corpora, swap this for a pgvector query (store the
embedding in a ``vector`` column and order by the ``<=>`` distance operator); the
call site only depends on the ``search`` signature below.
"""
import json

import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Chunk
from app.services.embeddings import get_embedder


async def search(
    db: AsyncSession, owner_id: str, query: str, top_k: int
) -> list[tuple[Chunk, float]]:
    """Return up to ``top_k`` (chunk, similarity) pairs, most similar first."""
    result = await db.execute(select(Chunk).where(Chunk.owner_id == owner_id))
    chunks = list(result.scalars().all())
    if not chunks:
        return []

    query_vec = np.asarray(get_embedder().embed_one(query), dtype=np.float32)
    matrix = np.asarray([json.loads(c.embedding) for c in chunks], dtype=np.float32)

    query_norm = query_vec / (np.linalg.norm(query_vec) or 1.0)
    matrix_norm = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9)
    sims = matrix_norm @ query_norm

    top_indices = np.argsort(-sims)[:top_k]
    return [(chunks[i], float(sims[i])) for i in top_indices]
