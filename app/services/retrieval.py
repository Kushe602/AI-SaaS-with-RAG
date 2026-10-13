"""Vector retrieval over a user's document chunks.

Ranking blends semantic cosine similarity with a lightweight keyword/TF score
(hybrid retrieval), then reranks the top candidates with Maximal Marginal
Relevance (MMR) for diversity. Everything runs in-process with NumPy, so it needs
no external services and is fully deterministic under the fake embeddings used in
tests. For large corpora, swap the body of ``search`` for a pgvector query (store
the embedding in a ``vector`` column and order by the ``<=>`` distance operator);
the call site only depends on the ``search`` signature below.
"""
from __future__ import annotations

import json
import re
from collections.abc import Sequence

import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models import Chunk
from app.services.embeddings import get_embedder

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "by", "can",
    "could", "did", "do", "does", "for", "from", "give", "has", "have", "how",
    "i", "in", "into", "is", "it", "its", "me", "of", "on", "or", "please",
    "show", "tell", "that", "the", "their", "this", "to", "was", "we", "what",
    "when", "where", "which", "who", "why", "will", "with", "would", "you",
}


def _tokenize(text: str) -> list[str]:
    tokens = _TOKEN_RE.findall(text.lower())
    # Small deterministic normalization for common inflections.
    normalized = []
    for token in tokens:
        if token.endswith("ies") and len(token) > 4:
            token = token[:-3] + "y"
        elif token.endswith("ing") and len(token) > 5:
            token = token[:-3]
        elif token.endswith("ed") and len(token) > 4:
            token = token[:-2]
        elif token.endswith("s") and len(token) > 4:
            token = token[:-1]
        normalized.append(token)
    return normalized


def condense_query(
    history_questions: Sequence[str], question: str, max_turns: int | None = None
) -> str:
    """Fold the most recent user questions into the current one for retrieval.

    A bare follow-up ("and its cost?") carries little signal alone; prefixing the
    last few user turns gives retrieval the context to resolve it. Deterministic.
    """
    if max_turns is None:
        max_turns = settings.history_turns
    recent = [q.strip() for q in history_questions if q and q.strip()][-max_turns:]
    return " ".join([*recent, question.strip()]).strip()


def _minmax(scores: np.ndarray) -> np.ndarray:
    """Scale scores to [0, 1]. All-equal (or single-element) inputs map to all-ones."""
    if scores.size == 0:
        return scores
    lo = float(scores.min())
    hi = float(scores.max())
    if hi - lo < 1e-9:
        return np.ones_like(scores)
    return (scores - lo) / (hi - lo)


def _keyword_scores(query: str, texts: Sequence[str]) -> np.ndarray:
    """Return IDF-weighted query coverage, reducing the impact of boilerplate."""
    query_terms = set(_tokenize(query)) - _STOP_WORDS
    scores = np.zeros(len(texts), dtype=np.float32)
    if not query_terms or not texts:
        return scores

    token_sets = [set(_tokenize(text)) for text in texts]
    count = len(token_sets)
    document_frequency = {
        term: sum(term in terms for terms in token_sets) for term in query_terms
    }
    weights = {
        term: np.log1p(
            (count - document_frequency[term] + 0.5)
            / (document_frequency[term] + 0.5)
        )
        for term in query_terms
    }
    total_weight = sum(weights.values())
    if total_weight <= 0:
        return scores

    for i, terms in enumerate(token_sets):
        scores[i] = sum(
            weight for term, weight in weights.items() if term in terms
        ) / total_weight
    return scores


def _mmr_select(
    relevance: np.ndarray, sim_matrix: np.ndarray, top_k: int, lam: float
) -> list[int]:
    """Maximal Marginal Relevance: trade off relevance against redundancy.

    Greedily picks the most relevant item, then repeatedly the item maximizing
    ``lam * relevance - (1 - lam) * max similarity to an already-picked item``.
    Ties resolve to the lower index, so the result is deterministic.
    """
    n = int(relevance.shape[0])
    if n == 0 or top_k <= 0:
        return []
    first = max(range(n), key=lambda i: (float(relevance[i]), -i))
    selected = [first]
    remaining = [i for i in range(n) if i != first]
    while remaining and len(selected) < top_k:
        best = None
        best_score = float("-inf")
        for i in remaining:
            redundancy = max(float(sim_matrix[i, j]) for j in selected)
            score = lam * float(relevance[i]) - (1.0 - lam) * redundancy
            if score > best_score:
                best_score = score
                best = i
        selected.append(best)
        remaining.remove(best)
    return selected


async def search(
    db: AsyncSession,
    owner_id: str,
    query: str,
    top_k: int,
    document_ids: Sequence[str] | None = None,
) -> list[tuple[Chunk, float]]:
    """Return up to ``top_k`` (chunk, score) pairs, most relevant first.

    Ranking blends raw cosine similarity with query-term coverage, removes weak
    matches before MMR reranking, and then selects diverse relevant chunks.
    ``document_ids`` scopes search to the owner's chosen documents.
    """
    stmt = select(Chunk).where(Chunk.owner_id == owner_id)
    if document_ids:
        stmt = stmt.where(Chunk.document_id.in_(list(document_ids)))
    result = await db.execute(stmt)
    chunks = list(result.scalars().all())
    if not chunks:
        return []

    query_vec = np.asarray(get_embedder().embed_one(query), dtype=np.float32)
    matrix = np.asarray([json.loads(c.embedding) for c in chunks], dtype=np.float32)

    query_norm = query_vec / (np.linalg.norm(query_vec) or 1.0)
    matrix_norm = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9)
    semantic = matrix_norm @ query_norm

    keyword = _keyword_scores(query, [c.content for c in chunks])

    # Avoid normalizing against the current result set: that can make unrelated
    # chunks look relevant when every candidate has a low score.
    # Hashing embeddings are not semantic. In demo mode, rank only by lexical
    # evidence rather than allowing hash collisions to promote unrelated chunks.
    if settings.use_fake_embeddings:
        combined = keyword
    else:
        alpha = settings.hybrid_alpha
        combined = alpha * np.clip(semantic, 0.0, 1.0) + (1.0 - alpha) * keyword
    relevant = np.flatnonzero(combined >= settings.retrieval_min_score)
    if relevant.size == 0 or top_k <= 0:
        return []

    order = relevant[np.argsort(-combined[relevant], kind="stable")]
    pool_size = min(len(order), max(top_k, settings.retrieval_candidates))
    pool = [int(i) for i in order[:pool_size]]
    pool_vectors = matrix_norm[pool]
    sim_matrix = pool_vectors @ pool_vectors.T
    local = _mmr_select(combined[pool], sim_matrix, top_k, settings.mmr_lambda)
    return [(chunks[pool[i]], float(combined[pool[i]])) for i in local]
