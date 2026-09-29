"""Answer generation with an OpenAI-compatible model, grounded in retrieved context (RAG)."""
from __future__ import annotations

from collections.abc import AsyncIterator, Sequence

from app.config import settings
from app.models import Chunk

SYSTEM_PROMPT = (
    "You are DocuChat, an assistant that answers questions strictly from the user's "
    "documents. Use only the numbered context passages provided. Cite the passages you "
    "use inline with bracketed numbers like [1] or [2]. If the answer is not contained "
    "in the context, say you couldn't find it in the provided documents. Be concise."
)


def build_context(chunks: Sequence[Chunk]) -> str:
    """Render retrieved chunks as a numbered context block for the prompt."""
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        parts.append(f"[{i}] {chunk.content}")
    return "\n\n".join(parts)


async def stream_answer(
    question: str,
    chunks: Sequence[Chunk],
    history: Sequence[tuple[str, str]] | None = None,
) -> AsyncIterator[str]:
    """Yield the answer token-by-token, grounded in ``chunks``.

    ``history`` is the prior conversation as ``(role, content)`` pairs so
    follow-up questions resolve against earlier turns. With ``USE_FAKE_LLM`` set —
    the keyless demo deployment — the answer is assembled extractively from the
    retrieved passages (and acknowledges the prior turn), so the full
    upload → retrieve → cited-answer flow works live without an API key.
    Otherwise an OpenAI-compatible model generates it (any provider/gateway
    speaking the OpenAI API — configure ``LLM_BASE_URL`` / ``LLM_API_KEY`` /
    ``LLM_MODEL``); with neither a key nor demo mode we return a gentle notice.
    """
    if settings.use_fake_llm:
        async for piece in _stream_demo_answer(question, chunks, history):
            yield piece
        return

    if not settings.llm_api_key:
        yield (
            "⚠️ No LLM_API_KEY is configured, so I can't generate an answer. "
            "Add an API key for any OpenAI-compatible provider to the .env file "
            "to enable chat."
        )
        return

    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=settings.llm_api_key, base_url=settings.llm_base_url)
    context = build_context(chunks) or "(no relevant passages found)"
    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for role, text in history or []:
        if role in ("user", "assistant") and text.strip():
            messages.append({"role": role, "content": text})
    messages.append(
        {"role": "user", "content": f"Context passages:\n{context}\n\nQuestion: {question}"}
    )

    stream = await client.chat.completions.create(
        model=settings.llm_model,
        max_tokens=settings.max_answer_tokens,
        messages=messages,
        stream=True,
    )
    async for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


async def _emit_words(text: str) -> AsyncIterator[str]:
    """Yield ``text`` word-by-word so the client renders an incremental stream."""
    for word in text.split(" "):
        yield word + " "


async def _stream_demo_answer(
    question: str,
    chunks: Sequence[Chunk],
    history: Sequence[tuple[str, str]] | None = None,
) -> AsyncIterator[str]:
    """Deterministic, keyless answer for the public demo.

    Streams a short extractive answer built from the top retrieved passages, with
    the same bracketed citations the real model is prompted to use, rendered as
    markdown. When the conversation has prior turns it opens by acknowledging the
    earlier question, so multi-turn follow-ups are exercised end-to-end without a
    live model.
    """
    prior_questions = [c for r, c in (history or []) if r == "user" and c.strip()]
    if prior_questions:
        async for piece in _emit_words(
            f'Following up on your earlier question "{prior_questions[-1].strip()}":\n\n'
        ):
            yield piece

    if not chunks:
        async for piece in _emit_words(
            "I couldn't find anything in your documents to answer that. "
            "Upload a document first, then ask about its contents. "
            "(Demo mode: answers are assembled without a live model.)"
        ):
            yield piece
        return

    async for piece in _emit_words(
        f'Here is what your documents say about **{question.strip()}**:\n\n'
    ):
        yield piece

    for i, chunk in enumerate(chunks[: settings.max_context_chunks], start=1):
        snippet = " ".join(chunk.content.split())
        if len(snippet) > 300:
            snippet = snippet[:300].rstrip() + "…"
        async for piece in _emit_words(f"- {snippet} [{i}]\n"):
            yield piece

    async for piece in _emit_words(
        "\n_Demo mode: this answer is assembled directly from the passages "
        "retrieved from your documents. Set an LLM_API_KEY to get fully "
        "synthesized answers._"
    ):
        yield piece
