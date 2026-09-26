"""LLM answer generation with Claude, grounded in retrieved context (RAG)."""
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


async def stream_answer(question: str, chunks: Sequence[Chunk]) -> AsyncIterator[str]:
    """Yield the answer token-by-token from Claude, grounded in ``chunks``."""
    if not settings.anthropic_api_key:
        yield (
            "⚠️ No ANTHROPIC_API_KEY is configured, so I can't generate an answer. "
            "Add your key to the .env file to enable chat."
        )
        return

    from anthropic import AsyncAnthropic

    client = AsyncAnthropic(api_key=settings.anthropic_api_key)
    context = build_context(chunks) or "(no relevant passages found)"
    user_message = f"Context passages:\n{context}\n\nQuestion: {question}"

    async with client.messages.stream(
        model=settings.chat_model,
        max_tokens=settings.max_answer_tokens,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    ) as stream:
        async for text in stream.text_stream:
            yield text
