"""The keyless demo answer path (USE_FAKE_LLM) used by the public deploy."""
from __future__ import annotations

from app.models import Chunk
from app.services import llm


async def test_fake_llm_answers_extractively_with_citations(monkeypatch):
    monkeypatch.setattr(llm.settings, "use_fake_llm", True)
    chunks = [Chunk(content="Mitochondria are the powerhouse of the cell.")]

    out = "".join(
        [piece async for piece in llm.stream_answer("What are mitochondria?", chunks)]
    )

    assert "powerhouse" in out
    assert "[1]" in out


async def test_fake_llm_handles_no_chunks(monkeypatch):
    monkeypatch.setattr(llm.settings, "use_fake_llm", True)

    out = "".join([piece async for piece in llm.stream_answer("anything?", [])])

    assert "couldn't find" in out.lower()
