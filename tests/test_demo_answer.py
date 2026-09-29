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


async def test_fake_llm_acknowledges_prior_turn(monkeypatch):
    monkeypatch.setattr(llm.settings, "use_fake_llm", True)
    chunks = [Chunk(content="The Saturn V rocket provided the thrust.")]
    history = [
        ("user", "What landed astronauts on the Moon?"),
        ("assistant", "The Apollo program [1]"),
    ]

    out = "".join(
        [piece async for piece in llm.stream_answer("Which rocket?", chunks, history=history)]
    )

    # The follow-up echoes the earlier question, exercising multi-turn context.
    assert "Following up" in out
    assert "What landed astronauts on the Moon?" in out
    assert "[1]" in out


async def test_fake_llm_omits_followup_without_history(monkeypatch):
    monkeypatch.setattr(llm.settings, "use_fake_llm", True)
    chunks = [Chunk(content="Standalone fact.")]

    out = "".join([piece async for piece in llm.stream_answer("First question?", chunks)])

    assert "Following up" not in out

