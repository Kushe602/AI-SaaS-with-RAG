"""Pytest fixtures. Environment is configured before importing the app so the
engine binds to the test database and embeddings never hit the network.
"""
import os

os.environ.setdefault("USE_FAKE_EMBEDDINGS", "1")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///./test_docuchat.db")
os.environ.setdefault("SECRET_KEY", "test-secret-key-please-use-at-least-32-bytes")

import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app import models  # noqa: F401  (register models on the metadata)
from app.database import Base, engine
from app.main import app


@pytest_asyncio.fixture(autouse=True)
async def _reset_database():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    yield


@pytest_asyncio.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
