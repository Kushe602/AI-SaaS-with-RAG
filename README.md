# 📄 DocuChat

> An AI SaaS that lets users upload documents and chat with them — answers are grounded in the source text using **retrieval-augmented generation (RAG)** and streamed live with citations.

![CI](https://github.com/your-username/docuchat/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

DocuChat is a full-stack, production-shaped web application built with **FastAPI + HTMX** and **Claude**. Users sign up, upload PDFs or text files, and ask questions in natural language. The app retrieves the most relevant passages from their documents and asks Claude to answer using only that context — citing its sources.

<!-- Replace with a real screenshot or GIF once you deploy: it's the first thing recruiters look at. -->
<!-- ![DocuChat screenshot](docs/screenshot.png) -->

---

## ✨ Features

- **Account system** — registration, login, and logout with hashed passwords (bcrypt) and JWT session cookies.
- **Document ingestion** — upload PDFs / text files; the app extracts text, chunks it, and embeds each chunk.
- **Semantic retrieval (RAG)** — questions are embedded and matched against document chunks by cosine similarity.
- **Streaming answers** — responses stream token-by-token over Server-Sent Events, with inline `[n]` citations.
- **Usage metering & plans** — per-user document and daily-question limits (free / pro tiers), ready for Stripe.
- **Fully tested** — a `pytest` suite covering auth, ingestion, retrieval, and the end-to-end chat flow.
- **Containerized** — one-command run with Docker Compose (FastAPI + Postgres), plus GitHub Actions CI.

---

## 🏗️ Architecture

```
        ┌──────────────┐   upload    ┌───────────────┐   embed    ┌──────────────┐
        │   Browser    │ ──────────▶ │  Ingestion    │ ─────────▶ │  Chunk store │
        │ (HTMX + SSE) │             │ extract+chunk │            │ (embeddings) │
        └──────┬───────┘             └───────────────┘            └──────┬───────┘
               │ ask question                                            │
               ▼                                                         │ cosine search
        ┌──────────────┐   top-k chunks   ┌───────────────┐             │
        │  Retrieval   │ ◀────────────────┼───────────────┼─────────────┘
        └──────┬───────┘                  └───────────────┘
               │ context + question
               ▼
        ┌──────────────┐   streamed tokens + citations
        │  Claude LLM  │ ─────────────────────────────▶  Browser
        └──────────────┘
```

The RAG pipeline: **extract → chunk → embed → retrieve → generate**. Retrieved passages are injected into the prompt as numbered context, and Claude is instructed to answer only from that context and cite passages inline.

---

## 🧰 Tech stack

| Layer          | Choice                                                        |
| -------------- | ------------------------------------------------------------- |
| API / backend  | FastAPI (async), Uvicorn                                      |
| Frontend       | Server-rendered Jinja2 + HTMX + Tailwind (CDN), SSE streaming |
| Database / ORM | SQLAlchemy 2.0 (async) — SQLite locally, Postgres in Docker   |
| Embeddings     | fastembed (ONNX, local, no GPU) — swappable                   |
| LLM            | Anthropic Claude (`claude-sonnet-5` by default)               |
| Auth           | bcrypt password hashing + JWT cookies (PyJWT)                 |
| Tests / CI     | pytest + httpx, GitHub Actions                                |

---

## 🚀 Quickstart (local, zero infrastructure)

Requires Python 3.11+. Uses SQLite and needs only an Anthropic API key.

```bash
git clone https://github.com/your-username/docuchat.git
cd docuchat
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
cp .env.example .env                                 # then edit .env
```

Set at least these in `.env`:

```bash
SECRET_KEY=<run: python -c "import secrets; print(secrets.token_hex(32))">
ANTHROPIC_API_KEY=sk-ant-...
```

Run it:

```bash
uvicorn app.main:app --reload
```

Open http://localhost:8000, create an account, upload a document, and start chatting.

> First launch downloads a small embedding model (~100 MB). To skip that (e.g. for a quick smoke test without real embeddings), set `USE_FAKE_EMBEDDINGS=1`.

---

## 🐳 Run with Docker (FastAPI + Postgres)

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export SECRET_KEY=$(python -c "import secrets; print(secrets.token_hex(32))")
docker compose up --build
```

This starts Postgres (with the `pgvector` extension available) and the app on http://localhost:8000.

---

## ⚙️ Configuration

All settings are read from environment variables / `.env` (see `.env.example`).

| Variable              | Default                              | Description                              |
| --------------------- | ------------------------------------ | ---------------------------------------- |
| `SECRET_KEY`          | `change-me-please`                   | JWT signing key — **set this**.          |
| `DATABASE_URL`        | `sqlite+aiosqlite:///./docuchat.db`  | Async SQLAlchemy URL.                    |
| `ANTHROPIC_API_KEY`   | _(none)_                             | Required for chat answers.               |
| `CHAT_MODEL`          | `claude-sonnet-5`                    | Claude model id.                         |
| `USE_FAKE_EMBEDDINGS` | `0`                                  | `1` = deterministic embeddings (tests).  |
| `EMBED_MODEL`         | `BAAI/bge-small-en-v1.5`             | fastembed model name.                    |
| `FREE_MAX_DOCUMENTS`  | `5`                                  | Free-plan document cap.                  |
| `FREE_DAILY_QUESTIONS`| `25`                                 | Free-plan questions per day.             |

---

## 🧪 Tests

The suite uses fake embeddings and a temporary SQLite database, so it runs anywhere with no external services or API keys:

```bash
USE_FAKE_EMBEDDINGS=1 pytest -q
```

Coverage spans auth (register/login/guarding), ingestion + chunking, retrieval ranking, and the full ask → stream → persist chat flow (with the LLM mocked).

---

## 📁 Project structure

```
app/
├── main.py            # FastAPI app + lifespan
├── config.py          # pydantic-settings
├── database.py        # async engine/session/Base
├── models.py          # User, Document, Chunk, Conversation, Message, UsageRecord
├── security.py        # bcrypt + JWT
├── dependencies.py    # auth dependencies
├── routers/           # auth, pages, documents, chat (SSE)
├── services/          # embeddings, chunking, ingestion, retrieval, llm, usage
└── templates/         # Jinja2 + HTMX views and partials
tests/                 # pytest suite
```

---

## 📈 Design notes & scaling to production

This project favors a **zero-setup default** so it runs and tests anywhere, while keeping clear seams for production upgrades:

- **Vector search** — retrieval ranks chunks with NumPy cosine similarity in-process (`app/services/retrieval.py`). This is simple and dependency-free. For large corpora, store embeddings in a `pgvector` column and replace the body of `search()` with an `ORDER BY embedding <=> :query` query — the Postgres + `pgvector` image is already wired up in `docker-compose.yml`.
- **Billing** — usage metering and plan limits are fully implemented (`app/services/usage.py`). To monetize, wire Stripe Checkout + webhooks to flip `User.plan` between `free` and `pro`; the limits then apply automatically.
- **Migrations** — tables are created on startup via `create_all` for convenience. A production deployment should manage schema with Alembic.
- **Embeddings** — the `Embedder` interface makes the backend swappable (local fastembed, a hosted embeddings API, etc.) without touching the pipeline.
- **Security hardening** — set a strong `SECRET_KEY`, serve over HTTPS, and enable the `Secure` cookie flag before deploying publicly.

## 🗺️ Roadmap

- [ ] pgvector-backed retrieval behind the existing `search()` seam
- [ ] Stripe billing for the pro plan
- [ ] Multi-file conversations & document scoping per chat
- [ ] Alembic migrations
- [ ] Markdown rendering of answers

## 📝 License

MIT — see [LICENSE](LICENSE).

