# 📄 DocuChat

> An AI SaaS that lets users upload documents and chat with them — answers are grounded in the source text using **retrieval-augmented generation (RAG)** and streamed live with citations.

[![Live demo](https://img.shields.io/badge/live%20demo-online-brightgreen)](https://docuchat-94mn.onrender.com)
[![CI](https://github.com/Kushe602/AI-SaaS-with-RAG/actions/workflows/ci.yml/badge.svg)](https://github.com/Kushe602/AI-SaaS-with-RAG/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
[![Deploy to Render](https://img.shields.io/badge/deploy-Render-46E3B7)](https://render.com/deploy?repo=https://github.com/Kushe602/AI-SaaS-with-RAG)

> **Live demo:** https://docuchat-94mn.onrender.com — hosted on a free instance,
> so the first request may take ~50s to wake it. It runs in a keyless demo mode
> (fake embeddings + extractive, cited answers), so you can register, upload a
> document, and try grounded RAG answers with no API key.

DocuChat is a full-stack, production-shaped web application built with **FastAPI + HTMX** and **any OpenAI-compatible LLM**. Users sign up, upload PDFs or text files, and ask questions in natural language. The app retrieves the most relevant passages from their documents and asks the model to answer using only that context — citing its sources.

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
        │  LLM (any)   │ ─────────────────────────────▶  Browser
        └──────────────┘
```

The RAG pipeline: **extract → chunk → embed → retrieve → generate**. Retrieved passages are injected into the prompt as numbered context, and the model is instructed to answer only from that context and cite passages inline.

---

## 🧰 Tech stack

| Layer          | Choice                                                        |
| -------------- | ------------------------------------------------------------- |
| API / backend  | FastAPI (async), Uvicorn                                      |
| Frontend       | Server-rendered Jinja2 + HTMX + Tailwind (CDN), SSE streaming |
| Database / ORM | SQLAlchemy 2.0 (async) — SQLite locally, Postgres in Docker   |
| Embeddings     | fastembed (ONNX, local, no GPU) — swappable                   |
| LLM            | Any OpenAI-compatible API (OpenAI, Groq, OpenRouter, local…)  |
| Auth           | bcrypt password hashing + JWT cookies (PyJWT)                 |
| Tests / CI     | pytest + httpx, GitHub Actions                                |

---

## 🚀 Quickstart (local, zero infrastructure)

Requires Python 3.11+. Uses SQLite and needs only an API key for any OpenAI-compatible LLM provider.

```bash
git clone https://github.com/Kushe602/AI-SaaS-with-RAG.git
cd AI-SaaS-with-RAG
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
cp .env.example .env                                 # then edit .env
```

Set at least these in `.env`:

```bash
SECRET_KEY=<run: python -c "import secrets; print(secrets.token_hex(32))">
LLM_API_KEY=...
LLM_BASE_URL=https://api.justwoker.icu/v1
LLM_MODEL=gpt-4o-mini
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
export LLM_API_KEY=...                               # any OpenAI-compatible provider key
export SECRET_KEY=$(python -c "import secrets; print(secrets.token_hex(32))")
docker compose up --build
```

This starts Postgres (with the `pgvector` extension available) and the app on http://localhost:8000.

---

## ☁️ Deploy a live demo (Render, free)

DocuChat ships a [`render.yaml`](render.yaml) Blueprint that deploys a **keyless demo**: deterministic hashing embeddings power retrieval, and answers are assembled extractively from the retrieved passages (with `[n]` citations), so the full **upload → retrieve → cited-answer** flow works live with no API key.

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/Kushe602/AI-SaaS-with-RAG)

Click the button (or in the Render dashboard use **New + → Blueprint** and pick this repo). Render builds the Dockerfile, generates a `SECRET_KEY`, and sets `USE_FAKE_EMBEDDINGS=true` + `USE_FAKE_LLM=true`. The free plan sleeps when idle (~50s cold start) and uses ephemeral SQLite (uploaded docs reset on restart). Set `LLM_API_KEY` (plus `LLM_BASE_URL` / `LLM_MODEL` for your provider) and drop `USE_FAKE_LLM` for fully model-generated answers.

---

## ⚙️ Configuration

All settings are read from environment variables / `.env` (see `.env.example`).

| Variable              | Default                              | Description                              |
| --------------------- | ------------------------------------ | ---------------------------------------- |
| `SECRET_KEY`          | `change-me-please`                   | JWT signing key — **set this**.          |
| `DATABASE_URL`        | `sqlite+aiosqlite:///./docuchat.db`  | Async SQLAlchemy URL.                    |
| `LLM_API_KEY`         | _(none)_                             | Key for any OpenAI-compatible provider.  |
| `LLM_BASE_URL`        | `https://api.justwoker.icu/v1`       | OpenAI-compatible API base URL.          |
| `LLM_MODEL`           | `gpt-4o-mini`                        | Model id served by your provider.        |
| `USE_FAKE_EMBEDDINGS` | `0`                                  | `1` = deterministic embeddings (tests).  |
| `USE_FAKE_LLM`        | `0`                                  | `1` = extractive keyless answers (demo). |
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

