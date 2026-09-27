FROM python:3.11-slim

WORKDIR /code
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

COPY pyproject.toml README.md ./
COPY app ./app

RUN pip install --upgrade pip && pip install .

EXPOSE 8000
# Bind $PORT when the platform provides one (Render/Railway/Fly/Heroku set it);
# fall back to 8000 for local `docker run`. `exec` hands signals straight to
# uvicorn for clean shutdowns.
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
