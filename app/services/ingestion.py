"""Document ingestion: extract text, chunk it, embed the chunks, and persist."""
import io
import json

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models import Chunk, Document
from app.services.chunking import chunk_text
from app.services.embeddings import get_embedder


def extract_text(filename: str, content_type: str | None, data: bytes) -> str:
    """Extract plain text from a supported upload (PDF, TXT, MD)."""
    name = filename.lower()
    if name.endswith(".pdf") or (content_type or "") == "application/pdf":
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    return data.decode("utf-8", errors="ignore")


async def ingest_document(
    db: AsyncSession,
    owner_id: str,
    filename: str,
    content_type: str | None,
    data: bytes,
) -> Document:
    """Ingest one uploaded file and return the persisted Document."""
    text = extract_text(filename, content_type, data)
    chunks = chunk_text(text, settings.chunk_size, settings.chunk_overlap)

    document = Document(
        owner_id=owner_id,
        filename=filename,
        content_type=content_type or "text/plain",
        num_chunks=len(chunks),
        status="ready",
    )
    db.add(document)
    await db.flush()  # assign document.id

    if chunks:
        vectors = get_embedder().embed(chunks)
        for index, (content, vector) in enumerate(zip(chunks, vectors, strict=True)):
            db.add(
                Chunk(
                    document_id=document.id,
                    owner_id=owner_id,
                    chunk_index=index,
                    content=content,
                    embedding=json.dumps(vector),
                )
            )

    await db.commit()
    await db.refresh(document)
    return document
