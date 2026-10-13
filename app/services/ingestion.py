"""Document ingestion: extract text, chunk it, embed the chunks, and persist."""
import io
import json
from pathlib import PurePath

from pypdf import PdfReader
from pypdf.errors import PdfReadError
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models import Chunk, Document
from app.services.chunking import chunk_text
from app.services.embeddings import get_embedder


class DocumentIngestionError(ValueError):
    """An upload cannot be converted into searchable text."""


def extract_text(filename: str, content_type: str | None, data: bytes) -> str:
    """Extract searchable text from PDF, TXT, or Markdown uploads."""
    suffix = PurePath(filename.lower()).suffix
    is_pdf = (content_type or "").split(";", 1)[0].strip().lower() == "application/pdf"
    if suffix == ".pdf" or is_pdf:
        try:
            reader = PdfReader(io.BytesIO(data))
            text = "\n".join((page.extract_text() or "") for page in reader.pages)
        except (PdfReadError, EOFError, OSError) as exc:
            raise DocumentIngestionError(
                "This PDF could not be read. Please try exporting or uploading it again."
            ) from exc
        except Exception as exc:
            # Malformed PDFs can raise different parser exceptions depending on the file.
            raise DocumentIngestionError(
                "This PDF could not be read. Please try exporting or uploading it again."
            ) from exc
    elif suffix in {".txt", ".md", ".markdown"}:
        text = data.decode("utf-8", errors="ignore")
    else:
        raise DocumentIngestionError(
            "Unsupported file type. Upload a searchable PDF, TXT, or Markdown file."
        )

    if not text.strip():
        raise DocumentIngestionError(
            "No readable text was found. If this is a scanned/image-only PDF, run OCR "
            "or upload a searchable PDF before trying again."
        )
    return text


async def ingest_document(
    db: AsyncSession,
    owner_id: str,
    filename: str,
    content_type: str | None,
    data: bytes,
) -> Document:
    """Ingest one upload; reject files that produce no searchable chunks."""
    text = extract_text(filename, content_type, data)
    chunks = chunk_text(text, settings.chunk_size, settings.chunk_overlap)
    if not chunks:
        raise DocumentIngestionError(
            "No searchable text could be extracted. Try a text-based PDF, TXT, or Markdown file."
        )

    # Embed before adding the document so an embedding failure cannot leave a
    # misleading 'ready' document with zero chunks in the database.
    vectors = get_embedder().embed(chunks)
    if len(vectors) != len(chunks):
        raise RuntimeError("The embedding backend returned an unexpected number of vectors.")

    document = Document(
        owner_id=owner_id,
        filename=filename,
        content_type=content_type or "text/plain",
        num_chunks=len(chunks),
        status="ready",
    )
    db.add(document)
    await db.flush()  # assign document.id

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
