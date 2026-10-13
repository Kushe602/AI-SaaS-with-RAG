import io

import pytest
from pypdf import PdfWriter
from sqlalchemy import func, select

from app.database import SessionLocal
from app.models import Chunk, User
from app.services.ingestion import DocumentIngestionError, extract_text, ingest_document


async def test_upload_creates_document_and_chunks(client):
    await client.post("/register", data={"email": "u@example.com", "password": "password123"})
    content = b"The mitochondria is the powerhouse of the cell. " * 40
    files = {"file": ("notes.txt", content, "text/plain")}
    response = await client.post("/documents/upload", files=files)

    assert response.status_code == 200
    assert "notes.txt" in response.text
    assert "1/5 documents used" in response.text

    async with SessionLocal() as db:
        chunk_count = (await db.execute(select(func.count(Chunk.id)))).scalar_one()
        assert chunk_count > 0


async def test_ingest_document_service_directly():
    async with SessionLocal() as db:
        user = User(email="svc@example.com", hashed_password="x")
        db.add(user)
        await db.commit()

        document = await ingest_document(
            db, user.id, "doc.txt", "text/plain", b"word " * 500
        )
        assert document.num_chunks > 1
        assert document.status == "ready"


def test_extract_text_rejects_unsupported_file_type():
    with pytest.raises(DocumentIngestionError, match="Unsupported file type"):
        extract_text("notes.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document", b"content")


def test_extract_text_rejects_empty_text_file():
    with pytest.raises(DocumentIngestionError, match="No readable text"):
        extract_text("empty.txt", "text/plain", b"   \n\t")


def test_extract_text_rejects_scanned_or_blank_pdf():
    buffer = io.BytesIO()
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    writer.write(buffer)

    with pytest.raises(DocumentIngestionError, match="No readable text"):
        extract_text("blank.pdf", "application/pdf", buffer.getvalue())


def test_extract_text_rejects_malformed_pdf():
    import pytest

    from app.services.ingestion import DocumentIngestionError, extract_text

    with pytest.raises(DocumentIngestionError, match="PDF could not be read"):
        extract_text("broken.pdf", "application/pdf", b"not a pdf")
