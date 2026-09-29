"""Document management routes: upload, list, view chunks, delete."""
from fastapi import APIRouter, Depends, File, Request, UploadFile
from fastapi.responses import HTMLResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import get_db
from app.dependencies import get_current_user
from app.models import Chunk, Document
from app.services import usage
from app.services.ingestion import ingest_document
from app.web import templates

router = APIRouter(prefix="/documents", tags=["documents"])


async def _render_panel(request: Request, db: AsyncSession, user, error: str | None = None):
    documents = (
        (await db.execute(
            select(Document)
            .where(Document.owner_id == user.id)
            .order_by(Document.created_at.desc())
        )).scalars().all()
    )
    limits = usage.plan_limits(user.plan)
    return templates.TemplateResponse(
        request,
        "partials/documents_panel.html",
        {
            "documents": documents,
            "documents_used": len(documents),
            "documents_limit": limits["max_documents"],
            "error": error,
        },
    )


@router.post("/upload", response_class=HTMLResponse)
async def upload(
    request: Request,
    file: UploadFile = File(...),
    user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    if not await usage.can_upload(db, user):
        return await _render_panel(
            request, db, user, error="Document limit reached for your plan."
        )

    data = await file.read()
    if len(data) > settings.max_upload_mb * 1024 * 1024:
        return await _render_panel(
            request, db, user, error=f"File exceeds {settings.max_upload_mb} MB limit."
        )

    await ingest_document(db, user.id, file.filename or "upload.txt", file.content_type, data)
    return await _render_panel(request, db, user)


@router.get("/{document_id}/chunks", response_class=HTMLResponse)
async def view_chunks(
    document_id: str,
    request: Request,
    user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    document = (
        await db.execute(select(Document).where(Document.id == document_id))
    ).scalar_one_or_none()
    if not document or document.owner_id != user.id:
        return HTMLResponse("", status_code=404)

    chunks = (
        (await db.execute(
            select(Chunk)
            .where(Chunk.document_id == document_id)
            .order_by(Chunk.chunk_index)
        )).scalars().all()
    )
    return templates.TemplateResponse(
        request,
        "partials/chunks_modal.html",
        {"document": document, "chunks": chunks},
    )


@router.post("/{document_id}/delete", response_class=HTMLResponse)
async def delete_document(
    document_id: str,
    request: Request,
    user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    document = (
        await db.execute(select(Document).where(Document.id == document_id))
    ).scalar_one_or_none()
    if document and document.owner_id == user.id:
        await db.delete(document)  # cascades to the document's chunks
        await db.commit()
    return await _render_panel(request, db, user)
