"""Authenticated per-user API configuration routes."""
from urllib.parse import urlparse

from fastapi import APIRouter, Depends, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from openai import AsyncOpenAI
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.dependencies import get_current_user
from app.models import UserLLMConfig
from app.services.user_llm_config import decrypt_api_key, encrypt_api_key

router = APIRouter(prefix="/settings/api", tags=["settings"])


def _valid_base_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


@router.post("/save")
async def save_api_config(
    provider: str = Form(...),
    base_url: str = Form(...),
    model: str = Form(...),
    api_key: str = Form(""),
    user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    provider, base_url, model = provider.strip(), base_url.strip().rstrip("/"), model.strip()
    if not provider or len(provider) > 100 or not model or len(model) > 200:
        return HTMLResponse("Provider and model are required.", status_code=400)
    if len(base_url) > 500 or not _valid_base_url(base_url):
        return HTMLResponse("Enter a valid API base URL.", status_code=400)

    saved = (await db.execute(
        select(UserLLMConfig).where(UserLLMConfig.owner_id == user.id)
    )).scalar_one_or_none()
    api_key = api_key.strip()
    if not saved and not api_key:
        return HTMLResponse("Enter an API key to save this configuration.", status_code=400)
    if saved:
        saved.provider, saved.base_url, saved.model = provider, base_url, model
        if api_key:
            saved.encrypted_api_key = encrypt_api_key(api_key)
    else:
        db.add(UserLLMConfig(
            owner_id=user.id, provider=provider, base_url=base_url, model=model,
            encrypted_api_key=encrypt_api_key(api_key),
        ))
    await db.commit()
    return RedirectResponse("/app?api_config=saved", status_code=303)


@router.post("/test")
async def test_api_config(
    base_url: str = Form(...),
    model: str = Form(...),
    api_key: str = Form(""),
    user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    base_url, model, api_key = base_url.strip().rstrip("/"), model.strip(), api_key.strip()
    if not _valid_base_url(base_url) or not model:
        return HTMLResponse('<p class="text-sm text-red-400">Enter a valid base URL and model.</p>', status_code=400)
    if not api_key:
        saved = (await db.execute(
            select(UserLLMConfig).where(UserLLMConfig.owner_id == user.id)
        )).scalar_one_or_none()
        if saved:
            api_key = decrypt_api_key(saved)
    if not api_key:
        return HTMLResponse('<p class="text-sm text-red-400">Enter an API key to test.</p>', status_code=400)
    try:
        async with AsyncOpenAI(
            api_key=api_key, base_url=base_url, timeout=12.0, max_retries=0
        ) as client:
            await client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": "Reply with OK."}],
                max_tokens=5,
            )
    except Exception:
        return HTMLResponse(
            '<p class="text-sm text-red-400">Connection failed. Check provider, key, URL, model, and account limits.</p>',
            status_code=400,
        )
    return HTMLResponse('<p class="text-sm text-emerald-400">Connection successful.</p>')


@router.post("/remove")
async def remove_api_config(
    user=Depends(get_current_user), db: AsyncSession = Depends(get_db)
):
    saved = (await db.execute(
        select(UserLLMConfig).where(UserLLMConfig.owner_id == user.id)
    )).scalar_one_or_none()
    if saved:
        await db.delete(saved)
        await db.commit()
    return RedirectResponse("/app?api_config=removed", status_code=303)
