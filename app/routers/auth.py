"""Authentication routes: register, login, logout (HTMX-friendly)."""
from fastapi import APIRouter, Depends, Form, Request, Response, status
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import get_db
from app.dependencies import get_optional_user
from app.models import User
from app.security import (
    COOKIE_NAME,
    create_access_token,
    hash_password,
    verify_password,
)
from app.web import templates

router = APIRouter(tags=["auth"])


def _set_auth_cookie(response: Response, user_id: str) -> None:
    token = create_access_token(user_id)
    response.set_cookie(
        COOKIE_NAME,
        token,
        httponly=True,
        samesite="lax",
        max_age=settings.access_token_expire_minutes * 60,
        path="/",
    )


def _error(message: str) -> HTMLResponse:
    return HTMLResponse(f'<p class="text-sm text-red-600">{message}</p>')


@router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request, user=Depends(get_optional_user)):
    if user:
        return RedirectResponse("/app", status_code=303)
    return templates.TemplateResponse(request, "login.html")


@router.get("/register", response_class=HTMLResponse)
async def register_page(request: Request, user=Depends(get_optional_user)):
    if user:
        return RedirectResponse("/app", status_code=303)
    return templates.TemplateResponse(request, "register.html")


@router.post("/register")
async def register(
    email: str = Form(...),
    password: str = Form(...),
    db: AsyncSession = Depends(get_db),
):
    email = email.strip().lower()
    if len(password) < 8:
        return _error("Password must be at least 8 characters.")
    existing = await db.execute(select(User).where(User.email == email))
    if existing.scalar_one_or_none():
        return _error("An account with that email already exists.")

    user = User(email=email, hashed_password=hash_password(password))
    db.add(user)
    await db.commit()

    response = Response(status_code=status.HTTP_204_NO_CONTENT)
    response.headers["HX-Redirect"] = "/app"
    _set_auth_cookie(response, user.id)
    return response


@router.post("/login")
async def login(
    email: str = Form(...),
    password: str = Form(...),
    db: AsyncSession = Depends(get_db),
):
    email = email.strip().lower()
    result = await db.execute(select(User).where(User.email == email))
    user = result.scalar_one_or_none()
    if not user or not verify_password(password, user.hashed_password):
        return _error("Invalid email or password.")

    response = Response(status_code=status.HTTP_204_NO_CONTENT)
    response.headers["HX-Redirect"] = "/app"
    _set_auth_cookie(response, user.id)
    return response


@router.post("/logout")
async def logout():
    response = Response(status_code=status.HTTP_204_NO_CONTENT)
    response.headers["HX-Redirect"] = "/"
    response.delete_cookie(COOKIE_NAME, path="/")
    return response
