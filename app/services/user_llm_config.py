"""Helpers for safely storing and resolving per-user OpenAI-compatible API settings."""
from __future__ import annotations

import base64
import hashlib

from cryptography.fernet import Fernet, InvalidToken

from app.config import settings
from app.models import UserLLMConfig


def _cipher() -> Fernet:
    # Derive a stable Fernet key from the server secret; never send it to the browser.
    digest = hashlib.sha256(settings.secret_key.encode("utf-8")).digest()
    return Fernet(base64.urlsafe_b64encode(digest))


def encrypt_api_key(api_key: str) -> str:
    return _cipher().encrypt(api_key.encode("utf-8")).decode("ascii")


def decrypt_api_key(config: UserLLMConfig) -> str:
    try:
        return _cipher().decrypt(config.encrypted_api_key.encode("ascii")).decode("utf-8")
    except (InvalidToken, ValueError, UnicodeDecodeError) as exc:
        raise RuntimeError(
            "Unable to decrypt the saved API key. Check whether SECRET_KEY was rotated."
        ) from exc


def effective_config(saved: UserLLMConfig | None) -> dict[str, str | None]:
    """Return provider settings, preferring a saved user configuration over env defaults."""
    if saved:
        return {
            "api_key": decrypt_api_key(saved),
            "base_url": saved.base_url,
            "model": saved.model,
            "provider": saved.provider,
        }
    return {
        "api_key": settings.llm_api_key,
        "base_url": settings.llm_base_url,
        "model": settings.llm_model,
        "provider": "Server default",
    }
