"""Tests for encryption of personal provider credentials."""
from app.services.user_llm_config import decrypt_api_key, encrypt_api_key
from app.models import UserLLMConfig


def test_api_key_encryption_round_trip():
    secret = "test-provider-secret-123"
    encrypted = encrypt_api_key(secret)
    assert encrypted != secret
    assert secret not in encrypted
    config = UserLLMConfig(encrypted_api_key=encrypted)
    assert decrypt_api_key(config) == secret
