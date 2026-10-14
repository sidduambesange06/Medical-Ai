"""
Smoke tests for MedBot.
These tests verify configuration loading and core utilities without requiring
external services (Supabase, Redis, Groq, Pinecone).
"""
import os
import sys
import json
from pathlib import Path
import pytest

# Project root — one level up from tests/
ROOT = Path(__file__).parent.parent

# Make sure the project root is on sys.path for direct imports
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# .env.example content checks
# ---------------------------------------------------------------------------

class TestEnvExample:
    def test_env_example_exists(self):
        assert (ROOT / ".env.example").exists(), ".env.example not found"

    def test_env_example_has_required_keys(self):
        content = (ROOT / ".env.example").read_text()
        required = [
            "SECRET_KEY",
            "SUPABASE_URL",
            "SUPABASE_KEY",
            "GROQ_API_KEY",
            "PINECONE_API_KEY",
            "REDIS_URL",
            "ADMIN_EMAIL",
            "ADMIN_PASSWORD",
        ]
        for key in required:
            assert key in content, f"{key} missing from .env.example"


# ---------------------------------------------------------------------------
# requirements.txt checks
# ---------------------------------------------------------------------------

class TestRequirementsFile:
    def test_requirements_txt_exists(self):
        assert (ROOT / "requirements.txt").exists(), "requirements.txt not found"

    def test_requirements_has_key_packages(self):
        content = (ROOT / "requirements.txt").read_text().lower()
        for pkg in ["fastapi", "uvicorn", "redis", "groq", "pinecone"]:
            assert pkg in content, f"{pkg} missing from requirements.txt"


# ---------------------------------------------------------------------------
# Hardcoded-secret checks in core/config.py (Phase 2 validation)
# ---------------------------------------------------------------------------

class TestConfigNoHardcodedSecrets:
    """
    After the credential fix (Phase 2), core/config.py must not contain
    literal credential defaults that were previously committed.
    """

    def test_admin_email_has_no_hardcoded_default(self):
        config_text = (ROOT / "core" / "config.py").read_text()
        assert "sidduambesange005@gmail.com" not in config_text, (
            "Hardcoded ADMIN_EMAIL default still present in core/config.py"
        )

    def test_admin_password_has_no_hardcoded_default(self):
        config_text = (ROOT / "core" / "config.py").read_text()
        assert "siddustar2004" not in config_text, (
            "Hardcoded ADMIN_PASSWORD default still present in core/config.py"
        )

    def test_secret_key_has_no_hardcoded_default(self):
        config_text = (ROOT / "core" / "config.py").read_text()
        assert "ca87f5c370b94fa7df1de744288f9180" not in config_text, (
            "Hardcoded SECRET_KEY default still present in core/config.py"
        )

    def test_config_reads_secret_key_from_env(self, monkeypatch):
        """
        ProductionConfig reads env vars at class definition time, so we must
        remove any cached module before re-importing.
        """
        monkeypatch.setenv("SECRET_KEY", "a" * 32)
        monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
        monkeypatch.setenv("SUPABASE_KEY", "dummy-key")
        monkeypatch.setenv("GROQ_API_KEY", "dummy-groq-key")

        # Remove cached module so class attributes are re-evaluated
        for mod_name in list(sys.modules.keys()):
            if mod_name in ("core.config", "core"):
                del sys.modules[mod_name]

        from core.config import ProductionConfig  # noqa: PLC0415

        assert ProductionConfig.SECRET_KEY == "a" * 32
        assert ProductionConfig.SUPABASE_URL == "https://example.supabase.co"


# ---------------------------------------------------------------------------
# security_manager: pure encrypt/decrypt round-trip (no DB needed)
# ---------------------------------------------------------------------------

class TestSecurityManagerEncryption:
    """
    ProductionSecurityManager.encrypt_data / decrypt_data are pure functions
    that only use the local AES-256-GCM key — no external services required.
    """

    def test_encrypt_decrypt_string_roundtrip(self):
        # Import lazily to avoid module-level side effects at collection time
        from security_manager import ProductionSecurityManager  # noqa: PLC0415

        mgr = ProductionSecurityManager(encryption_key="test_key_for_smoke_tests")
        plaintext = "sensitive medical data"
        encrypted = mgr.encrypt_data(plaintext)

        assert isinstance(encrypted, dict), "encrypt_data should return a dict"
        assert "encrypted_data" in encrypted
        assert "nonce" in encrypted
        assert "tag" in encrypted

        decrypted = mgr.decrypt_data(encrypted)
        assert decrypted == plaintext, "Decrypted value should match original"

    def test_encrypt_decrypt_dict_roundtrip(self):
        from security_manager import ProductionSecurityManager  # noqa: PLC0415

        mgr = ProductionSecurityManager(encryption_key="test_key_for_smoke_tests")
        data = {"patient": "test", "condition": "none"}
        encrypted = mgr.encrypt_data(data)
        decrypted = mgr.decrypt_data(encrypted)
        assert decrypted == data

    def test_password_hash_and_verify(self):
        from security_manager import ProductionSecurityManager  # noqa: PLC0415

        mgr = ProductionSecurityManager(encryption_key="test_key_for_smoke_tests")
        password = "StrongPassword123!"
        hashed = mgr.hash_password(password)
        assert hashed != password, "Hash must differ from plaintext"
        assert mgr.verify_password(hashed, password) is True
        assert mgr.verify_password(hashed, "wrong_password") is False

    def test_data_integrity_validation(self):
        from security_manager import ProductionSecurityManager  # noqa: PLC0415

        mgr = ProductionSecurityManager(encryption_key="test_key_for_smoke_tests")
        data = {"test": "integrity check"}
        encrypted = mgr.encrypt_data(data)
        assert mgr.validate_data_integrity(encrypted) is True


# ---------------------------------------------------------------------------
# book_registry.json sanity check
# ---------------------------------------------------------------------------

class TestBookRegistry:
    def test_book_registry_is_valid_json(self):
        registry_path = ROOT / "book_registry.json"
        if registry_path.exists():
            data = json.loads(registry_path.read_text())
            assert isinstance(data, (dict, list)), (
                "book_registry.json must contain a JSON object or array"
            )
        else:
            pytest.skip("book_registry.json not present — skipping")


# ---------------------------------------------------------------------------
# .gitignore — confirm .env is protected
# ---------------------------------------------------------------------------

class TestGitignore:
    def test_env_is_in_gitignore(self):
        gitignore_path = ROOT / ".gitignore"
        if not gitignore_path.exists():
            pytest.skip(".gitignore not present — skipping")
        content = gitignore_path.read_text()
        # Accept ".env" as a standalone line or as part of a comment block
        assert ".env" in content, ".env should be listed in .gitignore"
