from __future__ import annotations

from pathlib import Path

from arro_server import settings as settings_mod
from arro_server.settings import Settings


def test_csv_data_roots(monkeypatch, tmp_path: Path) -> None:
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    monkeypatch.setenv("ARRO_SERVER_DATA_ROOTS", f"{a},{b}")
    settings_mod.reset_settings_cache()
    s = Settings()
    roots = s.resolved_roots
    assert set(roots.values()) == {a.resolve(), b.resolve()}


def test_labeled_roots(monkeypatch, tmp_path: Path) -> None:
    a = tmp_path / "a"
    a.mkdir()
    monkeypatch.setenv("ARRO_SERVER_DATA_ROOTS", f"primary={a}")
    settings_mod.reset_settings_cache()
    s = Settings()
    roots = s.resolved_roots
    assert "primary" in roots
    assert roots["primary"] == a.resolve()


def test_collision_suffix(monkeypatch, tmp_path: Path) -> None:
    a = tmp_path / "shared"
    b = tmp_path / "nested" / "shared"
    a.mkdir()
    b.mkdir(parents=True)
    monkeypatch.setenv("ARRO_SERVER_DATA_ROOTS", f"{a},{b}")
    settings_mod.reset_settings_cache()
    s = Settings()
    labels = list(s.resolved_roots.keys())
    assert "shared" in labels
    assert any(name.startswith("shared-") for name in labels)


def test_items_limit_defaults() -> None:
    s = Settings()
    assert s.items_default_limit == 50
    assert s.items_max_limit == 100


def test_items_limit_validation_rejects_inconsistent_settings() -> None:
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        Settings(items_default_limit=0)
    with pytest.raises(ValidationError):
        Settings(items_default_limit=200, items_max_limit=100)


def test_max_response_elements_default() -> None:
    s = Settings()
    assert s.max_response_elements == 1_000_000


def test_max_response_elements_env_override(monkeypatch) -> None:
    monkeypatch.setenv("ARRO_SERVER_MAX_RESPONSE_ELEMENTS", "500000")
    settings_mod.reset_settings_cache()
    s = Settings()
    assert s.max_response_elements == 500_000


def test_max_response_elements_zero_is_valid() -> None:
    # Zero rejects every non-empty response — extreme but not invalid at the
    # settings level; enforcement happens at the route layer.
    s = Settings(max_response_elements=0)
    assert s.max_response_elements == 0
