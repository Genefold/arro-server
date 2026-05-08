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
