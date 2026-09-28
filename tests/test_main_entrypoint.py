from __future__ import annotations

import pytest

from arro_server.__main__ import main


@pytest.fixture
def captured_run(monkeypatch):
    import uvicorn

    calls: dict[str, object] = {}
    monkeypatch.setattr(uvicorn, "run", lambda *a, **kw: calls.update(kw))
    monkeypatch.delenv("ARRO_BIND_PORT", raising=False)
    monkeypatch.delenv("ARRO_SERVER_PORT", raising=False)
    return calls


def test_k8s_service_link_does_not_break_startup(captured_run, monkeypatch) -> None:
    monkeypatch.setenv("ARRO_SERVER_PORT", "tcp://10.96.2.213:8000")
    main()
    assert captured_run["port"] == 8000


def test_bind_port_wins_over_legacy_name(captured_run, monkeypatch) -> None:
    monkeypatch.setenv("ARRO_SERVER_PORT", "tcp://10.96.2.213:8000")
    monkeypatch.setenv("ARRO_BIND_PORT", "9000")
    main()
    assert captured_run["port"] == 9000


def test_invalid_bind_port_fails_loudly(captured_run, monkeypatch) -> None:
    monkeypatch.setenv("ARRO_BIND_PORT", "not-a-port")
    with pytest.raises(SystemExit, match="ARRO_BIND_PORT"):
        main()
