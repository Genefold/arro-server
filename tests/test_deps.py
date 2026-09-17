"""Tests for the get_tuner_adapter dependency function."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from arro_server.deps import get_tuner_adapter
from arro_server.tuner_adapter import TunerAdapter


@pytest.fixture
def adapter() -> MagicMock:
    return MagicMock(spec=TunerAdapter)


def _app_with_adapter(a: MagicMock) -> FastAPI:
    app = FastAPI()

    @app.get("/test")
    def _test(x: TunerAdapter = Depends(get_tuner_adapter)):
        return {"ok": True}

    app.state.tuner_adapter = a
    return app


def _app_without_adapter() -> FastAPI:
    app = FastAPI()

    @app.get("/test")
    def _test(x: TunerAdapter = Depends(get_tuner_adapter)):
        return {"ok": True}

    return app


class TestGetTunerAdapterDep:
    def test_returns_adapter_when_registered(self, adapter):
        with TestClient(_app_with_adapter(adapter)) as client:
            resp = client.get("/test")
        assert resp.status_code == 200

    def test_raises_503_when_not_registered(self):
        with TestClient(_app_without_adapter(), raise_server_exceptions=False) as client:
            resp = client.get("/test")
        assert resp.status_code == 503

    def test_503_response_has_detail_key(self):
        with TestClient(_app_without_adapter(), raise_server_exceptions=False) as client:
            resp = client.get("/test")
        assert "detail" in resp.json()

    def test_adapter_identity_preserved(self, adapter):
        """The dependency must return the exact same object, not a copy."""
        captured = {}
        app = FastAPI()
        app.state.tuner_adapter = adapter

        @app.get("/identity")
        def _identity(x: TunerAdapter = Depends(get_tuner_adapter)):
            captured["adapter"] = x
            return {}

        with TestClient(app) as client:
            client.get("/identity")

        assert captured["adapter"] is adapter
