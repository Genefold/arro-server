from __future__ import annotations

import os


def main() -> None:
    import uvicorn

    host = os.environ.get("ARRO_SERVER_HOST", "0.0.0.0")
    raw_port = os.environ.get("ARRO_BIND_PORT", "8000")
    try:
        port = int(raw_port)
    except ValueError as exc:
        raise SystemExit(f"ARRO_BIND_PORT must be an integer, got {raw_port!r}") from exc
    reload = os.environ.get("ARRO_SERVER_RELOAD", "0") == "1"
    uvicorn.run(
        "arro_server.app:create_app",
        factory=True,
        host=host,
        port=port,
        reload=reload,
    )


if __name__ == "__main__":
    main()
