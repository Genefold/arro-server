from __future__ import annotations

import os


def main() -> None:
    import uvicorn

    host = os.environ.get("ARROSPACE_HOST", "0.0.0.0")
    port = int(os.environ.get("ARROSPACE_PORT", "8000"))
    reload = os.environ.get("ARROSPACE_RELOAD", "0") == "1"
    uvicorn.run(
        "arrospace_server.app:create_app",
        factory=True,
        host=host,
        port=port,
        reload=reload,
    )


if __name__ == "__main__":
    main()
