"""arro-server: FastAPI server for Zarr v3 + ArrowSpace datasets."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("arro-server")
except PackageNotFoundError:
    __version__ = "dev"
