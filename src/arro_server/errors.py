"""Domain exceptions for arro-server.

All exceptions that cross module boundaries are defined here so that
route handlers can catch them without importing their source modules.
"""

from __future__ import annotations

from fastapi import HTTPException


class DatasetNotFound(HTTPException):
    """Raised when a dataset_id cannot be resolved."""

    def __init__(self, dataset_id: str) -> None:
        super().__init__(status_code=404, detail=f"Dataset '{dataset_id}' not found.")


class DatasetNotSliceable(HTTPException):
    """Raised when slicing is not supported for the dataset type."""

    def __init__(self, msg: str) -> None:
        super().__init__(status_code=422, detail=msg)


class InvalidSlice(ValueError, HTTPException):
    """Raised when a slice specification is syntactically or semantically invalid.

    Inherits from both ValueError (so pytest.raises(ValueError) catches it in
    unit tests) and HTTPException so that FastAPI can render it as a 400 response
    when it propagates to a route handler.
    """

    def __init__(self, msg: str) -> None:
        ValueError.__init__(self, msg)
        HTTPException.__init__(self, status_code=400, detail=msg)


class MetadataUnavailable(Exception):
    """Raised when sidecar / index metadata cannot be loaded."""


class OptionalDependencyMissing(Exception):
    """Raised when an optional package is required but not installed."""

    def __init__(self, package: str, feature: str) -> None:
        super().__init__(
            f"Optional dependency '{package}' is required for {feature}. "
            f"Install it with: pip install {package}"
        )
        self.package = package
        self.feature = feature


class VectorShapeMismatch(HTTPException):
    """Raised when request vectors have wrong shape for the target dataset."""

    def __init__(self, msg: str) -> None:
        super().__init__(status_code=422, detail=msg)


class VectorDtypeMismatch(HTTPException):
    """Raised when request vectors have dtype incompatible with the target dataset."""

    def __init__(self, msg: str) -> None:
        super().__init__(status_code=422, detail=msg)
