from __future__ import annotations


class ServiceError(RuntimeError):
    """A failure that maps directly onto an HTTP response for the local UI."""

    def __init__(self, status_code: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
