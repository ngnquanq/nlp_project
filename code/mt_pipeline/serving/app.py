"""The default E1 gateway application.

Importing this module builds an app from the process environment. The E2
sidecar imports :mod:`mt_pipeline.serving.factory` instead, so it does not
inherit that side effect.
"""

from __future__ import annotations

from mt_pipeline.serving.errors import ServiceError
from mt_pipeline.serving.factory import create_app


__all__ = ["ServiceError", "create_app", "app"]

app = create_app()
