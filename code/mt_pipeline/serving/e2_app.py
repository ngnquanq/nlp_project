"""The E2 sidecar application.

Runs in .conda/e2-ui (torch 2.x + transformers + peft + bitsandbytes), which
cannot host E1 because fairseq 0.12.2 requires numpy<2. Serves no frontend and
knows about no model other than E2.
"""

from __future__ import annotations

from mt_pipeline.serving.e2_translator import build_e2_translator
from mt_pipeline.serving.factory import create_app
from mt_pipeline.serving.registry import E2_KEY
from mt_pipeline.serving.settings import UISettings


def create_e2_app(settings: UISettings | None = None, translator_factory=None):
    return create_app(
        settings or UISettings.for_e2_sidecar(),
        e2_translator_factory=translator_factory or build_e2_translator,
        models=[E2_KEY],
    )


app = create_e2_app()
