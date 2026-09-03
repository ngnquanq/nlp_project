from __future__ import annotations

import uvicorn

from mt_pipeline.serving.settings import UISettings


def main() -> None:
    settings = UISettings.for_e2_sidecar()
    uvicorn.run(
        "mt_pipeline.serving.e2_app:app",
        host="127.0.0.1",
        port=settings.port,
        workers=1,
        access_log=True,
    )


if __name__ == "__main__":
    main()
