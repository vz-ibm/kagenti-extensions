"""Console/`python -m sparc_service` entry point."""

from __future__ import annotations

import uvicorn

from .settings import Settings


def main() -> None:
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s", datefmt="%Y-%m-%dT%H:%M:%SZ")
    # Demote noisy third-party loggers — their INFO adds no operational value
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    settings = Settings.from_env()
    uvicorn.run("sparc_service.api:app", host=settings.host, port=settings.port, log_level="info")


if __name__ == "__main__":
    main()
