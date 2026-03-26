"""Route registration modules for Xinference REST API.

Each module provides a ``register_routes(api)`` function that binds
domain-specific routes to ``api._router``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from . import admin, audio, embeddings, images, llm, models, rerank, videos

if TYPE_CHECKING:
    from ..restful_api import RESTfulAPI


def register_all_routes(api: RESTfulAPI) -> None:
    """Register all domain routes on the given RESTfulAPI instance."""
    admin.register_routes(api)
    models.register_routes(api)
    llm.register_routes(api)
    embeddings.register_routes(api)
    rerank.register_routes(api)
    audio.register_routes(api)
    images.register_routes(api)
    videos.register_routes(api)
