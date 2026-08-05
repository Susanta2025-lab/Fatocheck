"""Request-scoped context for correlation IDs.

Uses contextvars so concurrent requests do not share mutable global state.
"""

from __future__ import annotations

from contextvars import ContextVar

request_id_ctx: ContextVar[str] = ContextVar("request_id", default="-")
