"""Cost-per-task depends on provider pricing, which changes often and
varies by where you call a model from (direct API vs. OpenRouter vs.
self-hosted vLLM). This module intentionally ships empty — fill in your
own current $/million-token rates before trusting `summarize()`'s cost
column.
"""

from __future__ import annotations

# model name -> (input $ / 1M tokens, output $ / 1M tokens)
PRICE_PER_MILLION_TOKENS: dict[str, tuple[float, float]] = {}


def cost_usd(model: str, tokens_in: int, tokens_out: int) -> float | None:
    rates = PRICE_PER_MILLION_TOKENS.get(model)
    if rates is None:
        return None
    price_in, price_out = rates
    return (tokens_in * price_in + tokens_out * price_out) / 1_000_000
