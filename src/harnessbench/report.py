from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from harnessbench.runner import RunResult


@dataclass(frozen=True)
class ModelSummary:
    model: str
    tasks_run: int
    success_rate: float
    avg_latency_s: float
    total_cost_usd: float | None
    avg_context_tokens: float


def summarize(results: list[RunResult]) -> list[ModelSummary]:
    by_model: dict[str, list[RunResult]] = defaultdict(list)
    for result in results:
        by_model[result.model].append(result)

    summaries = []
    for model, rows in by_model.items():
        costs = [r.cost_usd for r in rows if r.cost_usd is not None]
        summaries.append(
            ModelSummary(
                model=model,
                tasks_run=len(rows),
                success_rate=sum(r.passed for r in rows) / len(rows),
                avg_latency_s=sum(r.latency_s for r in rows) / len(rows),
                total_cost_usd=sum(costs) if costs else None,
                avg_context_tokens=sum(r.tokens_in + r.tokens_out for r in rows) / len(rows),
            )
        )
    return sorted(summaries, key=lambda s: s.success_rate, reverse=True)


def to_markdown(summaries: list[ModelSummary]) -> str:
    lines = [
        "| model | success rate | avg latency (s) | total cost ($) | avg tokens |",
        "|---|---|---|---|---|",
    ]
    for s in summaries:
        cost = f"{s.total_cost_usd:.4f}" if s.total_cost_usd is not None else "n/a"
        lines.append(
            f"| {s.model} | {s.success_rate:.0%} | {s.avg_latency_s:.2f} | {cost} | {s.avg_context_tokens:.0f} |"
        )
    return "\n".join(lines)
