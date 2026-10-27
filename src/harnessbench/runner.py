from __future__ import annotations

from dataclasses import dataclass

from harnessbench.model import ModelAdapter
from harnessbench.pricing import cost_usd
from harnessbench.task import Task


@dataclass(frozen=True)
class RunResult:
    model: str
    task_id: str
    category: str
    passed: bool
    latency_s: float
    tokens_in: int
    tokens_out: int
    cost_usd: float | None


class BenchmarkRunner:
    def __init__(self, models: list[ModelAdapter], tasks: list[Task]) -> None:
        self._models = models
        self._tasks = tasks

    def run(self) -> list[RunResult]:
        results: list[RunResult] = []
        for model in self._models:
            for task in self._tasks:
                response = model.call(task.prompt)
                passed = task.verify(response.text)
                results.append(
                    RunResult(
                        model=model.name,
                        task_id=task.id,
                        category=task.category,
                        passed=passed,
                        latency_s=response.latency_s,
                        tokens_in=response.tokens_in,
                        tokens_out=response.tokens_out,
                        cost_usd=cost_usd(model.name, response.tokens_in, response.tokens_out),
                    )
                )
        return results
