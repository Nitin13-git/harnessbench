from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class Task:
    """One benchmark task.

    `verify` receives the model's raw text output and returns True if the
    task was solved. Keep verification independent of the model being
    graded — never ask a model to grade itself.
    """

    id: str
    prompt: str
    verify: Callable[[str], bool]
    category: str = "general"
