from __future__ import annotations

import argparse
import re

from harnessbench.model import MockAdapter
from harnessbench.report import summarize, to_markdown
from harnessbench.runner import BenchmarkRunner
from harnessbench.task import Task

_DEMO_TASKS = [
    Task(
        id="echo-number",
        prompt="Reply with exactly the digit 7 and nothing else.",
        verify=lambda text: bool(re.fullmatch(r"\s*7\s*", text)),
    ),
]


def main() -> None:
    parser = argparse.ArgumentParser(prog="harnessbench")
    parser.add_argument("--demo", action="store_true", help="run a mock adapter against the demo task set")
    args = parser.parse_args()

    if args.demo:
        model = MockAdapter("demo-mock", lambda _: "7")
        results = BenchmarkRunner([model], _DEMO_TASKS).run()
        print(to_markdown(summarize(results)))
        return

    parser.print_help()


if __name__ == "__main__":
    main()
