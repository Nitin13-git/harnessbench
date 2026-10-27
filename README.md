# HarnessBench

Run the same task set across many models and get one report back: success
rate, latency, cost, and context-token usage per model — so "which model
should run inside my harness" has a number behind it instead of a guess.

## How it works

- **`Task`** — a prompt plus an independent `verify(text) -> bool` function.
  The model being graded never grades itself.
- **`ModelAdapter`** — one interface, two implementations: `MockAdapter`
  (deterministic, for tests/dry runs) and `OpenAICompatibleAdapter`, which
  talks to any OpenAI-shaped `/chat/completions` endpoint — covering most
  open-weight models (Qwen, DeepSeek, GLM, Kimi, MiniMax, GPT-OSS, Devstral)
  through a provider like OpenRouter or a self-hosted vLLM server, without a
  dedicated SDK per model.
- **`BenchmarkRunner`** — runs every model against every task, collects
  pass/fail, latency, and token counts.
- **`summarize` / `to_markdown`** — aggregates per-model and renders a
  comparison table.

Pricing in `harnessbench.pricing` ships empty on purpose — provider rates
change too often to hardcode. Fill in `PRICE_PER_MILLION_TOKENS` with your
own current numbers before trusting the cost column.

## Quickstart

```bash
pip install -e ".[dev]"
harnessbench --demo
pytest
```

```python
from harnessbench import BenchmarkRunner, MockAdapter, OpenAICompatibleAdapter, Task, summarize, to_markdown

tasks = [Task(id="t1", prompt="...", verify=lambda text: "expected" in text)]

models = [
    OpenAICompatibleAdapter("qwen3-235b", base_url="https://openrouter.ai/api/v1",
                             api_key="...", model="qwen/qwen3-235b-a22b"),
    OpenAICompatibleAdapter("deepseek-v3.2", base_url="https://openrouter.ai/api/v1",
                             api_key="...", model="deepseek/deepseek-v3.2"),
]

results = BenchmarkRunner(models, tasks).run()
print(to_markdown(summarize(results)))
```

## Status

v0.1. Task verification is independent and deterministic by design — if you
need an LLM-judge task type, write a `Task` whose `verify` calls out to a
separate judge model explicitly, so that dependency stays visible rather
than hidden inside the runner.

## License

MIT
