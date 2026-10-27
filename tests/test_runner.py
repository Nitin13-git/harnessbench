from harnessbench.model import MockAdapter
from harnessbench.report import summarize, to_markdown
from harnessbench.runner import BenchmarkRunner
from harnessbench.task import Task


def _task(task_id: str, expected: str) -> Task:
    return Task(id=task_id, prompt=f"say {expected}", verify=lambda text: text.strip() == expected)


def test_runner_grades_each_model_against_each_task() -> None:
    good = MockAdapter("good-model", lambda prompt: prompt.removeprefix("say "))
    bad = MockAdapter("bad-model", lambda _: "wrong")
    tasks = [_task("t1", "yes"), _task("t2", "no")]

    results = BenchmarkRunner([good, bad], tasks).run()

    assert len(results) == 4
    good_results = [r for r in results if r.model == "good-model"]
    bad_results = [r for r in results if r.model == "bad-model"]
    assert all(r.passed for r in good_results)
    assert not any(r.passed for r in bad_results)


def test_summarize_ranks_models_by_success_rate() -> None:
    perfect = MockAdapter("perfect", lambda prompt: prompt.removeprefix("say "))
    flaky = MockAdapter("flaky", lambda prompt: "wrong" if "no" in prompt else prompt.removeprefix("say "))
    tasks = [_task("t1", "yes"), _task("t2", "no")]

    results = BenchmarkRunner([perfect, flaky], tasks).run()
    summaries = summarize(results)

    assert summaries[0].model == "perfect"
    assert summaries[0].success_rate == 1.0
    assert summaries[1].model == "flaky"
    assert summaries[1].success_rate == 0.5


def test_to_markdown_renders_a_table_row_per_model() -> None:
    model = MockAdapter("m", lambda _: "yes")
    results = BenchmarkRunner([model], [_task("t1", "yes")]).run()
    table = to_markdown(summarize(results))
    assert "| m |" in table
    assert "100%" in table
