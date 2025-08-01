import asyncio
from typing import Callable, TypeVar, Awaitable, Any
from pydantic import BaseModel

from .base import (
    BinaryEvaluationRun,
    BinaryEvaluationRunRow,
    BinaryEvaluationRunRowResult,
)


Input = TypeVar("Input", bound=dict[str, Any])
Output = TypeVar("Output")


class ApplicationConfiguration:
    def __init__(self, exec_fn: Callable[[Input], Awaitable[Output]]):
        self.exec_fn = exec_fn


class ApplicationDataset:
    def __init__(self, rows: list[Input]):
        self.rows = rows


class ApplicationRunRow:
    def __init__(self, input: Input, output: Output):
        self.input = input
        self.output = output

    def __repr__(self):
        return f"ApplicationRunRow(input={self.input}, output={self.output})"


class ApplicationRun:
    def __init__(self, rows: list[ApplicationRunRow]):
        self.rows = rows

    def __repr__(self):
        return f"ApplicationRun(rows={self.rows})"


async def generate_application_run(
    dataset: ApplicationDataset,
    configuration: ApplicationConfiguration,
    n_parallel: int = 5,
    show_progress: bool = True,
):
    from tqdm import tqdm

    rows = []

    if show_progress:
        progress_bar = tqdm(
            total=len(dataset.rows), desc="Generating application run rows"
        )

    for i in range(0, len(dataset.rows), n_parallel):
        dataset_batch = dataset.rows[i : i + n_parallel]

        tasks = [
            asyncio.create_task(configuration.exec_fn(**row)) for row in dataset_batch
        ]

        outputs_batch = await asyncio.gather(*tasks)

        for row, output in zip(dataset_batch, outputs_batch):
            rows.append(ApplicationRunRow(input=row, output=output))

        if show_progress:
            progress_bar.update(len(dataset_batch))

    if show_progress:
        progress_bar.close()

    return ApplicationRun(rows=rows)


class ApplicationJudgeResult(BaseModel):
    value: bool
    explanation: str | None = None


class ApplicationJudge:
    def __init__(
        self, callback: Callable[[Input, Awaitable[Output]], ApplicationJudgeResult]
    ):
        self.callback = callback


async def generate_application_evaluation_run(
    application_run: ApplicationRun,
    application_judge: ApplicationJudge,
    n_parallel: int = 5,
    show_progress: bool = True,
):
    from tqdm import tqdm

    rows = []

    if show_progress:
        progress_bar = tqdm(
            total=len(application_run.rows),
            desc="Generating application evaluation run",
        )

    for i in range(0, len(application_run.rows), n_parallel):
        batch = application_run.rows[i : i + n_parallel]

        tasks = [
            asyncio.create_task(application_judge.callback(row.input, row.output))
            for row in batch
        ]

        judge_results_batch = await asyncio.gather(*tasks)

        for judge_result in judge_results_batch:
            rows.append(
                BinaryEvaluationRunRow(
                    results=[
                        BinaryEvaluationRunRowResult(
                            value=judge_result.value,
                            explanation=judge_result.explanation,
                        )
                    ]
                )
            )

        if show_progress:
            progress_bar.update(len(batch))

    if show_progress:
        progress_bar.close()

    return BinaryEvaluationRun(rows=rows)


async def main():
    async def app_fn(x: int, y: int) -> int:
        return x + y

    application_configuration = ApplicationConfiguration(exec_fn=app_fn)
    application_dataset = ApplicationDataset(
        rows=[
            {"x": 1, "y": 2},
            {"x": 3, "y": 4},
        ]
    )

    async def judge_fn(input: dict, output: int) -> ApplicationJudgeResult:
        return ApplicationJudgeResult(
            value=output == input["x"] + input["y"],
            explanation=f"The output {output} is not equal to the sum of {input['x']} and {input['y']}."
            if output != input["x"] + input["y"]
            else None,
        )

    application_judge = ApplicationJudge(
        callback=judge_fn,
    )
    application_run = await generate_application_run(
        application_dataset, application_configuration
    )
    print(application_run)
    application_evaluation_run = await generate_application_evaluation_run(
        application_run, application_judge
    )
    print(application_evaluation_run)


if __name__ == "__main__":
    asyncio.run(main())
