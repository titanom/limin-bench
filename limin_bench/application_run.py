import asyncio
from typing import Callable, TypeVar, Awaitable
from pydantic import BaseModel

from .base import (
    BinaryEvaluationRun,
    BinaryEvaluationRunRow,
    BinaryEvaluationRunRowResult,
)


Input = TypeVar("Input")
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
    dataset: ApplicationDataset, configuration: ApplicationConfiguration
):
    rows = []
    for row in dataset.rows:
        output = await configuration.exec_fn(**row)
        rows.append(ApplicationRunRow(input=row, output=output))
    return ApplicationRun(rows=rows)


class ApplicationJudgeResult(BaseModel):
    value: bool
    explanation: str | None = None


class ApplicationJudge:
    def __init__(self, callback: Callable[[Input, Output], ApplicationJudgeResult]):
        self.callback = callback


async def generate_application_evaluation_run(
    application_run: ApplicationRun, application_judge: ApplicationJudge
):
    rows = []
    for row in application_run.rows:
        judge_result = application_judge.callback(row.input, row.output)
        rows.append(
            BinaryEvaluationRunRow(
                results=[
                    BinaryEvaluationRunRowResult(
                        value=judge_result.value, explanation=judge_result.explanation
                    )
                ]
            )
        )
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
    application_judge = ApplicationJudge(
        callback=lambda input, output: ApplicationJudgeResult(
            value=output == input["x"] + input["y"],
            explanation=f"The output {output} is not equal to the sum of {input['x']} and {input['y']}."
            if output != input["x"] + input["y"]
            else None,
        ),
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
