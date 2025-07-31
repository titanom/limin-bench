from typing import AsyncCallable, Callable, TypeVar
from limin import ModelConfiguration
from pydantic import BaseModel

from .base import BinaryEvaluationRun, BinaryEvaluationRunRowResult


Input = TypeVar("Input")
Output = TypeVar("Output")


class ApplicationConfiguration:
    exec_fn: AsyncCallable[[Input], Output]


class ApplicationDataset:
    rows: list[Input]


class ApplicationRunRow:
    input: Input
    output: Output


class ApplicationRun:
    rows: list[ApplicationRunRow]


async def generate_application_run(
    dataset: ApplicationDataset, configuration: ApplicationConfiguration
):
    rows = []
    for row in dataset.rows:
        output = await configuration.exec_fn(row)
        rows.append(ApplicationRunRow(input=row, output=output))
    return ApplicationRun(rows=rows)


class ApplicationJudgeResult(BaseModel):
    value: bool
    explanation: str | None = None


class ApplicationJudge:
    model_configuration: ModelConfiguration
    system_prompt: str
    callback: Callable[[Output], ApplicationJudgeResult]


async def generate_application_evaluation_run(
    application_run: ApplicationRun, application_judge: ApplicationJudge
):
    rows = []
    for row in application_run.rows:
        output = row.output
        judge_result = application_judge.callback(output)
        rows.append(
            BinaryEvaluationRunRowResult(
                value=judge_result.value, explanation=judge_result.explanation
            )
        )
    return BinaryEvaluationRun(rows=rows)
