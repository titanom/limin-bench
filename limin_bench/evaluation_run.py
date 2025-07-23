import asyncio
from limin import (
    generate_text_completion,
    generate_structured_completion,
)
from tqdm import tqdm

from .base import (
    BinaryEvaluationRun,
    BinaryEvaluationRunRow,
    BinaryEvaluationRunRowResult,
    BinaryJudge,
    ExplainedBinaryJudgement,
    ExplainedLikertJudgement,
    LikertEvaluationRun,
    LikertEvaluationRunRow,
    LikertEvaluationRunRowResult,
    LikertJudge,
    ModelRun,
    ModelRunRow,
)


async def generate_evaluation_run_row_binary(
    model_run_row: ModelRunRow,
    binary_judge: BinaryJudge,
    n_stability_runs: int = 1,
    structured: bool = False,
) -> BinaryEvaluationRunRow:
    results: list[BinaryEvaluationRunRowResult] = []

    conversation = model_run_row.content

    for _ in range(n_stability_runs):
        text = conversation.to_markdown()

        if structured:
            structured_response = await generate_structured_completion(
                text,
                response_model=ExplainedBinaryJudgement,
                model_configuration=binary_judge.model_configuration,
                system_prompt=binary_judge.system_prompt,
            )

            value = BinaryEvaluationRunRowResult(
                judge_response=structured_response.content.explanation,
                value=structured_response.content.value,
                explanation=structured_response.content.explanation,
            )
            results.append(value)
        else:
            response = await generate_text_completion(
                text,
                model_configuration=binary_judge.model_configuration,
                system_prompt=binary_judge.system_prompt,
            )

            if binary_judge.response_callback is None:
                raise ValueError("Callback is required if structured is False")

            callback_value = binary_judge.response_callback(response.content)
            results.append(
                BinaryEvaluationRunRowResult(
                    judge_response=response.content,
                    value=callback_value,
                    explanation=None,
                )
            )

    return BinaryEvaluationRunRow(
        conversation=conversation,
        results=results,
    )


async def generate_evaluation_run_binary(
    model_run: ModelRun,
    binary_judge: BinaryJudge,
    structured: bool = False,
    n_stability_runs: int = 1,
    n_parallel: int = 5,
    show_progress: bool = True,
) -> BinaryEvaluationRun:
    if binary_judge.response_callback is None and not structured:
        raise ValueError("Callback is required if structured is False")

    evaluations: list[BinaryEvaluationRunRow] = []

    if show_progress:
        progress_bar = tqdm(
            total=len(model_run), desc="Generating binary evaluation run"
        )

    for i in range(0, len(model_run), n_parallel):
        batch = model_run.rows[i : i + n_parallel]

        tasks = [
            asyncio.create_task(
                generate_evaluation_run_row_binary(
                    row, binary_judge, n_stability_runs, structured
                )
            )
            for row in batch
        ]

        evaluations_batch = await asyncio.gather(*tasks)
        evaluations.extend(evaluations_batch)

        if show_progress:
            progress_bar.update(len(batch))

    if show_progress:
        progress_bar.close()

    return BinaryEvaluationRun(rows=evaluations)


async def generate_evaluation_run_row_likert(
    model_run_row: ModelRunRow,
    likert_judge: LikertJudge,
    n_stability_runs: int = 1,
    structured: bool = False,
) -> LikertEvaluationRunRow:
    results: list[LikertEvaluationRunRowResult] = []

    conversation = model_run_row.content

    for _ in range(n_stability_runs):
        text = conversation.to_markdown()

        if structured:
            structured_response = await generate_structured_completion(
                text,
                response_model=ExplainedLikertJudgement,
                model_configuration=likert_judge.model_configuration,
                system_prompt=likert_judge.system_prompt,
            )

            value = LikertEvaluationRunRowResult(
                judge_response=structured_response.content.explanation,
                value=structured_response.content.value,
                explanation=structured_response.content.explanation,
            )
            results.append(value)
        else:
            response = await generate_text_completion(
                text,
                model_configuration=likert_judge.model_configuration,
                system_prompt=likert_judge.system_prompt,
            )

            if likert_judge.callback is None:
                raise ValueError("Callback is required if structured is False")

            callback_value = likert_judge.callback(response.content)
            results.append(
                LikertEvaluationRunRowResult(
                    judge_response=response.content,
                    value=callback_value,
                    explanation=None,
                )
            )

    return LikertEvaluationRunRow(
        conversation=conversation,
        results=results,
    )


async def generate_evaluation_run_likert(
    model_run: ModelRun,
    likert_judge: LikertJudge,
    structured: bool = False,
    n_stability_runs: int = 1,
    n_parallel: int = 5,
    show_progress: bool = True,
) -> LikertEvaluationRun:
    if likert_judge.callback is None and not structured:
        raise ValueError("Callback is required if structured is False")

    evaluations: list[LikertEvaluationRunRow] = []

    if show_progress:
        progress_bar = tqdm(
            total=len(model_run), desc="Generating likert evaluation run"
        )

    for i in range(0, len(model_run), n_parallel):
        batch = model_run.rows[i : i + n_parallel]

        tasks = [
            asyncio.create_task(
                generate_evaluation_run_row_likert(
                    row, likert_judge, n_stability_runs, structured
                )
            )
            for row in batch
        ]

        evaluations_batch = await asyncio.gather(*tasks)
        evaluations.extend(evaluations_batch)

        if show_progress:
            progress_bar.update(len(batch))

    if show_progress:
        progress_bar.close()

    return LikertEvaluationRun(rows=evaluations)
