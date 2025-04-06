import asyncio
from limin import (
    generate_text_completion,
    generate_structured_completion,
)
from tqdm import tqdm

from .base import (
    BinaryEvaluationRun,
    BinaryEvaluationRunRow,
    BinaryJudge,
    ExplainedBinaryJudgement,
    ExplainedLikertJudgement,
    LikertEvaluationRun,
    LikertEvaluationRunRow,
    LikertJudge,
    ModelRun,
    ModelRunRow,
)


async def generate_evaluation_run_row_binary(
    model_run_row: ModelRunRow, binary_judge: BinaryJudge, structured: bool = False
) -> BinaryEvaluationRunRow:
    conversation = model_run_row.content
    text = conversation.to_markdown()

    if structured:
        structured_response = await generate_structured_completion(
            text,
            response_model=ExplainedBinaryJudgement,
            model_configuration=binary_judge.model_configuration,
            system_prompt=binary_judge.system_prompt,
        )
        return BinaryEvaluationRunRow(
            conversation=conversation,
            result=structured_response.content.result,
            explanation=structured_response.content.explanation,
            judge_response=structured_response.content.explanation,
        )
    else:
        response = await generate_text_completion(
            text,
            model_configuration=binary_judge.model_configuration,
            system_prompt=binary_judge.system_prompt,
        )

        if binary_judge.response_callback is None:
            raise ValueError("Callback is required if structured is False")

        result = binary_judge.response_callback(response.content)
        return BinaryEvaluationRunRow(
            conversation=conversation,
            result=result,
            judge_response=response.content,
        )


async def generate_evaluation_run_binary(
    model_run: ModelRun,
    binary_judge: BinaryJudge,
    structured: bool = False,
    n_parallel: int = 5,
    show_progress: bool = True,
) -> BinaryEvaluationRun:
    if binary_judge.response_callback is None and not structured:
        raise ValueError("Callback is required if structured is False")

    evaluations = []

    if show_progress:
        progress_bar = tqdm(
            total=len(model_run), desc="Generating binary evaluation run"
        )

    for i in range(0, len(model_run), n_parallel):
        batch = model_run.rows[i : i + n_parallel]

        tasks = [
            asyncio.create_task(
                generate_evaluation_run_row_binary(row, binary_judge, structured)
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
    model_run_row: ModelRunRow, likert_judge: LikertJudge, structured: bool = False
) -> LikertEvaluationRunRow:
    conversation = model_run_row.content
    text = conversation.to_markdown()

    if structured:
        structured_response = await generate_structured_completion(
            text,
            response_model=ExplainedLikertJudgement,
            model_configuration=likert_judge.model_configuration,
            system_prompt=likert_judge.system_prompt,
        )
        return LikertEvaluationRunRow(
            conversation=conversation,
            result=structured_response.content.result,
            explanation=structured_response.content.explanation,
            judge_response=structured_response.content.explanation,
        )
    else:
        response = await generate_text_completion(
            text,
            model_configuration=likert_judge.model_configuration,
            system_prompt=likert_judge.system_prompt,
        )

        if likert_judge.callback is None:
            raise ValueError("Callback is required if structured is False")

        result = likert_judge.callback(response.content)
        return LikertEvaluationRunRow(
            conversation=conversation,
            result=result,
            judge_response=response.content,
        )


async def generate_evaluation_run_likert(
    model_run: ModelRun,
    likert_judge: LikertJudge,
    structured: bool = False,
    n_parallel: int = 5,
    show_progress: bool = True,
) -> LikertEvaluationRun:
    evaluations = []

    if show_progress:
        progress_bar = tqdm(
            total=len(model_run), desc="Generating likert evaluation run"
        )

    for i in range(0, len(model_run), n_parallel):
        batch = model_run.rows[i : i + n_parallel]

        tasks = [
            asyncio.create_task(
                generate_evaluation_run_row_likert(row, likert_judge, structured)
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
