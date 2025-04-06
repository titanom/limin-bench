import asyncio

from limin import ModelConfiguration, generate_text_completion
from limin_talk import Character, talk
from tqdm import tqdm
from .base import Dataset, ModelRun, ModelRunRow, get_conversation_from_prompts


async def generate_model_run_row(
    text: str,
    system_prompt: str | None = None,
    model_configuration: ModelConfiguration | None = None,
) -> ModelRunRow:
    response = await generate_text_completion(
        text, system_prompt=system_prompt, model_configuration=model_configuration
    )
    conversation = get_conversation_from_prompts(
        system_prompt=system_prompt, user_prompt=text, assistant_prompt=response.content
    )
    return ModelRunRow(content=conversation)


async def generate_model_run(
    dataset: Dataset,
    system_prompt: str | None = None,
    n_parallel: int = 5,
    model_configuration: ModelConfiguration | None = None,
    show_progress: bool = True,
) -> ModelRun:
    model_run_rows = []

    if show_progress:
        progress_bar = tqdm(total=len(dataset), desc="Generating model run rows")

    for i in range(0, len(dataset), n_parallel):
        dataset_batch = dataset.rows[i : i + n_parallel]

        tasks = [
            asyncio.create_task(
                generate_model_run_row(
                    text=text,
                    system_prompt=system_prompt,
                    model_configuration=model_configuration,
                )
            )
            for text in dataset_batch
        ]

        model_run_rows_batch = await asyncio.gather(*tasks)
        model_run_rows.extend(model_run_rows_batch)

        if show_progress:
            progress_bar.update(len(dataset_batch))

    if show_progress:
        progress_bar.close()

    return ModelRun(rows=model_run_rows)


async def generate_multi_turn_model_run_row(
    text: str,
    assistant_system_prompt: str,
    assistant_model_configuration: ModelConfiguration,
    user_simulator_system_prompt: str,
    user_simulator_model_configuration: ModelConfiguration,
    n_turns: int = 2,
) -> ModelRunRow:
    conversation = await talk(
        user_character=Character(
            system_prompt=user_simulator_system_prompt,
            model_configuration=user_simulator_model_configuration,
        ),
        assistant_character=Character(
            system_prompt=assistant_system_prompt,
            model_configuration=assistant_model_configuration,
        ),
        n_turns=n_turns,
        initial_message=text,
    )
    return ModelRunRow(content=conversation)


async def generate_multi_turn_model_run(
    dataset: Dataset,
    assistant_system_prompt: str,
    assistant_model_configuration: ModelConfiguration,
    user_simulator_system_prompt: str,
    user_simulator_model_configuration: ModelConfiguration,
    n_turns: int = 2,
    n_parallel: int = 5,
    show_progress: bool = True,
) -> ModelRun:
    model_run_rows = []

    if show_progress:
        progress_bar = tqdm(total=len(dataset), desc="Generating model run rows")

    for i in range(0, len(dataset), n_parallel):
        dataset_batch = dataset.rows[i : i + n_parallel]

        tasks = [
            asyncio.create_task(
                generate_multi_turn_model_run_row(
                    text,
                    assistant_system_prompt=assistant_system_prompt,
                    assistant_model_configuration=assistant_model_configuration,
                    user_simulator_system_prompt=user_simulator_system_prompt,
                    user_simulator_model_configuration=user_simulator_model_configuration,
                    n_turns=n_turns,
                )
            )
            for text in dataset_batch
        ]

        model_run_rows_batch = await asyncio.gather(*tasks)
        model_run_rows.extend(model_run_rows_batch)

        if show_progress:
            progress_bar.update(len(dataset_batch))

    if show_progress:
        progress_bar.close()

    return ModelRun(rows=model_run_rows)
