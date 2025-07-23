from limin_bench.base import BinaryJudge, PregeneratedMultiTurnDataset, LikertJudge
from limin_bench.evaluation_run import (
    generate_evaluation_run_binary,
    generate_evaluation_run_likert,
)
from limin_bench.model_run import (
    generate_multi_turn_model_run_from_pregenerated_dataset,
)
from limin import ModelConfiguration


async def main():
    pregenerated_dataset = PregeneratedMultiTurnDataset(
        rows=[
            [
                "Compose an short travel blog post about a recent trip to Hawaii, highlighting cultural experiences and must-see attractions (4 sentences).",
                "Rewrite your previous response. Start every sentence with the letter A.",
            ],
            [
                "Draft a professional email seeking your supervisors feedback on the 'Quarterly Financial Report' you prepared. Ask specifically about the data analysis, presentation style, and the clarity of conclusions drawn. Keep the email short and to the point.",
                "Rewrite your previous response. Start every sentence with the letter B.",
            ],
        ]
    )

    model_run = await generate_multi_turn_model_run_from_pregenerated_dataset(
        dataset=pregenerated_dataset,
        assistant_system_prompt="You are a helpful assistant.",
        assistant_model_configuration=ModelConfiguration(model="gpt-4o"),
    )

    print(model_run.model_dump_json())

    evaluation_run = await generate_evaluation_run_likert(
        model_run=model_run,
        likert_judge=LikertJudge(
            model_configuration=ModelConfiguration(model="gpt-4o"),
            system_prompt="You are given a conversation between a user and an assistant. You are to judge whether the assistant's response is helpful or not. Return an explanation for your judgement and a value from 1-4 indicating whether the response is helpful (3, 4) or not (1, 2).",
        ),
        n_stability_runs=2,
        structured=True,
    )

    print(evaluation_run.model_dump_json())


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
