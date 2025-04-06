import asyncio
from limin import ModelConfiguration
from limin_bench import (
    BinaryJudge,
    Dataset,
    generate_evaluation_run_binary,
    generate_model_run,
)


dataset = Dataset(
    rows=[
        "What is the capital of France?",
        "What is the capital of Germany?",
    ]
)

assistant_system_prompt = """
You are a helpful assistant.
Answer the user's questions factually correctly.
"""

judge_system_prompt = """
You are an LLM as a judge.
You will be given a conversation between a user and an assistant.
You will then judge whether the assistants answers are factually correct or not.
Return 'true' if the answers are factually correct, and 'false' if they are not.
You should also return a short one-sentence explanation for your judgement.
"""

binary_judge = BinaryJudge(
    model_configuration=ModelConfiguration(model="gpt-4o"),
    system_prompt=judge_system_prompt,
)


async def main():
    model_run = await generate_model_run(
        dataset=dataset,
        system_prompt=assistant_system_prompt,
        model_configuration=ModelConfiguration(model="gpt-4o"),
    )
    print("Full model run:")
    print(model_run.model_dump_json())

    evaluation_run = await generate_evaluation_run_binary(
        model_run=model_run, binary_judge=binary_judge, structured=True
    )

    print("Full evaluation run:")
    print(evaluation_run.model_dump_json())
    print("Accuracy:")
    print(evaluation_run.accuracy)


if __name__ == "__main__":
    asyncio.run(main())
