import asyncio
from dotenv import load_dotenv
from limin import ModelConfiguration
from limin_bench import (
    Dataset,
    LikertJudge,
    generate_evaluation_run_likert,
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
""".strip()

judge_system_prompt = """
You are an LLM as a judge.
You will be given a conversation between a user and an assistant.
You will then judge whether the assistants answers are factually correct or not.
Return a number between 1 and 4 where:
- 1 means that answer is strongly incorrect
- 2 means that answer is incorrect
- 3 means that answer is correct
- 4 means that answer is strongly correct
""".strip()

likert_judge = LikertJudge(
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
    print(model_run.to_markdown_table())

    evaluation_run = await generate_evaluation_run_likert(
        model_run=model_run, likert_judge=likert_judge, structured=True
    )

    print("Full evaluation run:")
    print(evaluation_run.to_markdown_table())
    print("Average score:")
    print(evaluation_run.avg)


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())
