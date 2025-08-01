import asyncio
from dotenv import load_dotenv
from limin import ModelConfiguration
from limin_bench import (
    BinaryEvaluationRun,
    BinaryJudge,
    Dataset,
    ModelRun,
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
""".strip()

judge_system_prompt = """
You are an LLM as a judge.
You will be given a conversation between a user and an assistant.
You will then judge whether the assistants answers are factually correct or not.
Return 'yes' if the answers are factually correct, and 'no' if they are not.
""".strip()

judge = BinaryJudge(
    model_configuration=ModelConfiguration(model="gpt-4o"),
    system_prompt=judge_system_prompt,
    response_callback=lambda x: x == "yes",
)


async def main():
    model_run = await generate_model_run(
        dataset=dataset,
        system_prompt=assistant_system_prompt,
        model_configuration=ModelConfiguration(model="gpt-4o"),
    )

    evaluation_run = await generate_evaluation_run_binary(
        model_run=model_run, binary_judge=judge
    )

    model_run.to_json_file("model_run.json")
    evaluation_run.to_json_file("evaluation_run.json")

    model_run_from_json = ModelRun.from_json_file("model_run.json")
    evaluation_run_from_json = BinaryEvaluationRun.from_json_file("evaluation_run.json")

    print("Deserialized model run:")
    print(model_run_from_json.to_markdown_table())

    print("Deserialized evaluation run:")
    print(evaluation_run_from_json.to_markdown_table(model_run=model_run_from_json))


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())
