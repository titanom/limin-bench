import asyncio
from dotenv import load_dotenv
from limin import ModelConfiguration
from limin_bench import (
    BinaryJudge,
    Dataset,
    generate_evaluation_run_binary,
    generate_multi_turn_model_run,
)


dataset = Dataset(
    rows=[
        "Question: What is 2+2?",
        "Question: What is 2+3?",
    ]
)

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

assistant_system_prompt = """
You are a helpful assistant.
Answer the user's questions factually correctly.
Begin every message with 'Answer:'.
""".strip()

user_simulator_system_prompt = """
You are a student.
Ask questions about mathematical concepts.
Only ask questions, don't do anything else.
Begin every message with 'Question:'.
""".strip()


async def main():
    model_run = await generate_multi_turn_model_run(
        dataset=dataset,
        assistant_system_prompt=assistant_system_prompt,
        assistant_model_configuration=ModelConfiguration(model="gpt-4o"),
        user_simulator_system_prompt=user_simulator_system_prompt,
        user_simulator_model_configuration=ModelConfiguration(model="gpt-4o"),
        n_turns=3,
    )
    print("Full model run:")
    print(model_run.to_markdown_table())

    evaluation_run = await generate_evaluation_run_binary(
        model_run=model_run, binary_judge=judge
    )

    print("Full evaluation run:")
    print(evaluation_run.to_markdown_table())
    print("Accuracy:")
    print(evaluation_run.accuracy)


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())
