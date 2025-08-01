import asyncio
import json
import aiohttp
from limin_bench.application_run import (
    ApplicationConfiguration,
    ApplicationDataset,
    ApplicationJudge,
    ApplicationJudgeResult,
    generate_application_run,
    generate_application_evaluation_run,
)


async def explain_word_via_api(word: str) -> str:
    """Call the explain API endpoint to get an explanation for a word."""
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "http://localhost:8000/explain",
            headers={"Content-Type": "application/json"},
            data=json.dumps({"word": word}),
        ) as response:
            result = await response.json()
            return result


async def judge_fn(input_data: dict, output_data: dict) -> ApplicationJudgeResult:
    """Judge whether the explanation is good or not."""
    word = input_data["word"]
    explanation = output_data["explanation"].lower()

    # Simple heuristic: explanation should contain the word and be reasonably long
    contains_word = word.lower() in explanation
    is_reasonable_length = len(explanation.split()) >= 5

    is_good = contains_word and is_reasonable_length

    explanation_text = None
    if not is_good:
        reasons = []
        if not contains_word:
            reasons.append(f"doesn't mention the word '{word}'")
        if not is_reasonable_length:
            reasons.append("is too short")
        explanation_text = f"Explanation {' and '.join(reasons)}."

    return ApplicationJudgeResult(value=is_good, explanation=explanation_text)


async def main():
    # Configuration for calling the explain API
    application_configuration = ApplicationConfiguration(exec_fn=explain_word_via_api)

    # Dataset with 5 words to explain
    application_dataset = ApplicationDataset(
        rows=[
            {"word": "serendipity"},
            {"word": "ephemeral"},
            {"word": "ubiquitous"},
            {"word": "mellifluous"},
            {"word": "petrichor"},
        ]
    )

    # Judge to evaluate explanations
    application_judge = ApplicationJudge(callback=judge_fn)

    # Generate application run
    application_run = await generate_application_run(
        application_dataset, application_configuration
    )
    print("Application Run Results:")
    print(application_run)

    # Generate evaluation run
    application_evaluation_run = await generate_application_evaluation_run(
        application_run, application_judge
    )
    print("\nEvaluation Results:")
    print(application_evaluation_run)


if __name__ == "__main__":
    asyncio.run(main())
