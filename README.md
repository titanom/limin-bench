# limin-bench

A Python library for benchmarking and evaluating LLMs.

Features:

✅ LLM as a judge benchmarking (binary and Likert scale)

✅ Benchmarking of multi-turn conversations

✅ Serialization and deserialization of model runs and evaluation runs

✅ Structured evaluation containing judge explanations

## Installation

Install the library with pip:

```bash
python -m pip install limin-bench
```

## Usage

### The Simplest Example

After you've installed the library, you can use it by importing the `limin` module and calling the functions you need.
You will also need to provide an API key for your API either by running `export OPENAI_API_KEY=$YOUR_API_KEY` or by creating an `.env` file in the root directory of your project and adding the following line:

```
OPENAI_API_KEY=$YOUR_API_KEY
```

Now let's create a simple example that will benchmark the performance of a model on a simple dataset:

```python
import asyncio
from limin import ModelConfiguration
from limin_bench import BinaryJudge, Dataset, generate_evaluation_run_binary, generate_model_run


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
Return 'yes' if the answers are factually correct, and 'no' if they are not.
"""

binary_judge = BinaryJudge(
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
        model_run=model_run,
        binary_judge=binary_judge
    )

    print("Full evaluation run:")
    print(evaluation_run.model_dump_json())
    print("Accuracy:")
    print(evaluation_run.accuracy)


if __name__ == "__main__":
    asyncio.run(main())
```

This will print the full evaluation run and the accuracy at the end:

```
Full evaluation run:
{ ... }

Accuracy:
1.0
```

This script does the following:

- Instantiates a dataset with two rows of text.
- Instantiates an LLM as a judge with a system prompt and a response callback that is responsible for processing the judge's response into a boolean value.
- Generates a model run (i.e. runs the model on all rows of the dataset).
- Generates an evaluation run by running the judge on all rows of the model run.
- Prints the full evaluation run and the accuracy.

You can find the full example in [`examples/single_turn.py`](examples/single_turn.py).

We will now explain these steps in more detail.

### Datasets

The `Dataset` class is a simple class that contains a list of rows of text.

```python
dataset = Dataset(
    rows=[
        "What is the capital of France?",
        "What is the capital of Germany?",
    ]
)
```

The `Dataset` class is used to store the data that you want to use for benchmarking.

This data can represent:
- questions / statements for the model to answer / respond to (e.g. `What is the capital of France?` or `Write me a poem about a cat`)
- initial user messages in a multi-turn conversation if you're doing multi-turn model runs (we will cover this later)

### Model Runs

The `ModelRun` class is used to store the results of running a model on a dataset.

You get a model run by executing the model you want to benchmark on the dataset.
The model run will contain all the model responses to the dataset rows.

Consider this example:

```python
model_run = await generate_model_run(
    dataset=dataset,
    model_configuration=ModelConfiguration(model="gpt-4o"),
)
print(model_run.model_dump_json())
```

> Just like with `limin`, everything in `limin-bench` inherits from `pydantic.BaseModel` which allows for easy serialization and deserialization of the data.

This will run the `gpt-4o` model on all rows of the dataset `dataset` and store the results in the `model_run` object.

Here is how the output could look like:

```json
{
    "rows": [
        {
            "content": {
                "messages": [
                    {
                        "role": "system",
                        "content": "\nYou are a helpful assistant.\nAnswer the user's questions factually correctly.\n"
                    },
                    {
                        "role": "user",
                        "content": "What is the capital of France?"
                    },
                    {
                        "role": "assistant",
                        "content": "The capital of France is Paris."
                    }
                ]
            }
        },
        {
            "content": {
                "messages": [
                    {
                        "role": "system",
                        "content": "\nYou are a helpful assistant.\nAnswer the user's questions factually correctly.\n"
                    },
                    {
                        "role": "user",
                        "content": "What is the capital of Germany?"
                    },
                    {
                        "role": "assistant",
                        "content": "The capital of Germany is Berlin."
                    }
                ]
            }
        }
    ]
}
```

> Note that rows contain `Conversation` objects instead of just the model responses because `limin-bench` supports benchmarking multi-turn conversations.
> We will talk about this later.

### Judges

Now that we have a model run, we can use it to evaluate the model.
For that, we need a judge.
The `limin-bench` library supports binary judges (which produce a boolean result) and likert scale judges (which produce an integer result).

Here is how you can instantiate a binary judge:

```python
judge = BinaryJudge(
    model_configuration=ModelConfiguration(model="gpt-4o"),
    system_prompt=judge_system_prompt,
    response_callback=lambda x: x == "yes",
)
```

The `model_configuration` and `system_prompt` are simply the configuration for the LLM that will be used as a judge.

The `response_callback` is a bit more interesting.
Here, you need to provide a function that takes the judge's response (which is a string) and returns a boolean value.
Note that the `response_callback` needs to play nicely with the `system_prompt` of the judge.
For example, if you tell the judge to return `yes/no` responses in the `system_prompt`, your `response_callback` should return `True` if the response is `yes` and `False` if the response is `no` (or something else).

> Instantiating a `LikertJudge` is very similar, except that your `response_callback` will need to return an integer value.

### Evaluation Runs

Now that we have a model run and a judge, we can generate an evaluation run:

```python
evaluation_run = await generate_evaluation_run_binary(
    model_run=model_run,
    binary_judge=judge
)
```

This will run the judge on all rows of the model run and store each evaluation result in an `EvaluationRunRow` object.
This how the `evaluation_run` object will look like:

```json
{
    "rows": [
        {
            "conversation": {
                "messages": [
                    {
                        "role": "system",
                        "content": "\nYou are a helpful assistant.\nAnswer the user's questions factually correctly.\n"
                    },
                    {
                        "role": "user",
                        "content": "What is the capital of France?"
                    },
                    {
                        "role": "assistant",
                        "content": "The capital of France is Paris."
                    }
                ]
            },
            "judge_response": "yes",
            "result": true,
            "explanation": null
        },
        {
            "conversation": {
                "messages": [
                    {
                        "role": "system",
                        "content": "\nYou are a helpful assistant.\nAnswer the user's questions factually correctly.\n"
                    },
                    {
                        "role": "user",
                        "content": "What is the capital of Germany?"
                    },
                    {
                        "role": "assistant",
                        "content": "The capital of Germany is Berlin."
                    }
                ]
            },
            "judge_response": "yes",
            "result": true,
            "explanation": null
        }
    ]
}
```

Every evaluation run row contains the following fields:

- `conversation`: The conversation that is being judged (this is the same as the `conversation` field in the `ModelRunRow` object)
- `judge_response`: The response from the judge
- `result`: The final evaluation result (e.g. `True` or `False` for a binary judge)
- `explanation`: The explanation of the judge's response

> Aside: Why are we separating model runs and evaluation runs?
> We are doing this because often you will want to use multiple judges on the same model run. 

### Multi-Turn Benchmarking

To generate a multi-turn model run, you can use the `generate_multi_turn_model_run` function.

Here, you need to provide the dataset, the assistant model configuration and system prompt (where the assistant is the model that you want to benchmark) and the user simulator model configuration and system prompt (where the user simulator is the model that simulates the user's behavior).

Note that we need the user simulator because we can only provide the initial user message in the dataset (since we don't know how the assistant will respond, we can't provide the user's second, third, etc. messages beforehand, so we need to generate them on the fly).

```python
assistant_system_prompt = """
You are a helpful assistant.
Answer the user's questions factually correctly.
"""

user_simulator_system_prompt = """
You are a student.
Ask questions about mathematical concepts.
"""

model_run = await generate_multi_turn_model_run(
    dataset=dataset,
    assistant_system_prompt=assistant_system_prompt,
    assistant_model_configuration=ModelConfiguration(model="gpt-4o"),
    user_simulator_system_prompt=user_simulator_system_prompt,
    user_simulator_model_configuration=ModelConfiguration(model="gpt-4o"),
)
```

This will generate a multi-turn model run with the assistant and user simulator models.
You can then evaluate the model run using a binary or likert scale judge.

You can pass the `n_turns` parameter to the `generate_multi_turn_model_run` function to control how many turns the assistant and user simulator will have (where 1 turn is a user message and an assistant response).

You can find a full example in [`examples/multi_turn.py`](examples/multi_turn.py).

### Serialization and Deserialization

You can serialize model runs and evaluation runs to JSON files using the `to_json_file` method:

```python
model_run.to_json_file("model_run.json")
evaluation_run.to_json_file("evaluation_run.json")
```

You can also deserialize model runs and evaluation runs from JSON files using the `from_json_file` classmethod:

```python
model_run_from_json = ModelRun.from_json_file("model_run.json")
evaluation_run_from_json = BinaryEvaluationRun.from_json_file("evaluation_run.json")
```

You can find a full example in [`examples/serialization.py`](examples/serialization.py).

### Structured Evaluation

You can use the `structured` parameter in the `generate_evaluation_run_binary` and `generate_evaluation_run_likert` functions to get a structured response from the judge.
Note that you should take care that the system prompt of the judge indicates what the structured response should look like.
When doing structured evaluations, you don't need to provide a `response_callback` to the judge.

Here is an example:

```python
judge_system_prompt = """
You are an LLM as a judge.
You will be given a conversation between a user and an assistant.
You will then judge whether the assistants answers are factually correct or not.
Return 'true' if the answers are factually correct, and 'false' if they are not.
You should also return a short one-sentence explanation for your judgement.
"""

binary_judge = BinaryJudge(
    model_configuration=ModelConfiguration(model="gpt-4o"),
    system_prompt=judge_system_prompt
)

evaluation_run = await generate_evaluation_run_binary(
    model_run=model_run,
    binary_judge=binary_judge,
    structured=True
)
```

Now, the `explanation` field will contain the explanation of the judge's response, for example:

```json
{
    "conversation": {
        "messages": [
            {
                "role": "system",
                "content": "\nYou are a helpful assistant.\nAnswer the user's questions factually correctly.\n"
            },
            {
                "role": "user",
                "content": "What is the capital of Germany?"
            },
            {
                "role": "assistant",
                "content": "The capital of Germany is Berlin."
            }
        ]
    },
    "judge_response": "The answer is correct as Berlin is indeed the capital of Germany.",
    "result": true,
    "explanation": "The answer is correct as Berlin is indeed the capital of Germany."
}
```

You can find a full example in [`examples/structured_evaluation.py`](examples/structured_evaluation.py).

### Likert Evaluation

Likert evaluations work the same as binary evaluations, except that you need to call the `*_likert` functions instead of the `*_binary` functions.

For example, here is how you can perform a structured Likert evaluation:

```python
import asyncio
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
"""

judge_system_prompt = """
You are an LLM as a judge.
You will be given a conversation between a user and an assistant.
You will then judge whether the assistants answers are factually correct or not.
Return a number between 1 and 4 where:
- 1 means that answer is strongly incorrect
- 2 means that answer is incorrect
- 3 means that answer is correct
- 4 means that answer is strongly correct
"""

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
    print(model_run.model_dump_json())

    evaluation_run = await generate_evaluation_run_likert(
        model_run=model_run, likert_judge=likert_judge, structured=True
    )

    print("Full evaluation run:")
    print(evaluation_run.model_dump_json())
    print("Average score:")
    print(evaluation_run.avg)


if __name__ == "__main__":
    asyncio.run(main())
```

This will be the full evaluation run:

```json
{
    "rows": [
        {
            "conversation": {
                "messages": [
                    {
                        "role": "system",
                        "content": "\nYou are a helpful assistant.\nAnswer the user's questions factually correctly.\n"
                    },
                    {
                        "role": "user",
                        "content": "What is the capital of France?"
                    },
                    {
                        "role": "assistant",
                        "content": "The capital of France is Paris."
                    }
                ]
            },
            "result": 4,
            "explanation": "The statement that the capital of France is Paris is factually correct. Paris has been the capital of France since the late 10th century and is widely recognized as such internationally.",
            "judge_response": "The statement that the capital of France is Paris is factually correct. Paris has been the capital of France since the late 10th century and is widely recognized as such internationally."
        },
        {
            "conversation": {
                "messages": [
                    {
                        "role": "system",
                        "content": "\nYou are a helpful assistant.\nAnswer the user's questions factually correctly.\n"
                    },
                    {
                        "role": "user",
                        "content": "What is the capital of Germany?"
                    },
                    {
                        "role": "assistant",
                        "content": "The capital of Germany is Berlin."
                    }
                ]
            },
            "result": 4,
            "explanation": "The answer correctly identifies Berlin as the capital of Germany, which is a well-established fact.",
            "judge_response": "The answer correctly identifies Berlin as the capital of Germany, which is a well-established fact."
        }
    ]
}
```

Pay attention to your judge system prompt when doing Likert evaluations.
It's also usually better to do structured Likert evaluations than non-structured ones.

You can find a full example in [`examples/likert.py`](examples/likert.py).
