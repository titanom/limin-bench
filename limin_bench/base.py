from abc import ABC, abstractmethod
import json
from typing import Callable, Generic, Iterable, Literal, Type, TypeVar
from limin import (
    Conversation,
    Message,
    ModelConfiguration,
)
from pydantic import BaseModel

T = TypeVar("T")
D = TypeVar("D", bound="BaseDataset")


class BaseDataset(BaseModel, ABC, Generic[T]):
    def to_json_file(self, file_path: str, indent: int = 4) -> None:
        with open(file_path, "w") as f:
            json.dump(self.model_dump(), f, indent=indent)

    @classmethod
    def from_json_file(cls: Type[D], file_path: str) -> D:
        with open(file_path, "r") as f:
            return cls.model_validate(json.load(f))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> T:
        return self.rows[index]

    def __iter__(self) -> Iterable[T]:
        return iter(self.rows)


class PregeneratedMultiTurnDataset(BaseDataset[list[str]]):
    """
    A pregenerated multi-turn dataset is a list of rows, where every row is a list of strings indicating the pregenerated user messages to use during the multi-turn evaluation.
    """

    rows: list[list[str]]


class Dataset(BaseDataset[str]):
    """
    A dataset is a list of rows, where every row is a single string indicating the user message.

    Note that you can still perform a multi-turn evaluation on a Dataset by providing a "user simulator" model configuration and system prompt which is responsible for generating additional user messages (after the initial user message).
    If you want to perform multi-turn evaluation with pregenerated user messages, you need to use the PregeneratedMultiTurnDataset class.
    """

    rows: list[str]


class ModelRunRow(BaseModel):
    content: Conversation


class ModelRun(BaseModel):
    rows: list[ModelRunRow]

    def to_json_file(self, file_path: str, indent: int = 4) -> None:
        with open(file_path, "w") as f:
            json.dump(self.model_dump(), f, indent=indent)

    @classmethod
    def from_json_file(cls, file_path: str) -> "ModelRun":
        with open(file_path, "r") as f:
            return cls.model_validate(json.load(f))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index: int) -> ModelRunRow:
        return self.rows[index]

    def __iter__(self):
        return iter(self.rows)


class BinaryEvaluationRunRowResult(BaseModel):
    judge_response: str
    value: bool
    explanation: str | None = None

class BinaryEvaluationRunRow(BaseModel):
    """
    A BinaryEvaluationRunRow represents an evaluation run over a single row of a model run.

    The conversation is the conversation from the model run.
    
    The judge_responses, results, and explanations represent the results of the evaluation run.
    Note that they are all lists in order to support "stability" runs, i.e. runs where we let the judge model evaluate the same conversation multiple times in order to check the stability of the evaluations.
    """
    conversation: Conversation

    results: list[BinaryEvaluationRunRowResult]

    @property
    def result(self, method: Literal["mean", "min", "max"] = "mean") -> int:
        """
        The result of the evaluation run.

        The method argument can be one of:
        - "mean": The mean of the results (where 0 = False and 1 = True).
        - "min": The minimum of the results (either 0 if there is a False result or 1 if all results are True).
        - "max": The maximum of the results (either 1 if there is a True result or 0 if all results are False).
        """
        values = [result.value for result in self.results]

        if method == "mean":
            return sum(values) / len(values)
        elif method == "min":
            return 0 if False in values else 1
        elif method == "max":
            return 1 if True in values else 0


class BinaryEvaluationRun(BaseModel):
    rows: list[BinaryEvaluationRunRow]

    @property
    def n_correct(self) -> int:
        return len([row for row in self.rows if row.result >= 0.5])

    @property
    def n_incorrect(self) -> int:
        return len(self) - self.n_correct

    @property
    def accuracy(self) -> float:
        return sum(row.result for row in self.rows) / len(self)

    def to_json_file(self, file_path: str, indent: int = 4) -> None:
        with open(file_path, "w") as f:
            json.dump(self.model_dump(), f, indent=indent)

    @classmethod
    def from_json_file(cls, file_path: str) -> "BinaryEvaluationRun":
        with open(file_path, "r") as f:
            return cls.model_validate(json.load(f))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index: int) -> BinaryEvaluationRunRow:
        return self.rows[index]

    def __iter__(self):
        return iter(self.rows)


class LikertEvaluationRunRow(BaseModel):
    conversation: Conversation
    result: int
    explanation: str | None = None
    judge_response: str | None = None


class LikertEvaluationRun(BaseModel):
    rows: list[LikertEvaluationRunRow]

    @property
    def avg(self) -> float:
        return sum(row.result for row in self.rows) / len(self)

    def to_json_file(self, file_path: str, indent: int = 4) -> None:
        with open(file_path, "w") as f:
            json.dump(self.model_dump(), f, indent=indent)

    @classmethod
    def from_json_file(cls, file_path: str) -> "LikertEvaluationRun":
        with open(file_path, "r") as f:
            return cls.model_validate(json.load(f))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index: int) -> LikertEvaluationRunRow:
        return self.rows[index]

    def __iter__(self):
        return iter(self.rows)


# Note that explanation should be serialized before result (so that the explanation comes first and the result second).
class ExplainedBinaryJudgement(BaseModel):
    explanation: str
    result: bool


class BinaryJudge(BaseModel):
    model_configuration: ModelConfiguration
    system_prompt: str
    response_callback: Callable[[str], bool] | None = None


# Note that explanation should be serialized before result (so that the explanation comes first and the result second).
class ExplainedLikertJudgement(BaseModel):
    explanation: str
    result: int


class LikertJudge(BaseModel):
    model_configuration: ModelConfiguration
    system_prompt: str
    callback: Callable[[str], int] | None = None


def get_conversation_from_prompts(
    system_prompt: str | None, user_prompt: str, assistant_prompt: str
) -> Conversation:
    messages = []
    if system_prompt:
        messages.append(Message(role="system", content=system_prompt))
    messages.append(Message(role="user", content=user_prompt))
    messages.append(Message(role="assistant", content=assistant_prompt))
    return Conversation(messages=messages)
