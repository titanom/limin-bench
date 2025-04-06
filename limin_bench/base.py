import json
from typing import Callable
from limin import (
    Conversation,
    Message,
    ModelConfiguration,
)
from pydantic import BaseModel


class Dataset(BaseModel):
    rows: list[str]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index: int) -> str:
        return self.rows[index]

    def __iter__(self):
        return iter(self.rows)


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


class BinaryEvaluationRunRow(BaseModel):
    conversation: Conversation
    judge_response: str
    result: bool
    explanation: str | None = None


class BinaryEvaluationRun(BaseModel):
    rows: list[BinaryEvaluationRunRow]

    @property
    def n_correct(self) -> int:
        return len([row for row in self.rows if row.result])

    @property
    def n_incorrect(self) -> int:
        return len(self) - self.n_correct

    @property
    def accuracy(self) -> float:
        return self.n_correct / len(self)

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
