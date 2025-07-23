from abc import ABC
import json
import statistics
from typing import Callable, Generic, Literal, Type, TypeVar
from limin import (
    Conversation,
    Message,
    ModelConfiguration,
)
from pydantic import BaseModel

T = TypeVar("T")
D = TypeVar("D", bound="BaseDataset")


class BaseDataset(BaseModel, ABC, Generic[T]):
    rows: list[T]

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

    def __iter__(self):
        return iter(self.rows)


def dict_to_markdown_table(
    data: dict[str, list[str]], max_column_length: int = 50
) -> str:
    """
    Convert a dictionary of column names to column values into a markdown table.

    Args:
        data: Dictionary where keys are column names and values are lists of cell values
        max_column_length: Maximum length of cell content before truncation

    Returns:
        A markdown table as a string
    """
    if not data or not any(data.values()):
        headers = list(data.keys()) if data else []
        header_row = "| " + " | ".join(headers) + " |"
        separator = "|" + "|".join(["-" * (len(h) + 2) for h in headers]) + "|"
        return header_row + "\n" + separator

    columns = list(data.keys())

    processed_data = {}
    for col_name, col_values in data.items():
        processed_values = []
        for value in col_values:
            # Escape pipe characters and replace newlines to avoid breaking the table
            escaped_value = str(value).replace("|", "\\|").replace("\n", " ")

            if len(escaped_value) > max_column_length:
                escaped_value = escaped_value[: max_column_length - 3] + "..."
            processed_values.append(escaped_value)
        processed_data[col_name] = processed_values

    col_widths = {}
    for col_name in columns:
        col_widths[col_name] = max(
            len(col_name),
            max(len(value) for value in processed_data[col_name])
            if processed_data[col_name]
            else 0,
        )

    header = (
        "| "
        + " | ".join(col_name.ljust(col_widths[col_name]) for col_name in columns)
        + " |"
    )
    separator = (
        "|" + "|".join("-" * (col_widths[col_name] + 2) for col_name in columns) + "|"
    )

    lines = [header, separator]

    num_rows = len(processed_data[columns[0]]) if columns else 0
    for row_idx in range(num_rows):
        row_values = [
            processed_data[col_name][row_idx].ljust(col_widths[col_name])
            for col_name in columns
        ]
        line = "| " + " | ".join(row_values) + " |"
        lines.append(line)

    return "\n".join(lines)


class PregeneratedMultiTurnDataset(BaseDataset[list[str]]):
    """
    A pregenerated multi-turn dataset is a list of rows, where every row is a list of strings indicating the pregenerated user messages to use during the multi-turn evaluation.
    """

    rows: list[list[str]]

    def to_markdown_table(self, max_column_length: int = 50) -> str:
        """
        Returns a markdown table representation of the pregenerated multi-turn dataset.
        Each row shows the turn number and the corresponding user message.
        """
        if not self.rows:
            return dict_to_markdown_table(
                {"Row": [], "Turn": [], "Message": []}, max_column_length
            )

        # Collect all data
        row_values = []
        turn_values = []
        message_values = []

        for row_idx, row in enumerate(self.rows):
            for turn_idx, message in enumerate(row):
                # Only show row number for the first turn of each row
                row_display = str(row_idx) if turn_idx == 0 else ""
                row_values.append(row_display)
                turn_values.append(str(turn_idx + 1))
                message_values.append(message)

        data = {"Row": row_values, "Turn": turn_values, "Message": message_values}

        return dict_to_markdown_table(data, max_column_length)


class Dataset(BaseDataset[str]):
    """
    A dataset is a list of rows, where every row is a single string indicating the user message.

    Note that you can still perform a multi-turn evaluation on a Dataset by providing a "user simulator" model configuration and system prompt which is responsible for generating additional user messages (after the initial user message).
    If you want to perform multi-turn evaluation with pregenerated user messages, you need to use the PregeneratedMultiTurnDataset class.
    """

    rows: list[str]

    def to_markdown_table(self, max_column_length: int = 50) -> str:
        """
        Returns a markdown table representation of the dataset.
        Each row shows the row number and the corresponding user message.
        """
        if not self.rows:
            return dict_to_markdown_table({"Row": [], "Message": []}, max_column_length)

        # Collect all data
        row_values = [str(row_idx) for row_idx in range(len(self.rows))]
        message_values = list(self.rows)

        data = {"Row": row_values, "Message": message_values}

        return dict_to_markdown_table(data, max_column_length)


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

    def to_markdown_table(self, max_column_length: int = 50) -> str:
        """
        Returns a markdown table representation of the model run.
        Each row shows the row number, turn, role, and message content.
        """
        if not self.rows:
            return dict_to_markdown_table(
                {"Row": [], "Turn": [], "Role": [], "Message": []}, max_column_length
            )

        # Collect all data
        row_values = []
        turn_values = []
        role_values = []
        message_values = []

        for row_idx, model_run_row in enumerate(self.rows):
            conversation = model_run_row.content
            current_turn = 0

            for message_idx, message in enumerate(conversation.messages):
                # Only show row number for the first message of each row
                row_display = str(row_idx) if message_idx == 0 else ""
                row_values.append(row_display)

                # Handle turn numbering
                if message.role == "system":
                    # System messages get turn number 0
                    turn_str = "0"
                elif message.role == "user":
                    # User messages start a new turn
                    current_turn += 1
                    turn_str = str(current_turn)
                else:
                    # Assistant messages continue the current turn (no turn number shown)
                    turn_str = ""

                turn_values.append(turn_str)
                role_values.append(message.role)
                message_values.append(message.content)

        data = {
            "Row": row_values,
            "Turn": turn_values,
            "Role": role_values,
            "Message": message_values,
        }

        return dict_to_markdown_table(data, max_column_length)  # type: ignore


class BinaryEvaluationRunRowResult(BaseModel):
    judge_response: str
    value: bool
    explanation: str | None = None


class BinaryEvaluationRunRow(BaseModel):
    """
    A BinaryEvaluationRunRow represents an evaluation run over a single row of a model run.

    The conversation is the conversation from the model run.

    The judge_responses, results, and explanations represent the results of the evaluation run.
    Note that they are all lists in order to support "stability" runs, i.e. runs where we let the judge model evaluate the same conversation multiple times in order to check the instability of the evaluations.
    """

    conversation: Conversation

    results: list[BinaryEvaluationRunRowResult]

    def value(self, method: Literal["mean", "min", "max"] = "mean") -> float:
        """
        The result of the evaluation run for the given row.

        The method argument can be one of:
        - "mean": The mean of the results (where 0 = False and 1 = True).
        - "min": The minimum of the results (either 0 if there is a False result or 1 if all results are True).
        - "max": The maximum of the results (either 1 if there is a True result or 0 if all results are False).
        """
        values = [int(result.value) for result in self.results]

        if method == "mean":
            return sum(values) / len(values)
        elif method == "min":
            return min(values)
        elif method == "max":
            return max(values)

    @property
    def instability(self) -> float:
        """
        The instability of the evaluation run for the given row.

        This is defined as the standard deviation of the results.
        """
        values = [int(result.value) for result in self.results]
        return statistics.stdev(values) if len(values) > 1 else 0.0


def _instability(
    instabilities: list[float], method: Literal["mean", "max", "fus"] = "mean"
) -> float:
    if method == "mean":
        return sum(instabilities) / len(instabilities)
    elif method == "max":
        return max(instabilities)
    elif method == "fus":
        return sum(1 for instability in instabilities if instability > 0.0) / len(
            instabilities
        )


class BinaryEvaluationRun(BaseModel):
    rows: list[BinaryEvaluationRunRow]

    @property
    def n_correct(self) -> int:
        return len([row for row in self.rows if row.value() >= 0.5])

    @property
    def n_incorrect(self) -> int:
        return len(self) - self.n_correct

    @property
    def accuracy(self) -> float:
        return sum(row.value() for row in self.rows) / len(self)

    def instability(self, method: Literal["mean", "max", "fus"] = "mean") -> float:
        """
        The instability of the evaluation run.

        The method argument can be one of:
        - "mean": The mean of the instability of the rows.
        - "max": The maximum of the instability of the rows.
        - "fus": The fraction of unstable rows.
        """
        return _instability([row.instability for row in self.rows], method)

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

    def to_markdown_table(self, max_column_length: int = 50) -> str:
        """
        Returns a markdown table representation of the Binary evaluation run.
        Each row shows the row number, turn, role, message content, evaluation explanation, evaluation value, and instability.
        The evaluation explanation, value, and instability are only shown after the last message of each row.
        """
        if not self.rows:
            return dict_to_markdown_table(
                {
                    "Row": [],
                    "Turn": [],
                    "Role": [],
                    "Message": [],
                    "Explanation": [],
                    "Value": [],
                    "Instability": [],
                },
                max_column_length,
            )

        row_values = []
        turn_values = []
        role_values = []
        message_values = []
        explanation_values = []
        value_values = []
        instability_values = []

        for row_idx, evaluation_run_row in enumerate(self.rows):
            conversation = evaluation_run_row.conversation
            current_turn = 0

            for message_idx, message in enumerate(conversation.messages):
                # Only show row number for the first message of each row
                row_display = str(row_idx) if message_idx == 0 else ""
                row_values.append(row_display)

                if message.role == "system":
                    # System messages get turn number 0
                    turn_str = "0"
                elif message.role == "user":
                    # User messages start a new turn
                    current_turn += 1
                    turn_str = str(current_turn)
                else:
                    # Assistant messages continue the current turn (no turn number shown)
                    turn_str = ""

                turn_values.append(turn_str)
                role_values.append(message.role)
                message_values.append(message.content)

                # Only show evaluation result after the last message of each row
                if message_idx == len(conversation.messages) - 1:
                    # We only show the first explanation
                    explanation_values.append(
                        evaluation_run_row.results[0].explanation or ""
                    )
                    value_values.append(str(evaluation_run_row.value()))
                    instability_values.append(f"{evaluation_run_row.instability:.3f}")
                else:
                    explanation_values.append("")
                    value_values.append("")
                    instability_values.append("")

        data = {
            "Row": row_values,
            "Turn": turn_values,
            "Role": role_values,
            "Message": message_values,
            "Explanation": explanation_values,
            "Value": value_values,
            "Instability": instability_values,
        }

        return dict_to_markdown_table(data, max_column_length)  # type: ignore


class LikertEvaluationRunRowResult(BaseModel):
    judge_response: str
    value: int
    explanation: str | None = None


class LikertEvaluationRunRow(BaseModel):
    """
    A LikertEvaluationRunRow represents an evaluation run over a single row of a model run.

    The conversation is the conversation from the model run.

    The results represent the results of the evaluation run.
    Note that they are a list in order to support "stability" runs, i.e. runs where we let the judge model evaluate the same conversation multiple times in order to check the stability of the evaluations.
    """

    conversation: Conversation

    results: list[LikertEvaluationRunRowResult]

    def value(self, method: Literal["mean", "min", "max"] = "mean") -> float:
        """
        The result of the evaluation run.

        The method argument can be one of:
        - "mean": The mean of the results.
        - "min": The minimum of the results.
        - "max": The maximum of the results.
        """
        values = [result.value for result in self.results]

        if method == "mean":
            return sum(values) / len(values)
        elif method == "min":
            return min(values)
        elif method == "max":
            return max(values)

    @property
    def instability(self) -> float:
        values = [result.value for result in self.results]
        return statistics.stdev(values) if len(values) > 1 else 0.0


class LikertEvaluationRun(BaseModel):
    rows: list[LikertEvaluationRunRow]

    @property
    def min(self) -> float:
        return min(row.value() for row in self.rows)

    @property
    def max(self) -> float:
        return max(row.value() for row in self.rows)

    @property
    def avg(self) -> float:
        return sum(row.value() for row in self.rows) / len(self)

    def instability(self, method: Literal["mean", "max", "fus"] = "mean") -> float:
        return _instability([row.instability for row in self.rows], method)

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

    def to_markdown_table(self, max_column_length: int = 50) -> str:
        """
        Returns a markdown table representation of the Likert evaluation run.
        Each row shows the row number, turn, role, message content, evaluation explanation, evaluation score, and stability.
        The evaluation explanation, score, and stability are only shown after the last message of each row.
        """
        if not self.rows:
            return dict_to_markdown_table(
                {
                    "Row": [],
                    "Turn": [],
                    "Role": [],
                    "Message": [],
                    "Explanation": [],
                    "Score": [],
                    "Instability": [],
                },
                max_column_length,
            )

        row_values = []
        turn_values = []
        role_values = []
        message_values = []
        explanation_values = []
        score_values = []
        instability_values = []

        for row_idx, evaluation_run_row in enumerate(self.rows):
            conversation = evaluation_run_row.conversation
            current_turn = 0

            for message_idx, message in enumerate(conversation.messages):
                # Only show row number for the first message of each row
                row_display = str(row_idx) if message_idx == 0 else ""
                row_values.append(row_display)

                if message.role == "system":
                    # System messages get turn number 0
                    turn_str = "0"
                elif message.role == "user":
                    # User messages start a new turn
                    current_turn += 1
                    turn_str = str(current_turn)
                else:
                    # Assistant messages continue the current turn (no turn number shown)
                    turn_str = ""

                turn_values.append(turn_str)
                role_values.append(message.role)
                message_values.append(message.content)

                # Only show evaluation result after the last message of each row
                if message_idx == len(conversation.messages) - 1:
                    # We only show the first explanation
                    explanation_values.append(
                        evaluation_run_row.results[0].explanation or ""
                    )
                    score_values.append(str(round(evaluation_run_row.value(), 2)))
                    instability_values.append(
                        str(round(evaluation_run_row.instability, 2))
                    )
                else:
                    explanation_values.append("")
                    score_values.append("")
                    instability_values.append("")

        data = {
            "Row": row_values,
            "Turn": turn_values,
            "Role": role_values,
            "Message": message_values,
            "Explanation": explanation_values,
            "Score": score_values,
            "Instability": instability_values,
        }

        return dict_to_markdown_table(data, max_column_length)  # type: ignore


# Note that explanation should be serialized before result (so that the explanation comes first and the result second).
class ExplainedBinaryJudgement(BaseModel):
    explanation: str
    value: bool


class BinaryJudge(BaseModel):
    model_configuration: ModelConfiguration
    system_prompt: str
    response_callback: Callable[[str], bool] | None = None


# Note that explanation should be serialized before result (so that the explanation comes first and the result second).
class ExplainedLikertJudgement(BaseModel):
    explanation: str
    value: int


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
