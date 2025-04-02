from typing import Callable
from limin import Conversation, generate_text_completion

# add to limin

def to_markdown(conversation: Conversation) -> str:
    return "\n".join([f"{message.role}: {message.content}" for message in conversation.messages])

# limin-bench

class Dataset:
    def __init__(self, conversations: list[Conversation]):
        self.conversations = conversations

    def __len__(self):
        return len(self.conversations)

    def __getitem__(self, index: int) -> Conversation:
        return self.conversations[index]

    def __iter__(self):
        return iter(self.conversations)


class BinaryJudge:
    def __init__(self, model: str, system_prompt: str, callback: Callable[[str], bool]):
        """
        :param callback: A function that takes the judge response string and returns whether that response indicates a positive or negative judgement.
        """
        self.model = model
        self.system_prompt = system_prompt
        self.callback = callback

    async def evaluate(self, conversation: Conversation) -> bool:
        response = await generate_text_completion(to_markdown(conversation), model=self.model, system_prompt=self.system_prompt)
        return self.callback(response.message)

