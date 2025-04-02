from limin import Conversation, Message
from limin_bench import BinaryJudge


example_conversation_wrong = Conversation([
    Message(role="user", content="Hello, how are you?"),
    Message(role="assistant", content="I'm good, tank you!"),
])

example_conversation_correct = Conversation([
    Message(role="user", content="Hello, how are you?"),
    Message(role="assistant", content="I'm good, thank you!"),
])

system_prompt = """
You are a binary judge that determines if the assistant responses are grammatically correct.
Respond with "yes" if the response is grammatically correct, and "no" if it is not.
"""

judge = BinaryJudge(model="gpt-4o", system_prompt=system_prompt, callback=lambda x: x == "yes")

async def main():
    response = await judge.evaluate(example_conversation_wrong)
    print(response)
    response = await judge.evaluate(example_conversation_correct)
    print(response)

if __name__ == "__main__":
    import asyncio
    import dotenv
    dotenv.load_dotenv()
    asyncio.run(main())
