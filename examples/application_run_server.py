from fastapi import FastAPI
from pydantic import BaseModel
from limin import generate_text_completion
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()


class ExplainRequest(BaseModel):
    word: str


class ExplainResponse(BaseModel):
    explanation: str


@app.post("/explain", response_model=ExplainResponse)
async def explain_word(request: ExplainRequest):
    prompt = f"Please explain the word '{request.word}' in a clear and concise way."
    completion = await generate_text_completion(prompt)
    return ExplainResponse(explanation=completion.content)


# curl -X POST "http://localhost:8000/explain" -H "Content-Type: application/json" -d '{"word": "serendipity"}'

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
