import json
from typing import Literal

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from providers import SYSTEM_PROMPT, get_provider

app = FastAPI(title="AutoExpert Chat API")


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[Message] = []


@app.get("/health")
async def health():
    return {"status": "ok", "model": get_provider().model}


@app.post("/chat")
async def chat(req: ChatRequest, model: str | None = None):
    # Optional: POST /chat?model=qwen2.5:3b to try a different local model
    llm = get_provider(model)

    messages = [m.model_dump() for m in req.history]
    messages.append({"role": "user", "content": req.message})

    async def event_stream():
        try:
            async for piece in llm.stream(SYSTEM_PROMPT, messages):
                if isinstance(piece, str):  # skip the final Usage object
                    yield f"data: {json.dumps(piece)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"event: error\ndata: {json.dumps(str(e))}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
