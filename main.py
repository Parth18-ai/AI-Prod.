import json
import os
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from ollama import AsyncClient
from pydantic import BaseModel

load_dotenv()

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MODEL_NAME = os.getenv("MODEL_NAME", "llama3.2")

# Paste your Week 3 persona here
SYSTEM_PROMPT = "You are AutoExpert, a friendly expert. Give clear, short answers."

app = FastAPI(title="AutoExpert Chat API")
client = AsyncClient(host=OLLAMA_HOST)


# ---------- Pydantic models (the order forms) ----------
class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[Message] = []   # earlier turns, sent by the client


# ---------- Routes (the menu) ----------
@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/chat")
async def chat(req: ChatRequest):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += [m.model_dump() for m in req.history]
    messages.append({"role": "user", "content": req.message})

    async def event_stream():
        try:
            stream = await client.chat(model=MODEL_NAME, messages=messages, stream=True)
            async for chunk in stream:
                token = chunk["message"]["content"]
                if token:
                    yield f"data: {json.dumps(token)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"event: error\ndata: {json.dumps(str(e))}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")