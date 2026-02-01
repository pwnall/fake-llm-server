import os
import sys
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any, Union
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from llama_cpp import Llama

# Global LLM instance
llm: Optional[Llama] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global llm
    model_path = os.environ.get("FAKE_LLM_MODEL_PATH")
    if not model_path:
        print("Error: FAKE_LLM_MODEL_PATH environment variable not set.", file=sys.stderr)
        sys.exit(1)
    
    print(f"Loading model from {model_path}...", file=sys.stderr)
    try:
        # n_ctx=0 means load from model, but explicitly setting a reasonable limit helps memory.
        # 2048 is standard for small models.
        llm = Llama(
            model_path=model_path,
            n_ctx=2048,
            verbose=False
        )
    except Exception as e:
        print(f"Failed to load model: {e}", file=sys.stderr)
        sys.exit(1)
    yield
    if llm:
        del llm

app = FastAPI(lifespan=lifespan)

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    stream: bool = False
    max_tokens: Optional[int] = None
    temperature: float = 0.8
    top_p: float = 0.95
    # Allow extra fields without failure
    class Config:
        extra = "ignore"

# Note: defined as synchronous 'def' to run in threadpool and not block event loop
@app.post("/v1/chat/completions")
def create_chat_completion(request: ChatCompletionRequest):
    if not llm:
        raise HTTPException(status_code=503, detail="Model not loaded")

    messages = [msg.model_dump() for msg in request.messages]

    try:
        response = llm.create_chat_completion(
            messages=messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stream=request.stream,
        )
        return response
    except Exception as e:
        print(f"Error during inference: {e}", file=sys.stderr)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/models")
def list_models():
    model_id = os.environ.get("FAKE_LLM_MODEL_ID", "default-model")
    return {
        "object": "list",
        "data": [
            {
                "id": model_id,
                "object": "model",
                "created": 1677610602,
                "owned_by": "fake-llm-server",
            }
        ]
    }
