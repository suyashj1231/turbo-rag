"""HTTP API. Local stand-in for API Gateway + Lambda.

    uv run uvicorn turbo_rag.api:app --reload
    curl -X POST localhost:8000/chat -H 'content-type: application/json' \\
         -d '{"message": "a novel about a preacher in Iowa"}'

Mapping to the serverless version:
- `lifespan` warmup  == Lambda cold start (load model + indexes once per container)
- `chat()` handler   == the Lambda handler (wrap `app` with Mangum to deploy)
- SessionStore       == DynamoDB table
Handlers are plain `def` (not async): retrieval and the LLM call are blocking,
and FastAPI runs sync handlers on a threadpool so they don't block the loop.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import CONFIG
from .pipeline import ChatService, build_retriever
from .sessions import SessionStore

store = SessionStore(CONFIG.sessions_db)
service = ChatService(store)


@asynccontextmanager
async def lifespan(_: FastAPI):
    build_retriever(use_reranker=CONFIG.retrieval_mode == "hybrid_rerank")  # warm start
    yield


app = FastAPI(title="turbo-rag", lifespan=lifespan)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    session_id: str | None = Field(default=None, description="omit to start a new session")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "retrieval_mode": CONFIG.retrieval_mode}


@app.post("/chat")
def chat(req: ChatRequest) -> dict:
    session_id = req.session_id or store.new_session_id()
    return service.chat(session_id, req.message)


@app.get("/sessions/{session_id}")
def get_session(session_id: str) -> dict:
    if not store.exists(session_id):
        raise HTTPException(status_code=404, detail="session not found")
    return {"session_id": session_id, "messages": store.history(session_id)}


@app.delete("/sessions/{session_id}", status_code=204)
def delete_session(session_id: str) -> None:
    store.delete(session_id)
