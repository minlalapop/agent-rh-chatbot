from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

from app.agent import HRChatAgent
from app.config import STATIC_DIR


class ChatRequest(BaseModel):
    user_id: str
    question: str


app = FastAPI(title="Agent RH interne", version="1.0.0")
agent = HRChatAgent()


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse((STATIC_DIR / "index.html").read_text(encoding="utf-8"))


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> FileResponse:
    raise HTTPException(status_code=404, detail="No favicon")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/users")
def list_users() -> list[dict[str, str]]:
    return [
        {"user_id": user.user_id, "name": user.name, "role": user.role}
        for user in agent.users.values()
    ]


@app.post("/chat")
def chat(payload: ChatRequest) -> dict:
    try:
        return agent.answer(user_id=payload.user_id, question=payload.question)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
