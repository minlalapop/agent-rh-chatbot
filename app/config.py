from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
KNOWLEDGE_DIR = BASE_DIR / "knowledge_base"
STATIC_DIR = BASE_DIR / "app" / "static"
CHAT_HISTORY_PATH = DATA_DIR / "chat_history.jsonl"

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openrouter").strip().lower()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openrouter/free").strip()
OPENROUTER_APP_NAME = os.getenv("OPENROUTER_APP_NAME", "agent-rh-interne").strip()
OPENROUTER_APP_URL = os.getenv("OPENROUTER_APP_URL", "http://localhost:8000").strip()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-lite").strip()
TOP_K = int(os.getenv("TOP_K", "5"))
LLM_TIMEOUT_SECONDS = int(os.getenv("LLM_TIMEOUT_SECONDS", "25"))
LLM_MODE = os.getenv("LLM_MODE", "auto").strip().lower()
