from __future__ import annotations

import requests

from app.config import (
    GEMINI_MODEL,
    GOOGLE_API_KEY,
    LLM_PROVIDER,
    LLM_TIMEOUT_SECONDS,
    OPENROUTER_API_KEY,
    OPENROUTER_APP_NAME,
    OPENROUTER_APP_URL,
    OPENROUTER_MODEL,
)


def call_gemini(prompt: str) -> tuple[str, str]:
    if not GOOGLE_API_KEY:
        raise RuntimeError("GOOGLE_API_KEY is missing. Add a valid Gemini API key to your environment.")

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GOOGLE_API_KEY}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.2,
            "topP": 0.9,
            "maxOutputTokens": 700,
        },
    }

    try:
        response = requests.post(url, json=payload, timeout=LLM_TIMEOUT_SECONDS)
        response.raise_for_status()
        data = response.json()
        candidates = data.get("candidates", [])
        if not candidates:
            raise RuntimeError("Gemini returned no candidates.")
        parts = candidates[0].get("content", {}).get("parts", [])
        text = "\n".join(part.get("text", "") for part in parts if part.get("text"))
        if not text.strip():
            raise RuntimeError("Gemini returned an empty response.")
        return text.strip(), GEMINI_MODEL
    except requests.HTTPError as exc:
        detail = exc.response.text[:500] if exc.response is not None else str(exc)
        raise RuntimeError(f"Gemini API HTTP error: {detail}") from exc
    except requests.RequestException as exc:
        raise RuntimeError(f"Gemini API request failed: {exc}") from exc


def call_openrouter(prompt: str) -> tuple[str, str]:
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is missing. Add a valid OpenRouter API key to your environment.")

    url = "https://openrouter.ai/api/v1/chat/completions"
    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a French internal HR assistant. "
                    "Answer directly from the provided evidence, synthesize steps clearly, "
                    "and do not tell the user to read files or documents. "
                    "Only cite source IDs at the end."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "temperature": 0.2,
        "max_tokens": 700,
    }
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": OPENROUTER_APP_URL,
        "X-Title": OPENROUTER_APP_NAME,
    }

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=LLM_TIMEOUT_SECONDS)
        response.raise_for_status()
        data = response.json()
        choices = data.get("choices", [])
        if not choices:
            raise RuntimeError("OpenRouter returned no choices.")
        message = choices[0].get("message", {})
        text = message.get("content", "")
        if isinstance(text, list):
            text = "\n".join(part.get("text", "") for part in text if isinstance(part, dict))
        text = str(text).strip()
        if not text:
            raise RuntimeError("OpenRouter returned an empty response.")
        model_used = str(data.get("model", OPENROUTER_MODEL))
        return text, model_used
    except requests.HTTPError as exc:
        detail = exc.response.text[:500] if exc.response is not None else str(exc)
        raise RuntimeError(f"OpenRouter API HTTP error: {detail}") from exc
    except requests.RequestException as exc:
        raise RuntimeError(f"OpenRouter API request failed: {exc}") from exc


def call_llm(prompt: str) -> tuple[str, str]:
    if LLM_PROVIDER == "openrouter":
        return call_openrouter(prompt)
    if LLM_PROVIDER == "gemini":
        return call_gemini(prompt)
    raise RuntimeError(f"Unsupported LLM_PROVIDER: {LLM_PROVIDER}")
