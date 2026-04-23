from __future__ import annotations

import html
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
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


@app.get("/documents/{doc_id}")
def get_document(doc_id: str, user_id: str) -> dict:
    try:
        return agent.get_document_view(user_id=user_id, doc_id=doc_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc


@app.post("/chat")
def chat(payload: ChatRequest) -> dict:
    try:
        return agent.answer(user_id=payload.user_id, question=payload.question)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.get("/history", response_model=None)
def history(request: Request, user_id: str | None = None, limit: int = 50, format: str = "auto"):
    if limit < 1 or limit > 500:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 500")
    entries = agent.list_history(user_id=user_id, limit=limit)

    wants_json = format == "json" or "application/json" in request.headers.get("accept", "")
    if wants_json:
        return JSONResponse(content=entries)

    rows = []
    for entry in reversed(entries):
        sources = ", ".join(source.get("doc_id", "") for source in entry.get("sources", [])) or "aucune"
        workflow = " -> ".join(
            f"{step.get('step', '')}: {step.get('details', '')}"
            for step in entry.get("workflow", [])
        ) or "n/a"
        rows.append(
            f"""
            <tr>
              <td>{html.escape(str(entry.get("timestamp", "")))}</td>
              <td>{html.escape(str(entry.get("user_id", "")))}</td>
              <td>{html.escape(str(entry.get("user_role", "")))}</td>
              <td>{html.escape(str(entry.get("question", "")))}</td>
              <td>{html.escape(str(entry.get("answer", "")))}</td>
              <td>{html.escape(sources)}</td>
              <td>{html.escape(workflow)}</td>
            </tr>
            """.strip()
        )

    filters = []
    if user_id:
        filters.append(f"Utilisateur: {user_id}")
    filters.append(f"Limite: {limit}")
    filters_text = html.escape(" | ".join(filters))

    page_html = f"""
    <!DOCTYPE html>
    <html lang="fr">
      <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <title>Historique des échanges</title>
        <style>
          :root {{
            color-scheme: light;
            --bg: #f6f7fb;
            --card: #ffffff;
            --line: #d9deea;
            --text: #172033;
            --muted: #5f6b85;
            --accent: #175cd3;
          }}

          * {{
            box-sizing: border-box;
          }}

          body {{
            margin: 0;
            font-family: "Avenir Next", "Segoe UI", sans-serif;
            background: linear-gradient(180deg, #eef3ff 0%, var(--bg) 100%);
            color: var(--text);
          }}

          .page {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 32px 20px 48px;
          }}

          .header {{
            margin-bottom: 18px;
          }}

          h1 {{
            margin: 0 0 8px;
            font-size: 28px;
            line-height: 1.1;
          }}

          .meta {{
            color: var(--muted);
            font-size: 14px;
          }}

          .card {{
            background: var(--card);
            border: 1px solid var(--line);
            border-radius: 18px;
            overflow: hidden;
            box-shadow: 0 18px 50px rgba(23, 32, 51, 0.08);
          }}

          .table-wrap {{
            overflow: auto;
          }}

          table {{
            width: 100%;
            border-collapse: collapse;
            min-width: 1100px;
          }}

          thead {{
            background: #eef4ff;
          }}

          th, td {{
            padding: 14px 16px;
            text-align: left;
            vertical-align: top;
            border-bottom: 1px solid var(--line);
            font-size: 14px;
            line-height: 1.45;
          }}

          th {{
            color: var(--accent);
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.04em;
          }}

          td {{
            white-space: pre-wrap;
          }}

          tbody tr:hover {{
            background: #f9fbff;
          }}

          .empty {{
            padding: 28px;
            color: var(--muted);
          }}
        </style>
      </head>
      <body>
        <main class="page">
          <div class="header">
            <h1>Historique des échanges</h1>
            <div class="meta">{filters_text}</div>
          </div>
          <section class="card">
            <div class="table-wrap">
              {
                "<table><thead><tr><th>Date</th><th>Utilisateur</th><th>Rôle</th><th>Question</th><th>Réponse</th><th>Sources</th><th>Workflow</th></tr></thead><tbody>"
                + "".join(rows)
                + "</tbody></table>"
                if rows
                else '<div class="empty">Aucun historique disponible pour ces filtres.</div>'
              }
            </div>
          </section>
        </main>
      </body>
    </html>
    """
    return HTMLResponse(content=page_html)
