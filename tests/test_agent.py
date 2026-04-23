from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi.testclient import TestClient

import app.agent as agent_module
import app.main as main_module
from app.agent import HRChatAgent
from app.main import app


def setup_module(module: object) -> None:
    agent_module.LLM_MODE = "auto"


def test_access_filter_hides_hr_only_docs_for_employee() -> None:
    agent = HRChatAgent()
    employee_result = agent.answer("e001", "Puis-je recevoir ma paie sur un autre compte ?")
    manager_result = agent.answer("m001", "Puis-je recevoir ma paie sur un autre compte ?")
    employee_source_ids = {source["doc_id"] for source in employee_result["sources"]}
    manager_source_ids = {source["doc_id"] for source in manager_result["sources"]}
    assert "paie_modification_coordonnees_bancaires" in employee_source_ids
    assert "paie_modification_coordonnees_bancaires" not in manager_source_ids


def test_hr_can_see_employee_hr_shared_payroll_procedure() -> None:
    agent = HRChatAgent()
    result = agent.answer("hr001", "Puis-je recevoir ma paie sur un autre compte ?")
    source_ids = {source["doc_id"] for source in result["sources"]}
    assert "paie_modification_coordonnees_bancaires" in source_ids


def test_user_context_is_injected_for_leave_balance() -> None:
    agent = HRChatAgent()
    result = agent.answer("e001", "Combien de jours de congés me reste-t-il ?")
    assert "12.5 jours" in result["answer"]


def test_standard_shortcuts_bypass_llm_and_use_standard_mode() -> None:
    agent = HRChatAgent()
    result = agent.answer("e001", "Quelle est la durée de ma période d'essai ?")
    assert "4 mois" in result["answer"]
    assert "2024-05-15" in result["answer"]
    assert result["workflow"][-1]["details"] == "standard"


def test_chat_history_is_persisted_as_json_lines() -> None:
    with TemporaryDirectory() as tmp_dir:
        history_path = Path(tmp_dir) / "chat_history.jsonl"
        agent = HRChatAgent(history_path=history_path)
        result = agent.answer("e001", "Quelle est la durée de ma période d'essai ?")

        lines = history_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["user_id"] == "e001"
        assert entry["question"] == "Quelle est la durée de ma période d'essai ?"
        assert entry["answer"] == result["answer"]
        assert entry["workflow"][-1]["details"] == "standard"


def test_history_endpoint_returns_recent_entries() -> None:
    with TemporaryDirectory() as tmp_dir:
        original_agent = main_module.agent
        history_path = Path(tmp_dir) / "chat_history.jsonl"
        main_module.agent = HRChatAgent(history_path=history_path)
        client = TestClient(app)
        try:
            client.post(
                "/chat",
                json={"user_id": "e001", "question": "Quelle est la durée de ma période d'essai ?"},
            )
            response = client.get("/history", params={"user_id": "e001", "limit": 10})
        finally:
            main_module.agent = original_agent

        assert response.status_code == 200
        body = response.json()
        assert len(body) == 1
        assert body[0]["user_id"] == "e001"
        assert body[0]["question"] == "Quelle est la durée de ma période d'essai ?"


def test_api_chat_endpoint() -> None:
    client = TestClient(app)
    response = client.post(
        "/chat",
        json={"user_id": "e001", "question": "Où puis-je télécharger mon contrat de travail ?"},
    )
    assert response.status_code == 200
    body = response.json()
    assert "rh_procedure_acces_documents_contractuels" in {item["doc_id"] for item in body["sources"]}
    first_source = body["sources"][0]
    assert "relevant_sections" in first_source
    assert isinstance(first_source["relevant_sections"], list)


def test_document_endpoint_respects_access() -> None:
    client = TestClient(app)
    ok_response = client.get(
        "/documents/paie_modification_coordonnees_bancaires",
        params={"user_id": "e001"},
    )
    forbidden_response = client.get(
        "/documents/paie_modification_coordonnees_bancaires",
        params={"user_id": "m001"},
    )
    assert ok_response.status_code == 200
    assert ok_response.json()["doc_id"] == "paie_modification_coordonnees_bancaires"
    assert forbidden_response.status_code == 403


def test_scenarios_from_dataset() -> None:
    agent = HRChatAgent()
    scenarios = json.loads(Path("tests/scenarios.json").read_text(encoding="utf-8"))
    for scenario in scenarios:
        result = agent.answer(scenario["user_id"], scenario["question"])
        answer = str(result["answer"])
        source_ids = {source["doc_id"] for source in result["sources"]}
        assert answer.strip()
        assert answer.lower() != "none"
        for expected_source in scenario["must_source"]:
            assert expected_source in source_ids
