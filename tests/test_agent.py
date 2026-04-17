from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from app.agent import HRChatAgent
from app.main import app


def test_access_filter_hides_hr_only_docs_for_employee() -> None:
    agent = HRChatAgent()
    result = agent.answer("e001", "Mon bulletin de paie comporte une erreur, comment corriger ?")
    source_ids = {source["doc_id"] for source in result["sources"]}
    assert "payroll_employee" in source_ids
    assert "hr_internal_payroll" not in source_ids


def test_hr_can_see_internal_payroll_procedure() -> None:
    agent = HRChatAgent()
    result = agent.answer("hr001", "Mon bulletin de paie comporte une erreur, comment corriger ?")
    source_ids = {source["doc_id"] for source in result["sources"]}
    assert "hr_internal_payroll" in source_ids


def test_user_context_is_injected_for_leave_balance() -> None:
    agent = HRChatAgent()
    result = agent.answer("e001", "Combien de jours de congés me reste-t-il ?")
    assert "12.5 jours" in result["answer"]


def test_api_chat_endpoint() -> None:
    client = TestClient(app)
    response = client.post(
        "/chat",
        json={"user_id": "e001", "question": "Où puis-je télécharger mon contrat de travail ?"},
    )
    assert response.status_code == 200
    body = response.json()
    assert "contract_admin_employee" in {item["doc_id"] for item in body["sources"]}


def test_scenarios_from_dataset() -> None:
    agent = HRChatAgent()
    scenarios = json.loads(Path("tests/scenarios.json").read_text(encoding="utf-8"))
    for scenario in scenarios:
        result = agent.answer(scenario["user_id"], scenario["question"])
        answer = result["answer"].lower()
        source_ids = {source["doc_id"] for source in result["sources"]}
        for expected in scenario["must_contain"]:
            assert expected.lower() in answer
        for expected_source in scenario["must_source"]:
            assert expected_source in source_ids
