from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from app.config import DATA_DIR, KNOWLEDGE_DIR, LLM_MODE, LLM_PROVIDER, TOP_K
from app.knowledge import DocumentChunk, KnowledgeBase, UserProfile, load_users, tokenize
from app.llm import call_llm


@dataclass
class Intent:
    topic: str
    label: str


def classify_intent(question: str) -> Intent:
    lower = question.lower()
    rules = [
        ("conges", "Congés & absences", ["congé", "conges", "rtt", "arrêt", "maladie", "absence", "jour férié", "sans solde"]),
        ("paie", "Paie & rémunération", ["paie", "salaire", "bulletin", "prime", "compte bancaire", "rémunération"]),
        ("organisation", "Organisation", ["n+1", "responsable", "organigramme", "hiérarchique"]),
        ("contrat", "Contrat & administration", ["contrat", "période d’essai", "période d'essai", "attestation", "adresse postale", "temps partiel", "ancienneté"]),
    ]
    for topic, label, keywords in rules:
        if any(keyword in lower for keyword in keywords):
            return Intent(topic=topic, label=label)
    return Intent(topic="general", label="Général")


def deduplicate_sources(chunks: list[DocumentChunk]) -> list[dict[str, str]]:
    seen: set[str] = set()
    sources: list[dict[str, str]] = []
    for chunk in chunks:
        if chunk.doc_id in seen:
            continue
        seen.add(chunk.doc_id)
        sources.append({"doc_id": chunk.doc_id, "title": chunk.title, "section": chunk.heading})
    return sources


def build_profile_context(user: UserProfile, question: str) -> list[str]:
    lower = question.lower()
    facts = [
        f"Employé: {user.name}",
        f"Rôle: {user.role}",
        f"Département: {user.department}",
        f"Type de contrat: {user.contract_type}",
        f"Date d'embauche: {user.hire_date}",
    ]

    if "contrat" in lower and ("télécharger" in lower or "telecharger" in lower):
        facts.append(f"Contrat téléchargeable via: {user.contract_download_url}")
    if "période d'essai" in lower or "periode d'essai" in lower or "periode d essai" in lower:
        facts.append(
            f"Période d'essai de {user.trial_period_months} mois, fin prévue le {user.trial_period_end}."
        )
    if "n+1" in lower or "responsable" in lower or "hiérarchique" in lower:
        facts.append(f"Manager direct: {user.manager}. Responsable N+1: {user.n_plus_1}.")
    if "congé" in lower or "conges" in lower or "rtt" in lower:
        facts.append(f"Solde actuel de congés: {user.leave_balance_days} jours.")
    if "bulletin" in lower or "paie" in lower:
        facts.append(f"Portail bulletin de paie: {user.payslip_portal_url}")
    if "compte" in lower and "paie" in lower:
        facts.append(f"Compte bancaire actuellement enregistré: ****{user.bank_account_last4}.")
    if "échéance" in lower or "echeance" in lower or "fin de contrat" in lower:
        if user.contract_end_date:
            facts.append(f"Date de fin de contrat actuelle: {user.contract_end_date}.")
        else:
            facts.append("Aucune date de fin de contrat n'est enregistrée pour ce profil.")
    return facts


def build_prompt(
    question: str,
    user: UserProfile,
    intent: Intent,
    profile_facts: list[str],
    chunks: list[DocumentChunk],
) -> str:
    sources = "\n".join(f"- [{chunk.doc_id}] {chunk.content}" for chunk in chunks)
    user_context = "\n".join(f"- {fact}" for fact in profile_facts)
    return f"""
Tu es un agent RH interne francophone. Réponds de manière concise, pratique et fiable.

Règles:
- N'invente rien hors des sources fournies.
- Si une information dépend du profil employé, utilise le contexte utilisateur.
- Si la procédure comporte plusieurs étapes, liste-les clairement.
- Cite les identifiants de sources à la fin sous la forme: Sources: [id1], [id2]

Catégorie détectée: {intent.label}
Question: {question}

Contexte utilisateur:
{user_context}

Sources récupérées:
{sources}

Réponds en français.
""".strip()


def demo_answer(question: str, profile_facts: list[str], chunks: list[DocumentChunk]) -> str:
    query_terms = set(tokenize(question))
    lines: list[str] = []
    relevant_profile_facts = [fact for fact in profile_facts if not fact.startswith("Rôle:") and not fact.startswith("Département:")]
    if relevant_profile_facts:
        lines.append("Contexte employé:")
        lines.extend(f"- {fact}" for fact in relevant_profile_facts[:4])

    if chunks:
        lines.append("Réponse RH:")
        for chunk in chunks[:3]:
            cleaned = chunk.content
            prefix = f"{chunk.heading}. "
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix) :]
            lines.append(f"- {cleaned.rstrip('.')}.")
    else:
        lines.append("Je n'ai pas trouvé de règle RH autorisée pour cette question.")

    sources = ", ".join(f"[{source['doc_id']}]" for source in deduplicate_sources(chunks))
    if sources:
        lines.append(f"Sources: {sources}")
    else:
        lines.append("Sources: aucune")
    return "\n".join(lines)


class HRChatAgent:
    def __init__(self, users_path: Path | None = None, knowledge_dir: Path | None = None):
        self.users = load_users(users_path or DATA_DIR / "users.json")
        self.knowledge = KnowledgeBase(knowledge_dir or KNOWLEDGE_DIR)

    def answer(self, user_id: str, question: str) -> dict:
        user = self.users.get(user_id)
        if not user:
            raise ValueError(f"Utilisateur inconnu: {user_id}")

        intent = classify_intent(question)
        profile_facts = build_profile_context(user, question)
        chunks = self.knowledge.retrieve(query=question, user_role=user.role, topic=intent.topic, limit=TOP_K)

        prompt = build_prompt(question, user, intent, profile_facts, chunks)
        if LLM_MODE == "llm_only":
            answer_text, model_used = call_llm(prompt)
            answer_mode = f"{LLM_PROVIDER}:{model_used}"
        else:
            try:
                answer_text, model_used = call_llm(prompt)
                answer_mode = f"{LLM_PROVIDER}:{model_used}"
            except RuntimeError:
                answer_mode = "demo"
                answer_text = demo_answer(question, profile_facts, chunks)

        workflow = [
            {"step": "intent", "details": intent.label},
            {"step": "access", "details": f"role={user.role}"},
            {"step": "retrieval", "details": f"{len(chunks)} passages autorisés trouvés"},
            {"step": "response", "details": answer_mode},
        ]

        return {
            "user": asdict(user),
            "intent": asdict(intent),
            "answer": answer_text,
            "sources": deduplicate_sources(chunks),
            "workflow": workflow,
        }
