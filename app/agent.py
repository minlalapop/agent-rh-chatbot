from __future__ import annotations

import json
from datetime import datetime, timezone
from dataclasses import asdict, dataclass
from pathlib import Path

from app.config import CHAT_HISTORY_PATH, DATA_DIR, KNOWLEDGE_DIR, LLM_MODE, LLM_PROVIDER, TOP_K
from app.knowledge import DocumentChunk, KnowledgeBase, KnowledgeDocument, UserProfile, load_users, tokenize
from app.llm import call_llm


@dataclass
class Intent:
    topic: str
    label: str


STANDARD_QUESTION_MAP = {
    "quelle est la durée de ma période d'essai ?": "trial_period",
    "je souhaite passer à temps partiel, quelle est la procédure ?": "part_time",
    "comment calculer ma prime de performance ?": "performance_bonus",
    "comment déclarer un arrêt maladie ?": "sick_leave",
}


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


def deduplicate_sources(chunks: list[DocumentChunk]) -> list[dict[str, object]]:
    grouped: dict[str, dict[str, object]] = {}
    for chunk in chunks:
        source = grouped.setdefault(
            chunk.doc_id,
            {
                "doc_id": chunk.doc_id,
                "title": chunk.title,
                "section": chunk.heading,
                "relevant_sections": [],
            },
        )
        sections = source["relevant_sections"]
        if chunk.heading not in sections:
            sections.append(chunk.heading)
    return list(grouped.values())


def normalize_question(question: str) -> str:
    return " ".join(question.strip().lower().split())


def strip_heading_prefix(chunk: DocumentChunk) -> str:
    prefix = f"{chunk.heading}. "
    if chunk.content.startswith(prefix):
        return chunk.content[len(prefix) :].strip()
    return chunk.content.strip()


def get_source_ids(chunks: list[DocumentChunk]) -> str:
    source_ids = ", ".join(f"[{source['doc_id']}]" for source in deduplicate_sources(chunks))
    return source_ids or "aucune"


def standard_answer(question: str, user: UserProfile, chunks: list[DocumentChunk]) -> str | None:
    standard_type = STANDARD_QUESTION_MAP.get(normalize_question(question))
    if standard_type is None:
        return None

    sources = get_source_ids(chunks)

    if standard_type == "trial_period":
        return (
            f"Votre période d'essai est de {user.trial_period_months} mois, avec une fin prévue le {user.trial_period_end}. "
            "Elle est définie dans votre contrat de travail et peut, selon les règles applicables, être renouvelée sous conditions.\n\n"
            "Points d'attention :\n"
            f"- Vous êtes actuellement sur un contrat de type {user.contract_type}.\n"
            "- Pendant la période d'essai, un délai de prévenance doit être respecté en cas de rupture.\n"
            f"Sources: {sources}"
        )

    if standard_type == "part_time":
        return (
            "Vous pouvez demander un passage à temps partiel, mais la demande doit être formalisée et validée avant toute mise en place. "
            "Si elle est acceptée, un avenant à votre contrat sera rédigé pour fixer le nouveau temps de travail et les conditions associées.\n\n"
            "Étapes :\n"
            "- Adressez une demande écrite à votre manager et/ou aux RH en précisant le taux souhaité et la date de début.\n"
            "- Votre demande est analysée sur les plans organisationnel et contractuel.\n"
            "- Un entretien peut être organisé pour ajuster les modalités.\n"
            "- En cas d'accord, l'avenant est rédigé puis signé.\n\n"
            "Points d'attention :\n"
            "- Le passage à temps partiel peut impacter votre salaire, vos congés et l'organisation de travail.\n"
            "- Les délais de traitement varient selon la période et la complexité du dossier.\n"
            f"Sources: {sources}"
        )

    if standard_type == "performance_bonus":
        return (
            "Votre prime de performance est calculée à partir d'objectifs et d'indicateurs mesurables définis en amont. "
            "Selon votre poste, elle peut dépendre de votre performance individuelle, des résultats de votre équipe et, dans certains cas, d'objectifs collectifs ou commerciaux.\n\n"
            "Étapes :\n"
            "- Vérifiez la période de calcul applicable à votre prime (mensuelle, trimestrielle, semestrielle ou annuelle).\n"
            "- Comparez vos résultats aux objectifs ou KPI fixés.\n"
            "- La performance est évaluée par le manager, puis validée par les RH avant versement.\n\n"
            "Points d'attention :\n"
            "- La prime peut être réduite ou non versée si les objectifs ne sont pas atteints ou si certaines conditions ne sont pas remplies.\n"
            "- Elle apparaît sur votre bulletin de paie comme ligne distincte et peut expliquer une variation de salaire.\n"
            f"Sources: {sources}"
        )

    if standard_type == "sick_leave":
        return (
            "En cas d'arrêt maladie, vous devez prévenir votre manager rapidement et transmettre votre arrêt de travail dans le délai réglementaire, en général sous 48 heures. "
            "Les documents doivent ensuite être enregistrés pour permettre le traitement administratif de votre absence.\n\n"
            "Étapes :\n"
            "- Informez votre manager dès que possible.\n"
            "- Déclarez l'absence dans le SIRH si ce canal est utilisé.\n"
            "- Transmettez l'avis d'arrêt de travail aux RH et les volets nécessaires à la sécurité sociale.\n"
            "- En cas de prolongation, envoyez un nouveau certificat dans le même délai.\n\n"
            "Points d'attention :\n"
            "- Le maintien de salaire dépend notamment de votre ancienneté, de la convention collective et de la durée de l'arrêt.\n"
            "- Une visite de reprise peut être nécessaire selon la durée de l'absence.\n"
            f"Sources: {sources}"
        )

    return None


def build_profile_context(user: UserProfile, question: str) -> list[str]:
    lower = question.lower()
    facts = [
        "Utilisateur connecté: profil employé authentifié",
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
    sources = "\n".join(
        f"- Source [{chunk.doc_id}] | Titre: {chunk.title} | Section: {chunk.heading}\n  Extrait: {chunk.content}"
        for chunk in chunks
    )
    user_context = "\n".join(f"- {fact}" for fact in profile_facts)
    return f"""
Tu es un agent RH interne francophone. Tu réponds comme un gestionnaire RH clair, concret et utile.

Règles:
- N'invente rien hors des sources fournies.
- Si une information dépend du profil employé, utilise le contexte utilisateur.
- Adresse-toi toujours à l'utilisateur en disant "vous" ou "votre".
- N'utilise jamais le prénom ou le nom du salarié dans la réponse, sauf si la question demande explicitement son identité.
- N'écris pas à la troisième personne sur l'utilisateur connecté. Évite "le salarié", "l'employé", "Alice" ou toute formulation distante quand la question concerne sa propre situation.
- Réponds directement à la question. N'écris pas "consultez le fichier", "voir le document" ou "les informations sont dans...".
- Extrais l'information utile des extraits et reformule-la en réponse opérationnelle.
- Si la question porte sur une procédure, donne des étapes concrètes, dans l'ordre.
- Si la question porte sur des conditions ou impacts, explicite-les clairement.
- Si une information exacte n'est pas présente, dis seulement ce qui est confirmé par les sources.
- Ne mentionne les identifiants de documents qu'à la toute fin.
- Structure attendue:
  1. Une réponse directe en 2 à 5 phrases.
  2. Si utile, une section "Étapes :" avec des puces courtes.
  3. Si utile, une section "Points d'attention :" avec impacts, délais ou validations.
  4. Dernière ligne obligatoire: Sources: [id1], [id2]

Catégorie détectée: {intent.label}
Question: {question}

Contexte utilisateur:
{user_context}

Extraits RH autorisés:
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
    def __init__(
        self,
        users_path: Path | None = None,
        knowledge_dir: Path | None = None,
        history_path: Path | None = None,
    ):
        self.users = load_users(users_path or DATA_DIR / "users.json")
        self.knowledge = KnowledgeBase(knowledge_dir or KNOWLEDGE_DIR)
        self.history_path = history_path or CHAT_HISTORY_PATH

    def answer(self, user_id: str, question: str) -> dict:
        user = self.users.get(user_id)
        if not user:
            raise ValueError(f"Utilisateur inconnu: {user_id}")

        intent = classify_intent(question)
        profile_facts = build_profile_context(user, question)
        chunks = self.knowledge.retrieve(query=question, user_role=user.role, topic=intent.topic, limit=TOP_K)

        deterministic_answer = standard_answer(question, user, chunks)
        if deterministic_answer is not None:
            answer_text = deterministic_answer
            answer_mode = "standard"
        else:
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

        result = {
            "user": asdict(user),
            "intent": asdict(intent),
            "answer": answer_text,
            "sources": deduplicate_sources(chunks),
            "workflow": workflow,
        }
        self._append_history_entry(user=user, question=question, result=result)
        return result

    def _append_history_entry(self, user: UserProfile, question: str, result: dict) -> None:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "user_id": user.user_id,
            "user_role": user.role,
            "question": question,
            "answer": result["answer"],
            "intent": result["intent"],
            "sources": result["sources"],
            "workflow": result["workflow"],
        }
        try:
            self.history_path.parent.mkdir(parents=True, exist_ok=True)
            with self.history_path.open("a", encoding="utf-8") as history_file:
                history_file.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError:
            # In some Docker setups the app filesystem is read-only.
            # History persistence should not break the chat endpoint.
            return

    def list_history(self, user_id: str | None = None, limit: int = 50) -> list[dict]:
        if limit <= 0:
            return []
        if not self.history_path.exists():
            return []

        entries: list[dict] = []
        for line in self.history_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if user_id and entry.get("user_id") != user_id:
                continue
            entries.append(entry)
        return entries[-limit:]

    def get_document_view(self, user_id: str, doc_id: str) -> dict:
        user = self.users.get(user_id)
        if not user:
            raise ValueError(f"Utilisateur inconnu: {user_id}")
        document = self.knowledge.get_document(doc_id)
        if document is None:
            raise ValueError(f"Document inconnu: {doc_id}")
        if user.role not in document.allowed_roles:
            raise PermissionError(f"Accès refusé au document: {doc_id}")

        return {
            "doc_id": document.doc_id,
            "title": document.title,
            "allowed_roles": document.allowed_roles,
            "tags": document.tags,
            "sections": [
                {"heading": section.heading, "body": section.body}
                for section in document.sections
            ],
        }
