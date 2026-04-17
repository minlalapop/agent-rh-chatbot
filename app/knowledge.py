from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9àâçéèêëîïôûùüÿñæœ+-]+")


@dataclass
class UserProfile:
    user_id: str
    name: str
    role: str
    department: str
    contract_type: str
    hire_date: str
    contract_end_date: str | None
    trial_period_months: int
    trial_period_end: str
    manager: str
    n_plus_1: str
    leave_balance_days: float
    contract_download_url: str
    payslip_portal_url: str
    bank_account_last4: str


@dataclass
class DocumentChunk:
    chunk_id: str
    doc_id: str
    title: str
    heading: str
    allowed_roles: list[str]
    tags: list[str]
    content: str
    tokens: list[str]


@dataclass
class DocumentSection:
    heading: str
    body: str


@dataclass
class KnowledgeDocument:
    doc_id: str
    title: str
    allowed_roles: list[str]
    tags: list[str]
    sections: list[DocumentSection]


def tokenize(text: str) -> list[str]:
    normalized_tokens: list[str] = []
    for token in TOKEN_PATTERN.findall(text.lower()):
        normalized = token.lower()
        if normalized.endswith("s") and len(normalized) > 4:
            normalized = normalized[:-1]
        normalized_tokens.append(normalized)
    return normalized_tokens


def load_users(path: Path) -> dict[str, UserProfile]:
    raw_users = json.loads(path.read_text(encoding="utf-8"))
    users: dict[str, UserProfile] = {}
    for item in raw_users:
        users[item["user_id"]] = UserProfile(**item)
    return users


def parse_frontmatter(raw_text: str) -> tuple[dict[str, str], str]:
    if not raw_text.startswith("---"):
        return {}, raw_text
    parts = raw_text.split("---", 2)
    if len(parts) < 3:
        return {}, raw_text
    metadata_block = parts[1].strip()
    content = parts[2].strip()
    metadata: dict[str, str] = {}
    for line in metadata_block.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        metadata[key.strip()] = value.strip()
    return metadata, content


def split_sections(content: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = []
    current_heading = "Général"
    buffer: list[str] = []
    for line in content.splitlines():
        if line.strip() == "---":
            continue
        if line.startswith("## "):
            if buffer:
                sections.append((current_heading, "\n".join(buffer).strip()))
                buffer = []
            current_heading = line[3:].strip()
            continue
        if line.startswith("# "):
            continue
        buffer.append(line)
    if buffer:
        sections.append((current_heading, "\n".join(buffer).strip()))
    return sections


class KnowledgeBase:
    def __init__(self, knowledge_dir: Path):
        self.knowledge_dir = knowledge_dir
        self.documents = self._load_documents()
        self.chunks = self._build_chunks()
        self.idf = self._compute_idf()

    def _load_documents(self) -> dict[str, KnowledgeDocument]:
        documents: dict[str, KnowledgeDocument] = {}
        for doc_path in sorted(self.knowledge_dir.glob("*.md")):
            raw_text = doc_path.read_text(encoding="utf-8")
            metadata, content = parse_frontmatter(raw_text)
            doc_id = metadata.get("id", doc_path.stem)
            title = metadata.get("title", doc_id)
            allowed_roles = [role.strip() for role in metadata.get("allowed_roles", "employee,manager,hr").split(",")]
            tags = [tag.strip() for tag in metadata.get("tags", "").split(",") if tag.strip()]
            sections = [
                DocumentSection(heading=heading, body=body)
                for heading, body in split_sections(content)
            ]
            documents[doc_id] = KnowledgeDocument(
                doc_id=doc_id,
                title=title,
                allowed_roles=allowed_roles,
                tags=tags,
                sections=sections,
            )
        return documents

    def _build_chunks(self) -> list[DocumentChunk]:
        chunks: list[DocumentChunk] = []
        for document in self.documents.values():
            chunk_index = 1
            for section in document.sections:
                paragraphs = [part.strip() for part in section.body.split("\n\n") if part.strip()]
                for paragraph in paragraphs:
                    if paragraph == "---":
                        continue
                    text = f"{section.heading}. {paragraph}".strip()
                    chunks.append(
                        DocumentChunk(
                            chunk_id=f"{document.doc_id}#{chunk_index}",
                            doc_id=document.doc_id,
                            title=document.title,
                            heading=section.heading,
                            allowed_roles=document.allowed_roles,
                            tags=document.tags,
                            content=text,
                            tokens=tokenize(text),
                        )
                    )
                    chunk_index += 1
        return chunks

    def _compute_idf(self) -> dict[str, float]:
        doc_count = max(len(self.chunks), 1)
        doc_frequency: Counter[str] = Counter()
        for chunk in self.chunks:
            doc_frequency.update(set(chunk.tokens))
        return {
            term: math.log((doc_count + 1) / (frequency + 1)) + 1
            for term, frequency in doc_frequency.items()
        }

    def retrieve(self, query: str, user_role: str, topic: str | None, limit: int = 4) -> list[DocumentChunk]:
        query_tokens = tokenize(query)
        ranked: list[tuple[float, DocumentChunk]] = []
        for chunk in self.chunks:
            if user_role not in chunk.allowed_roles:
                continue
            score = self._score_chunk(query_tokens, chunk, topic)
            if score > 0:
                ranked.append((score, chunk))
        ranked.sort(key=lambda item: item[0], reverse=True)
        return [chunk for _, chunk in ranked[:limit]]

    def _score_chunk(self, query_tokens: list[str], chunk: DocumentChunk, topic: str | None) -> float:
        if not query_tokens:
            return 0.0
        chunk_counts = Counter(chunk.tokens)
        lexical = sum(chunk_counts[token] * self.idf.get(token, 1.0) for token in set(query_tokens))
        coverage = sum(1 for token in set(query_tokens) if token in chunk_counts)
        topic_bonus = 3.0 if topic and topic in chunk.tags else 0.0
        heading_bonus = 1.5 if any(token in tokenize(chunk.heading) for token in query_tokens) else 0.0
        return lexical + coverage + topic_bonus + heading_bonus

    def get_document(self, doc_id: str) -> KnowledgeDocument | None:
        return self.documents.get(doc_id)
