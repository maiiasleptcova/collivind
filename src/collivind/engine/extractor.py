"""LLM-powered extraction of structured memories from raw text.

Produces category, summary, entities, relationships, and tags from
unstructured conversation content. Used by hooks and the auto-extract tool.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from collivind.models.entity import EntityType
from collivind.models.memory import MemoryCategory

logger = logging.getLogger(__name__)

VALID_CATEGORIES = {c.value for c in MemoryCategory}
VALID_ENTITY_TYPES = {t.value for t in EntityType}

EXTRACTION_PROMPT = """\
You are a memory extraction system. Analyze the following text and extract \
structured memories that would be valuable for a software development AI assistant.

TEXT:
{text}

PROJECT: {project_id}

Extract memories as a JSON array. Each memory object must have:
- "content": the factual statement (1-2 sentences, self-contained)
- "summary": short label (3-6 words)
- "category": one of {categories}
- "confidence": 0.0-1.0 how certain this fact is
- "tags": relevant keyword tags (array of strings)
- "entities": array of {{"name": "...", "type": "..."}} where type is one of {entity_types}

Rules:
- Extract only concrete, reusable facts — skip greetings, questions, and filler
- Each memory should be independently understandable without context
- Prefer specific names over pronouns
- If no extractable memories exist, return an empty array

Respond with ONLY the JSON array, no other text."""


@dataclass
class ExtractionResult:
    content: str
    summary: str
    category: str
    confidence: float = 1.0
    tags: List[str] = field(default_factory=list)
    entities: List[Dict[str, str]] = field(default_factory=list)


def build_extraction_prompt(text: str, project_id: str = "default") -> str:
    return EXTRACTION_PROMPT.format(
        text=text,
        project_id=project_id,
        categories=", ".join(sorted(VALID_CATEGORIES)),
        entity_types=", ".join(sorted(VALID_ENTITY_TYPES)),
    )


def parse_extraction_response(response_text: str) -> List[ExtractionResult]:
    text = response_text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [line for line in lines if not line.strip().startswith("```")]
        text = "\n".join(lines)

    try:
        raw = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("Failed to parse extraction response as JSON")
        return []

    if not isinstance(raw, list):
        raw = [raw]

    results = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        content = item.get("content", "").strip()
        if not content:
            continue

        category = item.get("category", "fact")
        if category not in VALID_CATEGORIES:
            category = "fact"

        entities = []
        for e in item.get("entities", []):
            if isinstance(e, dict) and "name" in e:
                etype = e.get("type", "concept")
                if etype not in VALID_ENTITY_TYPES:
                    etype = "concept"
                entities.append({"name": e["name"], "type": etype})

        results.append(
            ExtractionResult(
                content=content,
                summary=item.get("summary", content[:50]),
                category=category,
                confidence=min(1.0, max(0.0, float(item.get("confidence", 1.0)))),
                tags=item.get("tags", []),
                entities=entities,
            )
        )

    return results


def extraction_results_to_add_args(
    results: List[ExtractionResult],
    project_id: str = "default",
    session_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Convert extraction results to arguments suitable for batch_add_memories."""
    memories = []
    for r in results:
        mem: Dict[str, Any] = {
            "content": r.content,
            "summary": r.summary,
            "category": r.category,
            "project_id": project_id,
            "confidence": r.confidence,
            "tags": r.tags,
            "entities": r.entities,
        }
        if session_id:
            mem["session_id"] = session_id
        memories.append(mem)
    return memories
