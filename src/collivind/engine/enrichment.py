"""Semantic enrichment for memory and query embeddings.

Expands text before embedding to improve recall. The original content
is stored unchanged — only the vector captures the wider semantic field.
"""

from typing import List, Optional

from collivind.models.memory import MemoryCategory, MemoryCreate

CATEGORY_KEYWORDS = {
    MemoryCategory.FACT: ["factual information", "known state"],
    MemoryCategory.DECISION: ["decision made", "choice", "rationale"],
    MemoryCategory.PATTERN: ["recurring pattern", "best practice", "convention"],
    MemoryCategory.ERROR: ["bug", "error", "failure", "fix", "debugging"],
    MemoryCategory.ARCHITECTURE: ["system design", "architecture", "infrastructure"],
    MemoryCategory.PREFERENCE: ["preference", "configuration", "setting", "style"],
    MemoryCategory.SNIPPET: ["code example", "implementation", "usage"],
}

QUERY_EXPANSIONS = {
    "db": ["database", "storage", "data"],
    "api": ["endpoint", "REST", "interface", "HTTP"],
    "ci": ["continuous integration", "pipeline", "build"],
    "cd": ["continuous deployment", "deploy", "release"],
    "auth": ["authentication", "authorization", "login", "security"],
    "config": ["configuration", "settings", "preferences"],
    "perf": ["performance", "speed", "latency", "optimization"],
    "test": ["testing", "unit test", "integration test", "QA"],
    "deploy": ["deployment", "release", "infrastructure"],
    "cache": ["caching", "redis", "memoization"],
    "error": ["bug", "failure", "exception", "crash"],
    "framework": ["library", "tool", "dependency"],
}


def build_enriched_text(
    memory: MemoryCreate,
    entity_names: Optional[List[str]] = None,
) -> str:
    """Build an enriched text for embedding that improves recall."""
    parts = [memory.content]

    if memory.summary and memory.summary != memory.content:
        parts.append(memory.summary)

    cat_keywords = CATEGORY_KEYWORDS.get(memory.category, [])
    if cat_keywords:
        parts.append(" ".join(cat_keywords))

    if entity_names:
        parts.append(" ".join(entity_names))

    if memory.tags:
        parts.append(" ".join(memory.tags))

    return " | ".join(parts)


def build_enriched_query(
    query: str,
    category: Optional[str] = None,
    tags: Optional[List[str]] = None,
    entity_names: Optional[List[str]] = None,
) -> str:
    """Expand a search query with synonyms and contextual terms."""
    parts = [query]

    query_lower = query.lower()
    for trigger, expansions in QUERY_EXPANSIONS.items():
        if trigger in query_lower:
            parts.append(" ".join(expansions))

    if category:
        try:
            cat = MemoryCategory(category)
            cat_kw = CATEGORY_KEYWORDS.get(cat, [])
            if cat_kw:
                parts.append(" ".join(cat_kw))
        except ValueError:
            pass

    if tags:
        parts.append(" ".join(tags))

    if entity_names:
        parts.append(" ".join(entity_names))

    return " | ".join(parts)
