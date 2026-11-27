"""What a topic is, in one or two sentences an agent wrote.

The explorer's wiki page for an entity opens with this. Collivind never
writes it (no model runs here): the agent does, through `collivind_describe_topic`
or `collivind describe`, when a checkpoint reports `description_due`. Without
one, the page builds a plain sentence from the entity's type and links.
Stored in the entity's own properties, so export, import and migrate carry it.
"""

from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

from collivind.models.entity import EntityNode
from collivind.models.relationship import RelType

DESCRIPTION_KEY = "description"
MAX_DESCRIPTION = 400
MAX_DUE = 3  # topics asked about per save, so a checkpoint never turns into a writing job
_MEMBER_TYPES = {RelType.ABOUT.value, RelType.MENTIONS.value}


def clean_description(text: Any) -> str:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("description must be a non-empty string")
    flat = " ".join(text.split())
    if len(flat) > MAX_DESCRIPTION:
        raise ValueError(f"description is longer than {MAX_DESCRIPTION} characters; one or two sentences")
    return flat


def describe_entity(manager, name: str, text: Any) -> Dict[str, Any]:
    """Store `text` as the description of the entity `name` (an alias finds its entity). LookupError if unknown."""
    description = clean_description(text)
    entity = manager.graph_store.get_entity(manager.canonical_entity_name(name))
    if entity is None:
        raise LookupError(f"No entity named {name!r}")
    previous = (entity.properties or {}).get(DESCRIPTION_KEY) or ""
    manager.graph_store.restore_entity(
        EntityNode(
            id=entity.id,
            name=entity.name,
            type=entity.type,
            properties={**(entity.properties or {}), DESCRIPTION_KEY: description},  # aliases survive
            created_at=entity.created_at,
            updated_at=datetime.now(timezone.utc),
        )
    )
    return {"entity": entity.name, "id": entity.id, "description": description, "replaced": previous}


def _live_members(graph, entity_id: str, enough: int) -> int:
    """Live memories about the entity, counted up to `enough` (a hub has hundreds)."""
    count = 0
    for edge in graph.edges_of(entity_id):
        rel = edge.type.value if hasattr(edge.type, "value") else edge.type
        if rel in _MEMBER_TYPES and edge.target_id == entity_id:
            m = graph.get_memory(edge.source_id)
            if m is not None and not (m.superseded_by or m.valid_to):
                count += 1
                if count >= enough:
                    break
    return count


def description_due(manager, memory_ids: Iterable[str], threshold: int) -> Optional[Dict[str, Any]]:
    """Topics the just-saved memories are about that have `threshold`+ memories and no description.

    Reads only the edges of the batch's own entities, never the whole store.
    """
    if threshold <= 0:
        return None
    graph = manager.graph_store
    seen: List[str] = []
    for memory_id in memory_ids:
        for edge in graph.edges_of(memory_id):
            rel = edge.type.value if hasattr(edge.type, "value") else edge.type
            if rel in _MEMBER_TYPES and edge.source_id == memory_id and edge.target_id not in seen:
                seen.append(edge.target_id)
    due = []
    for entity_id in seen:
        entity = graph.get_entity(entity_id)
        if entity is None or (entity.properties or {}).get(DESCRIPTION_KEY):
            continue
        if _live_members(graph, entity_id, threshold) >= threshold:
            due.append({"entity": entity.name})
        if len(due) >= MAX_DUE:
            break
    if not due:
        return None
    names = ", ".join(f'"{d["entity"]}"' for d in due)
    return {
        "topics": due,
        "instruction": (
            f"These topics have enough memories but no description: {names}. For each, call "
            "`collivind_describe_topic` with `entity_name` and one or two sentences saying what it is and what it "
            "is for in this project (a reader of the memory wiki sees it first)."
        ),
    }
