"""What each edge type means, in words an agent or a person can read.

One table feeds the MCP tool descriptions, the extraction prompt, the
explorer's legend and panel, and `get_entity`'s traversal output, so an edge
reads the same everywhere: "Qdrant  is part of  the docker stack", and from
the other end "the docker stack  contains  Qdrant".
"""

from dataclasses import dataclass
from typing import Any, Dict, NamedTuple, Tuple, Union

from collivind.models.relationship import RelationshipCreate, RelType

# Free text an agent or person attaches to one edge: why or how it holds.
NOTE_KEY = "note"
MAX_NOTE = 300


class Meaning(NamedTuple):
    forward: str  # read from the source: "A <forward> B"
    inverse: str  # read from the target: "B <inverse> A"
    description: str  # when to use it


MEANINGS: Dict[str, Meaning] = {
    # entity -> entity: how things in the system fit together
    "DEPENDS_ON": Meaning("depends on", "is needed by", "cannot work without the other"),
    "USES": Meaning("uses", "is used by", "calls, imports or runs the other as a tool"),
    "PART_OF": Meaning("is part of", "contains", "a component, module or member of the other"),
    "IMPLEMENTS": Meaning("implements", "is implemented by", "realises an interface, spec, feature or idea"),
    "INTERACTS_WITH": Meaning(
        "interacts with", "interacts with", "talks to, sends data to or integrates with, as peers"
    ),
    "RELATED_TO": Meaning("is related to", "is related to", "connected, but none of the above fits"),
    # memory -> entity
    "ABOUT": Meaning("is about", "is the subject of", "the memory's main subject"),
    "MENTIONS": Meaning("mentions", "is mentioned by", "named in passing"),
    # memory -> memory: how knowledge connects
    "CAUSED_BY": Meaning("was caused by", "caused", "the other is the cause or trigger"),
    "RESOLVES": Meaning("resolves", "is resolved by", "a fix or decision that settles the other"),
    "SUPERSEDES": Meaning("supersedes", "is superseded by", "a newer version of the other"),
    "SUMMARIZES": Meaning("summarizes", "is summarized by", "a summary and one of the detail memories it covers"),
    "CONTRADICTS": Meaning("contradicts", "is contradicted by", "both cannot be true; one needs retiring"),
    "RELATES_TO": Meaning("relates to", "relates to", "on the same subject, no stronger link known"),
    # session edges, written by capture only
    "PRODUCED": Meaning("produced", "was produced by", "a session wrote the memory"),
    "WORKED_ON": Meaning("worked on", "was worked on in", "a session touched the entity"),
}


def phrase(rel_type: str, outgoing: bool = True) -> str:
    meaning = MEANINGS.get(rel_type)
    if meaning is None:
        return rel_type.lower().replace("_", " ")
    return meaning.forward if outgoing else meaning.inverse


def glossary(types: Tuple[str, ...]) -> str:
    """'USES (uses: calls, imports or runs ...); PART_OF (...)' for tool descriptions and prompts."""
    return "; ".join(f"{t} ({MEANINGS[t].forward}: {MEANINGS[t].description})" for t in types if t in MEANINGS)


def clean_note(value: object) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError("note must be a string")
    note = " ".join(value.split())
    if len(note) > MAX_NOTE:
        raise ValueError(f"note is longer than {MAX_NOTE} characters")
    return note


ENTITY_RELATIONS = ("DEPENDS_ON", "USES", "PART_OF", "IMPLEMENTS", "INTERACTS_WITH", "RELATED_TO")


@dataclass(frozen=True)
class EntityRelation:
    """'<source entity> <type> <target entity>', named by the agent, saved with a memory."""

    source: str
    type: str
    target: str
    note: str = ""


# A target_index in a batch save, until the memory it points at has an id.
BATCH_REF = "batch-index:"


def parse_relationship(raw: Dict[str, Any], in_batch: bool = False) -> Union[RelationshipCreate, EntityRelation]:
    """One item of a save's `relationships` list.

    `{target_id, type}` links the new memory to an existing node, as before.
    `{target_index, type}` (batch saves only) links it to an earlier memory of
    the same batch, which has no id yet: the error and its fix saved together.
    `{source, type, target}` (entity names) states how two entities relate;
    unknown names are created as concepts. Any of them may carry a `note`.
    """
    if not isinstance(raw, dict):
        raise ValueError("each relationship must be an object")
    note = clean_note(raw.get("note"))
    rel_type = RelType(raw.get("type"))
    if "source" in raw or "target" in raw:
        names = [raw.get("source"), raw.get("target")]
        if not all(isinstance(n, str) and n.strip() for n in names):
            raise ValueError("an entity relationship needs both a source and a target name")
        if rel_type.value not in ENTITY_RELATIONS:
            raise ValueError(f"{rel_type.value} cannot link two entities; use one of {', '.join(ENTITY_RELATIONS)}")
        return EntityRelation(str(names[0]).strip(), rel_type.value, str(names[1]).strip(), note)
    properties = dict(raw.get("properties") or {})
    if note:
        properties[NOTE_KEY] = note
    if "target_id" not in raw and "target_index" in raw:
        index = raw["target_index"]
        if not in_batch:
            raise ValueError("target_index only works in collivind_batch_add; use target_id")
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise ValueError("target_index must be the position (0, 1, ...) of an earlier memory in this batch")
        return RelationshipCreate(source_id="", target_id=f"{BATCH_REF}{index}", type=rel_type, properties=properties)
    return RelationshipCreate(source_id="", target_id=raw["target_id"], type=rel_type, properties=properties)
