"""One merge scenario that runs against any real GraphStore (SQLite in unit, Neo4j in integration).

Cybergine is split across 'Cybergine', 'v5_flask' and 'Cybergine v5_flask'.
After merging the two into Cybergine, the whole edge set must equal EXPECTED.
"""

from collivind.models import EntityCreate, EntityType, MemoryCategory, MemoryCreate, RelationshipCreate, RelType
from collivind.storage.interfaces import GraphStore

KEEP, OTHERS = "Cybergine", ("v5_flask", "Cybergine v5_flask")
GONE = {"v5_flask", "cybergine_v5_flask"}


def build(graph: GraphStore) -> dict:
    """Seed the split store; return {label: memory id}."""
    for name in ("Cybergine", "v5_flask", "Cybergine v5_flask", "cytm", "Flask", "Docker"):
        graph.create_entity(EntityCreate(name=name, type=EntityType.PROJECT))
    mem = {}
    for label in ("m1", "m2", "m3", "m4"):
        mem[label] = graph.create_memory(
            MemoryCreate(content=f"{label} text", summary=label, category=MemoryCategory.FACT, project_id="v5_flask")
        ).id

    def link(s, t, rel, note=""):
        graph.create_relationship(
            RelationshipCreate(
                source_id=s,
                target_id=t,
                type=rel,
                confidence=0.7,
                source="extraction",
                properties={"note": note} if note else {},
            )
        )

    link(mem["m1"], "cybergine", RelType.ABOUT)
    link(mem["m1"], "v5_flask", RelType.ABOUT)  # becomes a duplicate
    link(mem["m2"], "v5_flask", RelType.MENTIONS)
    link(mem["m2"], "v5_flask", RelType.MENTIONS)  # stacked twin (SQLite keeps both rows)
    link(mem["m3"], "cybergine_v5_flask", RelType.ABOUT)
    link(mem["m4"], "docker", RelType.ABOUT)  # untouched
    link("cybergine", "cytm", RelType.USES)
    link("cybergine_v5_flask", "cytm", RelType.USES)  # becomes a duplicate
    link("v5_flask", "flask", RelType.USES, note="web framework")  # outgoing entity edge
    link("docker", "v5_flask", RelType.INTERACTS_WITH)  # incoming entity edge
    link("cybergine_v5_flask", "cybergine", RelType.PART_OF)  # becomes a self-loop
    link("v5_flask", "cybergine_v5_flask", RelType.PART_OF)  # between two merged: self-loop
    return mem


def expected(mem: dict) -> set:
    return {
        (mem["m1"], "ABOUT", "cybergine"),
        (mem["m2"], "MENTIONS", "cybergine"),
        (mem["m3"], "ABOUT", "cybergine"),
        (mem["m4"], "ABOUT", "docker"),
        ("cybergine", "USES", "cytm"),
        ("cybergine", "USES", "flask"),
        ("docker", "INTERACTS_WITH", "cybergine"),
    }


def edge_set(graph: GraphStore) -> list:
    """Every edge as a triple; a list so a duplicate row shows up as a repeat."""
    return sorted((e.source_id, getattr(e.type, "value", e.type), e.target_id) for e in graph.iter_relationships())
