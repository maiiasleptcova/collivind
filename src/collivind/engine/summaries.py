"""Summary memories: an agent-written list of dated facts over a group of memories.

A group is either one session (every live memory with that session_id, the
placeholder that exists before any summary text) or a named topic (the
memories an agent picks). Its summary is an ordinary memory with category
"summary", a `summary-of:<kind>:<key>` tag naming the group, and one
SUMMARIZES edge to each member. Rewriting it mints a new memory that
supersedes the old one (the usual version chain) and links the old members
plus the new ones. Collivind never writes summary text itself: the agent
reads the group (`group_view`) and passes the text back (`save_summary`).

Summaries skip the add_memory pipeline on purpose: near-duplicate dedup would
let a summary retire an unrelated memory or the reverse, and the conflict
check compares same-category memories, which for summaries is meaningless.
"""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from collivind.engine.enrichment import build_enriched_text
from collivind.models import MemoryCreate, MemoryNode, RelationshipCreate, RelType
from collivind.models.entity import entity_slug
from collivind.models.memory import MemoryCategory, MemorySource

GROUP_TAG = "summary-of:"
MAX_SUMMARY_CHARS = 20_000

SUMMARY_INSTRUCTION = (
    "Write the summary of these memories as a list of self-contained facts, one per line, each starting "
    'with "- ". Name who or what each fact is about (no bare pronouns), give absolute dates (resolve '
    '"yesterday", "next week" against the memory dates), and keep every concrete detail: decisions, '
    "names, numbers, versions, file paths, errors and their fixes, open questions. No greeting or "
    "commentary. Aim for roughly half the length of the members combined, not a short abstract: the "
    "detail is what makes recall work. Then call this tool again with `summary` and the same `member_ids`."
)


def is_summary(memory: Optional[MemoryNode]) -> bool:
    return memory is not None and memory.category == MemoryCategory.SUMMARY  # str enum: neo4j reads a str


def is_live(memory: Optional[MemoryNode]) -> bool:
    return memory is not None and not memory.valid_to and not memory.superseded_by


@dataclass(frozen=True)
class Group:
    kind: str  # "session" or "topic"
    key: str

    @property
    def tag(self) -> str:
        return f"{GROUP_TAG}{self.kind}:{self.key}"

    @property
    def session_id(self) -> Optional[str]:
        return self.key if self.kind == "session" else None


def group_for(session_id: Optional[str] = None, topic: Optional[str] = None) -> Group:
    """Exactly one of a session id or a topic name."""
    session_id = (session_id or "").strip() or None
    topic = (topic or "").strip() or None
    if bool(session_id) == bool(topic):
        raise ValueError("give exactly one of session_id or topic")
    if session_id:
        return Group("session", session_id)
    slug = entity_slug(topic or "")
    if not slug:
        raise ValueError("topic name has no usable characters")
    return Group("topic", slug)


def group_of(summary: MemoryNode) -> Optional[Group]:
    for tag in summary.tags or []:
        if tag.startswith(GROUP_TAG):
            kind, _, key = tag[len(GROUP_TAG) :].partition(":")
            if kind in ("session", "topic") and key:
                return Group(kind, key)
    return None


def live_summaries(manager, project_id: str) -> List[MemoryNode]:
    return manager.graph_store.find_memories(project_id, category=MemoryCategory.SUMMARY.value)


def group_summaries(manager, project_id: str, group: Group) -> List[MemoryNode]:
    """Every live summary of a group, newest first (more than one only after an interrupted save)."""
    found = [s for s in live_summaries(manager, project_id) if group.tag in (s.tags or [])]
    return sorted(found, key=lambda s: s.created_at, reverse=True)


def live_summary(manager, project_id: str, group: Group) -> Optional[MemoryNode]:
    found = group_summaries(manager, project_id, group)
    return found[0] if found else None


def _head(manager, memory: Optional[MemoryNode], cache: Dict[str, Optional[MemoryNode]]) -> Optional[MemoryNode]:
    """Follow a superseded member to its current version (a dedup merge or an invalidate)."""
    seen = set()
    while memory is not None and memory.superseded_by and memory.superseded_by not in seen:
        seen.add(memory.superseded_by)
        nxt = memory.superseded_by
        if nxt not in cache:
            cache[nxt] = manager.graph_store.get_memory(nxt)
        memory = cache[nxt]
    return memory


def member_ids(manager, summary_ids: Iterable[str]) -> Dict[str, List[str]]:
    """The ids each summary links to, as stored (no node reads)."""
    ids = list(dict.fromkeys(summary_ids))
    if not ids:
        return {}
    links = manager.graph_store.get_neighbors_batch(ids, [RelType.SUMMARIZES.value], direction="OUT")
    return {sid: list(dict.fromkeys(n.get("id") for n in links.get(sid) or [] if n.get("id"))) for sid in ids}


def members_of(manager, summary_ids: Iterable[str], cap: Optional[int] = None) -> Dict[str, List[MemoryNode]]:
    """Live member memories of each summary, in edge order; a superseded member
    counts as its current version. `cap` bounds the node reads per summary.

    ponytail: one get_memory per member (a Bolt round trip in docker mode);
    add a batch read to GraphStore if routing latency on Neo4j matters.
    """
    cache: Dict[str, Optional[MemoryNode]] = {}
    out: Dict[str, List[MemoryNode]] = {}
    for sid, mids in member_ids(manager, summary_ids).items():
        members: Dict[str, MemoryNode] = {}
        for mid in mids[:cap] if cap else mids:
            if mid not in cache:
                cache[mid] = manager.graph_store.get_memory(mid)
            m = _head(manager, cache[mid], cache)
            if m is not None and is_live(m) and not is_summary(m):
                members[m.id] = m
        out[sid] = list(members.values())
    return out


def summaries_of(manager, memory_id: str) -> List[MemoryNode]:
    """Live summaries a memory belongs to (normally one)."""
    links = manager.graph_store.get_neighbors(memory_id, [RelType.SUMMARIZES.value], direction="IN")
    found = [manager.graph_store.get_memory(n.get("id")) for n in links]
    return [s for s in found if is_live(s) and is_summary(s)]


def _session_members(manager, project_id: str, group: Group) -> List[MemoryNode]:
    if group.kind != "session":
        return []
    return [m for m in manager.graph_store.find_memories(project_id, session_id=group.key) if not is_summary(m)]


def unsummarized(manager, project_id: str, group: Group) -> List[MemoryNode]:
    """Session members no live summary of the group links yet (a topic has no implicit members)."""
    current = live_summary(manager, project_id, group)
    linked = set(member_ids(manager, [current.id])[current.id]) if current else set()
    return [m for m in _session_members(manager, project_id, group) if m.id not in linked]


def group_view(manager, project_id: str, group: Group, member_ids: Optional[List[str]] = None) -> Dict[str, Any]:
    """What an agent needs to write (or rewrite) a group's summary."""
    current = live_summary(manager, project_id, group)
    linked = members_of(manager, [current.id])[current.id] if current else []
    extra = _checked_members(manager, project_id, member_ids or []) if member_ids else []
    pool = {m.id: m for m in linked + _session_members(manager, project_id, group) + extra}
    members = sorted(pool.values(), key=lambda m: m.created_at)
    linked_ids = {m.id for m in linked}
    return {
        "group": {"kind": group.kind, "key": group.key},
        "project_id": project_id,
        "current_summary": (
            {"id": current.id, "version": current.version, "content": current.content} if current else None
        ),
        "member_ids": [m.id for m in members],
        "members": [
            {
                "id": m.id,
                "date": m.created_at.date().isoformat(),
                "category": getattr(m.category, "value", m.category),
                "content": m.content,
                "new": m.id not in linked_ids,
            }
            for m in members
        ],
        "instruction": SUMMARY_INSTRUCTION
        + (" Rewrite the current summary so it also covers the members marked new." if current else ""),
    }


def _checked_members(manager, project_id: str, member_ids: List[str]) -> List[MemoryNode]:
    out = []
    for mid in dict.fromkeys(member_ids):
        m = manager.graph_store.get_memory(mid)
        if m is None:
            raise LookupError(f"no memory {mid!r}")
        if is_summary(m):
            raise ValueError(f"{mid} is a summary; a summary cannot be a member of another")
        if m.project_id != project_id:
            raise ValueError(f"{mid} belongs to project {m.project_id!r}, not {project_id!r}")
        if not is_live(m):
            raise ValueError(f"{mid} has been superseded or invalidated; use its current version")
        out.append(m)
    return out


def _validate_text(text: str) -> str:
    text = (text or "").strip()
    if not text:
        raise ValueError("summary text is required")
    if len(text) > MAX_SUMMARY_CHARS:
        raise ValueError(f"summary is {len(text)} characters; keep it under {MAX_SUMMARY_CHARS}")
    if not any(line.lstrip().startswith("- ") for line in text.splitlines()):
        raise ValueError('a summary is a list of facts: put each one on its own line starting with "- "')
    return text


def _title(text: str, group: Group, n: int) -> str:
    """What lists and graph labels show; the member count is drawn separately."""
    first = next((ln.strip()[2:] for ln in text.splitlines() if ln.strip().startswith("- ")), "")
    where = "session" if group.kind == "session" else f"topic {group.key}"
    return f"Summary: {first[:110]}" if first else f"Summary of {n} memories ({where})"


def save_summary(
    manager, project_id: str, group: Group, text: str, member_ids: Optional[List[str]] = None
) -> MemoryNode:
    """Store (or rewrite) a group's summary and link it to its members.

    member_ids defaults to every live memory of a session group; a topic
    group needs them. A rewrite keeps the previous summary's live members, so
    a summary only ever grows, and supersedes the previous summary.
    """
    text = _validate_text(text)
    if member_ids is None and group.kind == "session":
        given = _session_members(manager, project_id, group)
    else:
        given = _checked_members(manager, project_id, member_ids or [])
    previous = group_summaries(manager, project_id, group)
    kept = [m for ms in members_of(manager, [p.id for p in previous]).values() for m in ms]
    members = list({m.id: m for m in kept + given}.values())
    if not members:
        raise ValueError("a summary needs at least one member memory")

    create = MemoryCreate(
        content=text,
        summary=_title(text, group, len(members)),
        category=MemoryCategory.SUMMARY,
        project_id=project_id,
        session_id=group.session_id,
        source=MemorySource.MANUAL,
        tags=[group.tag],
    )
    vector = manager.embedding_provider.embed(build_enriched_text(create))  # before any write
    node = manager._graph_write(manager.graph_store.create_memory, create, outcome="the summary was NOT saved")
    for m in members:
        manager.graph_store.create_relationship(
            RelationshipCreate(source_id=node.id, target_id=m.id, type=RelType.SUMMARIZES, source="summary")
        )
    # Supersede only once the new summary is complete. An interrupted save
    # leaves the old one live next to it; the next save supersedes both
    # (every live summary of the group) and keeps all their members.
    for old in previous:
        manager.graph_store.invalidate_memory(old.id, node.id, "summary_rewritten")
        manager.vector_store.delete(old.id)
    node = manager.graph_store.get_memory(node.id) or node  # with its version fields
    manager.vector_store.upsert(node.id, vector, node.to_dict())
    return node


def summary_due(manager, project_id: str, session_id: Optional[str], threshold: int) -> Optional[Dict[str, Any]]:
    """A request for the agent when a session holds `threshold`+ memories no summary covers."""
    if not session_id or threshold <= 0:
        return None
    group = Group("session", session_id)
    pending = unsummarized(manager, project_id, group)
    if len(pending) < threshold:
        return None
    return {
        "session_id": session_id,
        "project_id": project_id,
        "new_memories": len(pending),
        "instruction": (
            f"This session has {len(pending)} memories no summary covers. Call `collivind_summarize` with "
            f'session_id "{session_id}" to read them, then call it again with the summary text.'
        ),
    }
