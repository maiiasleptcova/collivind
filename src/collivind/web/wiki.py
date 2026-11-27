"""The store read as a wiki: one page per topic, written in words, not drawn as dots.

The graph shows that two memories are linked; it does not say what a topic
is, what each memory under it is for, or why two things are connected. A page
does, the way OpenWiki's reader does for a codebase (langchain-ai/openwiki,
whose layout and Open Knowledge Format this borrows; no code is copied):

- a topic is an entity memories are about, a summary, or a project;
- it opens with what it is: the description an agent stored on it
  (`describe_topic`), else a sentence built from its type and typed links;
- its memories are grouped by purpose (problems with their fixes, decisions,
  how it works, facts ...), each as its label, date and a one-line gist;
- every related topic and every link between memories carries its reason in
  words ("uses", "12 shared memories", "its summary covers 8 of these").

Everything is structural and deterministic: no model call, same store, same
pages. The reader (/api/wiki) and the OKF export (web/okf.py) render the same
page dicts, so what a reader sees is what a commit of the export contains.
"""

import math
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from collivind.engine.projects import META_PREFIX, is_meta_entity
from collivind.models.memory import MemoryNode
from collivind.models.relations import ENTITY_RELATIONS, phrase
from collivind.models.relationship import RelType
from collivind.storage.interfaces import GraphStore
from collivind.web.graph import HUB_DEGREE, _enum, load

DESCRIPTION_KEY = "description"  # entity.properties["description"], written by the agent
PROJECT_PREFIX = META_PREFIX  # a project topic's id: "project:<project id>"
PAGE_SIZE = 8  # memories per section before "show more"; a hub is read in pages
MAX_PAGE = 200
RELATED_LIMIT = 12
MAP_LINKS = 5  # shared-memory links per topic drawn on the topic map
GIST_LEN = 180

# section key, heading, categories, what the heading means to a stranger
SECTIONS: Tuple[Tuple[str, str, Tuple[str, ...]], ...] = (
    ("problems", "Problems and fixes", ("error",)),
    ("decisions", "Decisions", ("decision",)),
    ("how", "How it works", ("architecture", "pattern")),
    ("facts", "Facts", ("fact",)),
    ("preferences", "Preferences", ("preference",)),
    ("snippets", "Snippets", ("snippet",)),
)
SECTION_OF = {cat: key for key, _, cats in SECTIONS for cat in cats}
KIND_WORD = {
    "error": "Problem",
    "decision": "Decision",
    "architecture": "How it works",
    "pattern": "Pattern",
    "fact": "Fact",
    "preference": "Preference",
    "snippet": "Snippet",
    "summary": "Summary",
}
PLURAL_WORD = {
    "error": "problems",
    "decision": "decisions",
    "architecture": "notes on how it works",
    "pattern": "patterns",
    "fact": "facts",
    "preference": "preferences",
    "snippet": "snippets",
}
MEMBER_TYPES = {RelType.ABOUT.value, RelType.MENTIONS.value}
# memory -> memory links a reader should see on the row, strongest first
ROW_LINKS = ("RESOLVES", "CAUSED_BY", "SUPERSEDES", "CONTRADICTS", "DEPENDS_ON", "RELATES_TO")
MAX_ROW_LINKS = 4


def _date(m: MemoryNode) -> str:
    return m.created_at.date().isoformat() if m.created_at else ""


def _newest_first(m: MemoryNode) -> Tuple[str, str]:
    return (m.created_at.isoformat() if m.created_at else "", m.id)


def _sentences(text: str) -> List[str]:
    flat = " ".join((text or "").split())
    return [s for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9`\"'(])", flat) if s]


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def gist(m: MemoryNode) -> str:
    """One line that adds to the label: the first sentence of the content the label does not already say."""
    label = _norm(m.summary)
    for sentence in _sentences(m.content):
        said = _norm(sentence)
        # skip what the label already says, including a label cut from this very sentence
        if not said or (label and (said in label or said.startswith(label[: max(20, len(label) - 3)]))):
            continue
        return sentence if len(sentence) <= GIST_LEN else sentence[: GIST_LEN - 1].rstrip() + "…"
    return ""


def label(m: MemoryNode) -> str:
    return (m.summary or "").strip() or (_sentences(m.content) or [m.content or m.id])[0][:140]


def _counted(n: int, word: str) -> str:
    """'1 memory', '2 memories', '3 problems'."""
    if n == 1:
        return f"{n} {word}"
    return f"{n} {word[:-1]}ies" if word.endswith("y") and word[-2:-1] not in "aeiou" else f"{n} {word}s"


def _and(items: List[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


@dataclass(frozen=True)
class Topic:
    id: str
    kind: str  # "entity", "summary" or "project"
    type: str  # the entity type, "summary" or "project"
    title: str
    description: str
    described_by: str  # "agent" when someone wrote it, else "collivind"
    members: Tuple[str, ...]  # live memory ids, newest first


class Wiki:
    """Every page of one store (or one project), from a single read."""

    def __init__(self, graph: GraphStore, project: Optional[str] = None):
        data = load(graph, project)
        self.project = project
        self.memories: Dict[str, MemoryNode] = {m.id: m for m in data.memories}
        self.live = {i for i, m in self.memories.items() if not (m.superseded_by or m.valid_to)}
        self.entities = data.entities
        self.notes = {k: v[0] for k, v in data.detail.items()}
        self.metas = {e.id[len(META_PREFIX) :]: e.properties or {} for e in graph.iter_entities() if is_meta_entity(e)}

        about: Dict[str, Set[str]] = {}  # entity -> live memories about or mentioning it
        self.attached: Dict[str, List[str]] = {}  # memory -> entities
        self.entity_links: List[Tuple[str, str, str]] = []
        self.memory_links: List[Tuple[str, str, str]] = []
        covers: Dict[str, Set[str]] = {}  # summary -> live members
        for s, rel, t in sorted(set(data.raw + data.derived)):
            if rel in MEMBER_TYPES and s in self.live and t in self.entities:
                about.setdefault(t, set()).add(s)
            elif rel in ENTITY_RELATIONS and s in self.entities and t in self.entities and s != t:
                self.entity_links.append((s, rel, t))
            elif rel == RelType.SUMMARIZES.value and s in self.live:
                member = self._current(t)
                if member in self.live:
                    covers.setdefault(s, set()).add(member)
            elif s in self.memories and t in self.memories and s != t:
                self.memory_links.append((s, rel, t))
        for e, ids in about.items():
            for i in ids:
                self.attached.setdefault(i, []).append(e)
        self.degree = {e: len(ids) for e, ids in about.items()}
        for i, ents in self.attached.items():  # most specific topic first: the one fewest memories share
            ents.sort(key=lambda e: (self.degree[e], e))
        # each memory's links, strongest kind first: a row lists a few, an error its causes and fixes
        self.links_of: Dict[str, List[Tuple[str, str, str]]] = {}
        rank = {rel: n for n, rel in enumerate(ROW_LINKS)}
        for link in sorted(self.memory_links, key=lambda x: (rank.get(x[1], 99), x)):
            for end in (link[0], link[2]):
                self.links_of.setdefault(end, []).append(link)
        self.covered_by: Dict[str, List[str]] = {}
        for s, ids in covers.items():
            for i in ids:
                self.covered_by.setdefault(i, []).append(s)

        self.topics: Dict[str, Topic] = {}
        linked = {e for s, _, t in self.entity_links for e in (s, t)}
        for e in sorted(self.entities):
            if e in about or (e in linked and any(o in about for o in self._entity_neighbours(e))):
                self.topics[e] = self._entity_topic(e, about.get(e, set()))
        for s in sorted(covers):
            m = self.memories[s]
            if _enum(m.category) == "summary":
                facts = [ln.strip().lstrip("-• ").strip() for ln in (m.content or "").splitlines() if ln.strip()]
                self.topics[s] = Topic(
                    s, "summary", "summary", label(m), facts[0] if facts else "", "agent", self._order(covers[s])
                )
        projects = sorted({self.memories[i].project_id for i in self.live})
        for pid in projects:
            ids = {i for i in self.live if self.memories[i].project_id == pid}
            self.topics[PROJECT_PREFIX + pid] = self._project_topic(pid, ids)
        self.related_cache: Dict[str, List[Dict[str, Any]]] = {}

    # --- building ----------------------------------------------------------

    def _current(self, memory_id: str) -> str:
        seen = set()
        while memory_id in self.memories and self.memories[memory_id].superseded_by and memory_id not in seen:
            seen.add(memory_id)
            memory_id = self.memories[memory_id].superseded_by or memory_id
        return memory_id

    def _order(self, ids: Iterable[str]) -> Tuple[str, ...]:
        return tuple(sorted(ids, key=lambda i: _newest_first(self.memories[i]), reverse=True))

    def _entity_neighbours(self, e: str) -> List[str]:
        return [t if s == e else s for s, _, t in self.entity_links if e in (s, t)]

    def _entity_topic(self, e: str, ids: Set[str]) -> Topic:
        entity = self.entities[e]
        etype = str(_enum(entity.type))
        written = str((entity.properties or {}).get(DESCRIPTION_KEY) or "").strip()
        if written:
            return Topic(e, "entity", etype, entity.name, written, "agent", self._order(ids))
        links: List[Any] = []
        for s, rel, t in self.entity_links[:]:
            if e in (s, t) and len(links) < 3:
                other = self.entities[t if s == e else s].name
                links.append(f"{phrase(rel, outgoing=s == e)} {other}")
        article = "an" if etype[:1] in "aeiou" else "a"
        text = f"{entity.name} is {article} {etype}"
        text += f" that {_and(links)}." if links else "."
        if not ids:
            text += " No memory is about it yet; it is known through its links."
        else:
            counts: Dict[str, int] = {}
            for i in ids:
                word = PLURAL_WORD.get(str(_enum(self.memories[i].category)), "notes")
                counts[word] = counts.get(word, 0) + 1
            top = [w for w, _ in sorted(counts.items(), key=lambda x: (-x[1], x[0]))[:2]]
            text += f" {_counted(len(ids), 'memory')} {'is' if len(ids) == 1 else 'are'} about it, mostly {_and(top)}."
        return Topic(e, "entity", etype, entity.name, text, "collivind", self._order(ids))

    def _project_topic(self, pid: str, ids: Set[str]) -> Topic:
        meta = self.metas.get(pid, {})
        title = str(meta.get("display_name") or pid)
        written = str(meta.get("description") or "").strip()
        if written:
            return Topic(PROJECT_PREFIX + pid, "project", "project", title, written, "agent", self._order(ids))
        topics = {e for i in ids for e in self.attached.get(i, ())}
        text = f"Project {title}: {_counted(len(ids), 'memory')} about {_counted(len(topics), 'topic')}."
        return Topic(PROJECT_PREFIX + pid, "project", "project", title, text, "collivind", self._order(ids))

    # --- reading -----------------------------------------------------------

    def stats(self, topic: Topic, ids: Optional[Tuple[str, ...]] = None) -> str:
        """'12 memories: 3 decisions, 2 problems (1 with a fix) ... · 2 Oct 2026 to 25 Nov 2026'."""
        ids = topic.members if ids is None else ids
        if not ids:
            return "No memories yet."
        counts: Dict[str, int] = {}
        for i in ids:
            cat = str(_enum(self.memories[i].category))
            counts[cat] = counts.get(cat, 0) + 1
        fixed = sum(1 for i in ids if str(_enum(self.memories[i].category)) == "error" and self._fixes(i))
        parts = []
        for cat in ("error", "decision", "architecture", "pattern", "fact", "preference", "snippet", "summary"):
            if counts.get(cat):
                word = KIND_WORD[cat].lower()
                part = _counted(counts[cat], "problem" if cat == "error" else word)
                if cat == "architecture":
                    part = f"{counts[cat]} on how it works"
                if cat == "error":
                    part += f" ({fixed} with a recorded fix)"
                parts.append(part)
        dates = sorted(d for d in (_date(self.memories[i]) for i in ids) if d)
        span = (
            f" · {dates[0]} to {dates[-1]}" if dates and dates[0] != dates[-1] else (f" · {dates[0]}" if dates else "")
        )
        return f"{_counted(len(ids), 'memory')}: {', '.join(parts)}{span}"

    def _fixes(self, error_id: str) -> List[Tuple[str, str]]:
        return [
            (s, self.notes.get((s, r, t), ""))
            for s, r, t in self.links_of.get(error_id, ())
            if r == "RESOLVES" and t == error_id and s in self.live
        ]

    def _causes(self, error_id: str) -> List[Tuple[str, str]]:
        return [
            (t, self.notes.get((s, r, t), ""))
            for s, r, t in self.links_of.get(error_id, ())
            if r == "CAUSED_BY" and s == error_id and t in self.live
        ]

    def topic_ref(self, topic_id: str) -> Dict[str, Any]:
        t = self.topics[topic_id]
        return {"id": t.id, "kind": t.kind, "type": t.type, "title": t.title}

    def memory_row(self, memory_id: str, here: Optional[str] = None) -> Dict[str, Any]:
        """A memory as a reader meets it: what kind, what it says in a line, what it is attached to."""
        m = self.memories[memory_id]
        cat = str(_enum(m.category))
        attached = [self.topic_ref(e) for e in self.attached.get(memory_id, []) if e != here and e in self.topics]
        summaries = [self.topic_ref(s) for s in self.covered_by.get(memory_id, []) if s != here and s in self.topics]
        links: List[Any] = []
        for s, rel, t in self.links_of.get(memory_id, ()):
            if rel not in ROW_LINKS or len(links) >= MAX_ROW_LINKS:
                continue
            other = t if s == memory_id else s
            o = self.memories[other]
            links.append(
                {
                    "type": rel,
                    "phrase": phrase(rel, outgoing=s == memory_id),
                    "note": self.notes.get((s, rel, t), ""),
                    "memory": {
                        "id": other,
                        "label": label(o),
                        "category": str(_enum(o.category)),
                        "live": other in self.live,
                    },
                }
            )
        return {
            "id": m.id,
            "label": label(m),
            "gist": gist(m),
            "content": m.content,
            "category": cat,
            "kind": KIND_WORD.get(cat, cat),
            "date": _date(m),
            "project": m.project_id,
            "about": attached[:4],
            "more_about": max(0, len(attached) - 4),
            "summaries": summaries[:2],
            "links": links,
        }

    def _members(self, topic: Topic, within: Optional[str], query: str = "") -> Tuple[str, ...]:
        """The memories a page lists. A project is the biggest hub of all: its page lists its topics,
        every problem (what a reader most often comes for), and otherwise only what no topic files."""
        ids = topic.members
        if topic.kind == "project" and not within:
            ids = tuple(i for i in ids if not self.attached.get(i) or str(_enum(self.memories[i].category)) == "error")
        if within and within in self.topics:
            keep = set(self.topics[within].members)
            ids = tuple(i for i in ids if i in keep)
        words = query.lower().split()
        if words:
            ids = tuple(
                i
                for i in ids
                if all(w in f"{self.memories[i].summary} {self.memories[i].content}".lower() for w in words)
            )
        return ids

    def sections(
        self,
        topic: Topic,
        within: Optional[str] = None,
        only: Optional[str] = None,
        offset: int = 0,
        limit: int = PAGE_SIZE,
        query: str = "",
    ) -> List[Dict[str, Any]]:
        """Memories grouped by purpose; problems carry their causes and fixes, wherever those are filed."""
        ids = [i for i in self._members(topic, within, query) if str(_enum(self.memories[i].category)) != "summary"]
        in_pairs: Set[str] = set()
        problems = [i for i in ids if str(_enum(self.memories[i].category)) == "error"]
        for i in problems:
            in_pairs |= {f for f, _ in self._fixes(i)} | {c for c, _ in self._causes(i)}
        out = []
        for key, title, cats in SECTIONS:
            if only and key != only:
                continue
            mine = [i for i in ids if str(_enum(self.memories[i].category)) in cats]
            rows = [i for i in mine if key == "problems" or i not in in_pairs]
            if not mine:
                continue
            page = rows[offset : offset + limit]
            items = []
            for i in page:
                row = self.memory_row(i, topic.id)
                if key == "problems":
                    row["fixes"] = [{**self.memory_row(f, topic.id), "note": n} for f, n in self._fixes(i)]
                    row["causes"] = [{**self.memory_row(c, topic.id), "note": n} for c, n in self._causes(i)]
                items.append(row)
            out.append(
                {
                    "key": key,
                    "title": title,
                    "total": len(rows),
                    "offset": offset,
                    "items": items,
                    "more": max(0, len(rows) - offset - len(page)),
                    "with_problems": len(mine) - len(rows),  # shown beside the problems they settle or caused
                }
            )
        return out

    def related(self, topic_id: str) -> List[Dict[str, Any]]:
        """Topics this one links to, each with the reason in words, strongest first."""
        if topic_id in self.related_cache:
            return self.related_cache[topic_id]
        topic = self.topics[topic_id]
        out: List[Dict[str, Any]] = []
        seen = {topic_id}

        def add(other: str, reason: str, weight: float, count: int = 0) -> None:
            if other in self.topics and other not in seen:
                seen.add(other)
                out.append({**self.topic_ref(other), "reason": reason, "weight": weight, "count": count})

        members = set(topic.members)
        if topic.kind == "entity":
            for s, rel, t in self.entity_links:
                if topic_id in (s, t):
                    other = t if s == topic_id else s
                    note = self.notes.get((s, rel, t), "")
                    said = f"{topic.title} {phrase(rel, outgoing=s == topic_id)} {self.entities[other].name}"
                    both = len(members & set(self.topics[other].members)) if other in self.topics else 0
                    shared_note = f" ({_counted(both, 'shared memory')})" if both else ""
                    add(other, (f"{said}: {note}" if note else said) + shared_note, 1000.0, both)
        if topic.kind in ("entity", "summary"):
            shared: Dict[str, int] = {}
            for i in members:
                for e in self.attached.get(i, ()):
                    shared[e] = shared.get(e, 0) + 1
            floor = 1 if topic.kind == "summary" or len(members) < 6 else 2
            ranked = sorted(
                ((e, n) for e, n in shared.items() if e != topic_id and n >= floor),
                key=lambda x: (-x[1] / math.sqrt(self.degree.get(x[0], 1)), -x[1], x[0]),
            )
            for e, n in ranked[:RELATED_LIMIT]:
                reason = (
                    f"{'1 of its memories is' if n == 1 else f'{n} of its memories are'} about {self.topics[e].title}"
                    if topic.kind == "summary"
                    else f"{_counted(n, 'shared memory')}"
                )
                add(e, reason, float(n) / math.sqrt(self.degree.get(e, 1)), n)
        if topic.kind != "summary":
            cover: Dict[str, int] = {}
            for i in members:
                for s in self.covered_by.get(i, ()):
                    cover[s] = cover.get(s, 0) + 1
            for s, n in sorted(cover.items(), key=lambda x: (-x[1], x[0]))[:6]:
                add(s, f"its summary covers {n} of these", float(n), n)
        if topic.kind == "project":
            counts: Dict[str, int] = {}
            for i in members:
                for e in self.attached.get(i, ()):
                    counts[e] = counts.get(e, 0) + 1
            for e, n in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
                add(e, f"{_counted(n, 'memory')} in this project", float(n), n)
        else:
            per: Dict[str, int] = {}
            for i in members:
                per[self.memories[i].project_id] = per.get(self.memories[i].project_id, 0) + 1
            for pid, n in sorted(per.items(), key=lambda x: (-x[1], x[0])):
                add(PROJECT_PREFIX + pid, f"{n} of these are in this project", 0.0, n)
        self.related_cache[topic_id] = out
        return out

    def backlinks(self, topic_id: str) -> List[Dict[str, Any]]:
        """Pages whose related list names this one (computed over every page: a few thousand rows)."""
        return [
            self.topic_ref(t)
            for t in sorted(self.topics)
            if t != topic_id and any(r["id"] == topic_id for r in self.related(t))
        ]

    def page(
        self,
        topic_id: str,
        within: Optional[str] = None,
        only: Optional[str] = None,
        offset: int = 0,
        limit: int = PAGE_SIZE,
        query: str = "",
    ) -> Optional[Dict[str, Any]]:
        topic = self.topics.get(topic_id)
        if topic is None:
            return None
        body: Dict[str, Any] = {
            **self.topic_ref(topic_id),
            "description": topic.description,
            "described_by": topic.described_by,
            "stats": self.stats(topic),
            "hub": len(topic.members) > HUB_DEGREE,
            "within": self.topic_ref(within) if within in self.topics else None,
            "query": query,
            "sections": self.sections(topic, within, only, offset, limit, query),
        }
        if only:
            return body
        if topic.kind == "summary":
            body["facts"] = [
                ln.strip().lstrip("-• ").strip()
                for ln in (self.memories[topic_id].content or "").splitlines()
                if ln.strip()
            ]
        if topic.kind == "project":
            body["unfiled"] = sum(1 for i in topic.members if not self.attached.get(i))
        body["related"] = self.related(topic_id)
        body["backlinks"] = self.backlinks(topic_id)
        return body

    def index(self) -> Dict[str, Any]:
        """Every topic with its count, and the links the topic map draws (typed, else the strongest shared)."""
        topics = [
            {
                **self.topic_ref(t.id),
                "count": len(t.members),
                "description": t.description,
                "described_by": t.described_by,
            }
            for t in self.topics.values()
        ]
        links: List[Any] = []
        seen = set()
        for t in self.topics:
            strong = [r for r in self.related(t) if r["weight"] > 0][:MAP_LINKS]
            for r in strong:
                key = tuple(sorted((t, r["id"])))
                if key not in seen:
                    seen.add(key)
                    links.append({"source": t, "target": r["id"], "reason": r["reason"]})
        return {"topics": topics, "links": links, "project": self.project}
