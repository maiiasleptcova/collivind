"""JSON API over MemoryManager.

Split from the HTTP plumbing so the request handling is testable without
binding a socket. Every handler returns (status, payload).
"""

from typing import Any, Dict, Optional, Tuple

from collivind.engine.memory_manager import LINKABLE
from collivind.engine.projects import list_projects, set_project_meta, validate_project_id
from collivind.models import MemoryCategory, MemoryCreate, RelType, SearchQuery
from collivind.models.entity import EntityType
from collivind.models.relations import MEANINGS
from collivind.web.graph import DEFAULT_LIMIT, MAX_LIMIT, build_graph

Result = Tuple[int, Dict[str, Any]]


def _is_tag_list(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(t, str) for t in value)


def _memory_dict(node) -> Dict[str, Any]:
    d = node.to_dict()
    cat = d.get("category")
    d["category"] = cat.value if hasattr(cat, "value") else cat
    return d


def _memory_dict_value(m, field: str) -> Any:
    value = getattr(m, "project_id" if field == "project" else field)
    return value.value if hasattr(value, "value") else value


MAX_ID = 512
ALL_PROJECTS = "*"
MAX_QUERY = 2000
MAX_SEEDS = 200
MAX_TOPIC_MEMORIES = 200
SORTS = {
    "newest": lambda m: (-_ts(m),),
    "oldest": lambda m: (_ts(m),),
    "confidence": lambda m: (-(m.confidence or 0.0), -_ts(m)),
}
GROUPS = {"", "project", "category"}


def _ts(m) -> float:
    return m.created_at.timestamp() if m.created_at else 0.0


def _csv(params: Dict[str, str], key: str, cap: int) -> list:
    values = [v for v in (params.get(key) or "").split(",") if v]
    if len(values) > cap:
        raise ValueError(f"{key} takes at most {cap} values")
    return values


def _id(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or len(value) > MAX_ID:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _rel_type(value: Any) -> RelType:
    try:
        return RelType(value)
    except ValueError:
        raise ValueError(f"unknown relationship type {value!r}") from None


def _int_param(params: Dict[str, str], key: str, default: int, low: int, high: int) -> int:
    try:
        value = int(params.get(key, default))
    except ValueError:
        raise ValueError(f"{key} must be an integer") from None
    if not low <= value <= high:
        raise ValueError(f"{key} must be between {low} and {high}")
    return value


class MemoryAPI:
    """The operations the UI needs, over an existing MemoryManager."""

    def __init__(self, manager, mode: str = "unknown"):
        self.manager = manager
        self.mode = mode

    # --- read -------------------------------------------------------------

    def status(self) -> Result:
        return 200, {
            "mode": self.mode,
            "categories": [c.value for c in MemoryCategory],
            "linkable": {f"{a}:{b}": list(types) for (a, b), types in LINKABLE.items()},
            "meanings": {t: m._asdict() for t, m in MEANINGS.items()},
        }

    def get_memory(self, memory_id: str) -> Result:
        node = self.manager.get_memory(memory_id)
        if node is None:
            return 404, {"error": f"No memory {memory_id}"}
        return 200, _memory_dict(node)

    def graph(
        self,
        project: Optional[str],
        limit: int,
        types: Optional[set] = None,
        seeds: Optional[set] = None,
        depth: int = 1,
    ) -> Result:
        return 200, build_graph(
            self.manager.graph_store, project=project, limit=limit, types=types, seeds=seeds, depth=depth
        )

    def projects(self) -> Result:
        return 200, {"projects": list_projects(self.manager.graph_store)}

    def update_project(self, project_id: str, body: Dict[str, Any]) -> Result:
        try:
            return 200, set_project_meta(self.manager.graph_store, project_id, body)
        except ValueError as e:
            return 400, {"error": str(e)}

    def list_memories(
        self,
        project: str = "default",
        query: str = "",
        limit: int = 50,
        offset: int = 0,
        sort: str = "newest",
        group: str = "",
    ) -> Result:
        """Search when given a query, otherwise a page of current memories.

        Searching costs an embedding round-trip; browsing must not, so an
        empty query deliberately takes the cheaper path.
        """
        every = project == ALL_PROJECTS
        if query.strip():
            # An empty project_id is "no project filter" to the search engine.
            results = self.manager.search(SearchQuery(query=query, project_id="" if every else project, limit=limit))
            return 200, {
                "memories": [{**_memory_dict(r.memory), "score": r.score} for r in results],
                "searched": True,
            }
        # ponytail: full scan of the graph store per page; the timeline query
        # has no offset or sort, and stores of a few thousand scan in ms.
        # A paged store query if stores grow past ~100k.
        live = [
            m
            for m in self.manager.graph_store.iter_memories()
            if (every or m.project_id == project) and not m.superseded_by and not m.valid_to
        ]
        order = SORTS[sort]
        if group:
            live.sort(key=lambda m: (str(_memory_dict_value(m, group)), *order(m)))
        else:
            live.sort(key=order)
        page = live[offset : offset + limit]
        memories = [_memory_dict(n) for n in page]
        return 200, {"memories": memories, "searched": False, "total": len(live), "offset": offset}

    def get_versions(self, memory_id: str) -> Result:
        chain = self.manager.get_version_chain(memory_id)
        return 200, {"versions": [_memory_dict(n) for n in chain]}

    def get_entity(self, name: str) -> Result:
        entity = self.manager.get_entity(name)
        if entity is None:
            return 404, {"error": f"No entity named {name!r}"}
        return 200, entity

    # --- write ------------------------------------------------------------

    def create(self, body: Dict[str, Any]) -> Result:
        if not isinstance(body.get("content", ""), str) or not isinstance(body.get("summary", ""), str):
            return 400, {"error": "content and summary must be strings"}
        content = (body.get("content") or "").strip()
        if not content:
            return 400, {"error": "content is required"}
        try:
            category = MemoryCategory(body.get("category") or "fact")
        except ValueError:
            return 400, {"error": f"unknown category {body.get('category')!r}"}
        # create was the open half of the same guard update already had: a
        # non-list here reaches SQLite as a JSON object, comes back as a dict,
        # and throws in the UI — taking the whole list render down with it.
        if "tags" in body and not _is_tag_list(body["tags"]):
            return 400, {"error": "tags must be a list of strings"}

        try:
            node = self.manager.add_memory(
                MemoryCreate(
                    content=content,
                    summary=(body.get("summary") or content[:120]).strip(),
                    category=category,
                    project_id=validate_project_id(body.get("project_id") or "default"),
                    tags=body.get("tags") or [],
                    confidence=body.get("confidence", 1.0),
                )
            )
        except ValueError as e:
            return 400, {"error": str(e)}
        return 201, _memory_dict(node)

    def update(self, memory_id: str, body: Dict[str, Any]) -> Result:
        fields = {k: body[k] for k in ("content", "summary", "tags", "confidence") if k in body}

        # Category is the one editable field with a closed vocabulary, and a
        # bad value would reach the store as a string no reader recognises.
        if "category" in body:
            try:
                fields["category"] = MemoryCategory(body["category"])
            except ValueError:
                return 400, {"error": f"unknown category {body['category']!r}"}

        if "tags" in fields and not _is_tag_list(fields["tags"]):
            return 400, {"error": "tags must be a list of strings"}

        # Untyped values reach the sqlite driver and surface as a 500 carrying
        # its internal message; the engine rejects blanks, this rejects shapes.
        for key in ("content", "summary"):
            if key in fields and not isinstance(fields[key], str):
                return 400, {"error": f"{key} must be a string"}

        if "confidence" in fields:
            # bool is an int subclass, so True would sail through float() as 1.0.
            if isinstance(fields["confidence"], bool):
                return 400, {"error": "confidence must be a number between 0 and 1"}
            try:
                confidence = float(fields["confidence"])
            except (TypeError, ValueError):
                return 400, {"error": "confidence must be a number between 0 and 1"}
            if not 0.0 <= confidence <= 1.0:  # also rejects NaN, which compares False
                return 400, {"error": "confidence must be between 0 and 1"}
            fields["confidence"] = confidence

        if not fields:
            return 400, {"error": "nothing to update"}
        try:
            node = self.manager.update_memory(memory_id, **fields)
        except ValueError as e:
            # The engine validates too; a drift between the two lists must not
            # turn a bad request into a 500.
            return 400, {"error": str(e)}
        if node is None:
            return 404, {"error": f"No memory {memory_id}"}
        return 200, _memory_dict(node)

    def _link_args(self, body: Dict[str, Any]) -> Tuple[str, str, RelType]:
        return (
            _id(body.get("source_id"), "source_id"),
            _id(body.get("target_id"), "target_id"),
            _rel_type(body.get("type")),
        )

    def link(self, body: Dict[str, Any]) -> Result:
        try:
            source, target, rel = self._link_args(body)
            self.manager.link(source, target, rel, note=body.get("note") or "")
        except LookupError as e:
            return 404, {"error": str(e.args[0])}
        except ValueError as e:
            return 400, {"error": str(e)}
        return 201, {"source": source, "target": target, "type": rel.value}

    def link_topic(self, body: Dict[str, Any]) -> Result:
        ids = body.get("memory_ids")
        if not isinstance(ids, list) or not ids or len(ids) > MAX_TOPIC_MEMORIES:
            return 400, {"error": f"memory_ids must be a list of 1 to {MAX_TOPIC_MEMORIES} ids"}
        name = body.get("name")
        if not isinstance(name, str) or not name.strip() or len(name) > 200:
            return 400, {"error": "name must be 1 to 200 characters"}
        try:
            entity_type = EntityType(body.get("type") or "concept")
            result = self.manager.link_topic(name, entity_type, [_id(i, "memory id") for i in ids])
        except LookupError as e:
            return 404, {"error": str(e.args[0])}
        except ValueError as e:
            return 400, {"error": str(e)}
        return 201, result

    def unlink(self, body: Dict[str, Any]) -> Result:
        try:
            source, target, rel = self._link_args(body)
            removed = self.manager.unlink(source, target, rel)
        except ValueError as e:
            return 400, {"error": str(e)}
        if not removed:
            return 404, {"error": "no such link"}
        return 200, {"removed": removed}

    def forget(self, memory_id: str) -> Result:
        ok = self.manager.forget(memory_id)
        if not ok:
            return 404, {"error": f"No memory {memory_id}"}
        return 200, {"forgotten": memory_id}


def route(api: MemoryAPI, method: str, path: str, params: Dict[str, str], body: Any) -> Result:
    """Map a request onto MemoryAPI. Returns (status, payload).

    Callers pass the decoded path; ids with a slash (file entities) travel in
    the query string or body, never as a path segment.
    """
    if not isinstance(body, dict):
        return 400, {"error": "body must be a JSON object"}
    parts = [p for p in path.strip("/").split("/") if p]
    if not parts or parts[0] != "api":
        return 404, {"error": "not found"}
    parts = parts[1:]
    try:
        return _dispatch(api, method, parts, params, body)
    except ValueError as e:
        return 400, {"error": str(e)}


def _dispatch(api: MemoryAPI, method: str, parts: list, params: Dict[str, str], body: Dict[str, Any]) -> Result:
    if parts == ["status"] and method == "GET":
        return api.status()

    if parts == ["graph"] and method == "GET":
        project = params.get("project") or None
        if project is not None:
            validate_project_id(project)
        types = _csv(params, "types", len(RelType))
        for t in types:
            _rel_type(t)
        seeds = _csv(params, "seeds", MAX_SEEDS)
        for seed in seeds:
            _id(seed, "seed")
        return api.graph(
            project,
            _int_param(params, "limit", DEFAULT_LIMIT, 1, MAX_LIMIT),
            types=set(types) if "types" in params else None,
            seeds=set(seeds) or None,
            depth=_int_param(params, "depth", 1, 1, 3),
        )

    if parts == ["projects"] and method == "GET":
        return api.projects()

    if parts == ["project"] and method == "PATCH":
        return api.update_project(params.get("id", ""), body)

    if parts == ["links"]:
        if method == "POST":
            return api.link(body)
        if method == "DELETE":
            return api.unlink(body)

    if parts == ["topics"] and method == "POST":
        return api.link_topic(body)

    if parts == ["memories"]:
        if method == "GET":
            query = params.get("q", "")
            if len(query) > MAX_QUERY:
                return 400, {"error": f"query is longer than {MAX_QUERY} characters"}
            project = params.get("project", "default")
            if project != ALL_PROJECTS:
                validate_project_id(project)
            sort = params.get("sort", "newest")
            if sort not in SORTS:
                return 400, {"error": f"sort must be one of {', '.join(SORTS)}"}
            group = params.get("group", "")
            if group not in GROUPS:
                return 400, {"error": "group must be project or category"}
            return api.list_memories(
                project,
                query,
                _int_param(params, "limit", 50, 1, 500),
                offset=_int_param(params, "offset", 0, 0, 1_000_000),
                sort=sort,
                group=group,
            )
        if method == "POST":
            return api.create(body)

    if len(parts) == 2 and parts[0] == "memories":
        if method == "GET":
            return api.get_memory(parts[1])
        if method == "PATCH":
            return api.update(parts[1], body)
        if method == "DELETE":
            return api.forget(parts[1])

    if len(parts) == 3 and parts[0] == "memories" and parts[2] == "versions" and method == "GET":
        return api.get_versions(parts[1])

    if parts == ["entity"] and method == "GET":
        return api.get_entity(_id(params.get("name"), "name"))

    if len(parts) == 2 and parts[0] == "entities" and method == "GET":
        return api.get_entity(parts[1])

    return 404, {"error": f"no route for {method} /api/{'/'.join(parts)}"}
