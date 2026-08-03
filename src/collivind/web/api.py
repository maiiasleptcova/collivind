"""JSON API over MemoryManager.

Split from the HTTP plumbing so the request handling is testable without
binding a socket. Every handler returns (status, payload).
"""

from typing import Any, Dict, Tuple

from collivind.models import MemoryCategory, MemoryCreate, SearchQuery

Result = Tuple[int, Dict[str, Any]]


def _is_tag_list(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(t, str) for t in value)


def _memory_dict(node) -> Dict[str, Any]:
    d = node.to_dict()
    cat = d.get("category")
    d["category"] = cat.value if hasattr(cat, "value") else cat
    return d


class MemoryAPI:
    """The operations the UI needs, over an existing MemoryManager."""

    def __init__(self, manager, mode: str = "unknown"):
        self.manager = manager
        self.mode = mode

    # --- read -------------------------------------------------------------

    def status(self) -> Result:
        return 200, {"mode": self.mode}

    def list_memories(self, project: str = "default", query: str = "", limit: int = 50) -> Result:
        """Search when given a query, otherwise the timeline.

        Searching costs an embedding round-trip; browsing must not, so an
        empty query deliberately takes the cheaper path.
        """
        if query.strip():
            results = self.manager.search(SearchQuery(query=query, project_id=project, limit=limit))
            return 200, {
                "memories": [{**_memory_dict(r.memory), "score": r.score} for r in results],
                "searched": True,
            }
        nodes = self.manager.get_timeline(project, limit=limit)
        return 200, {"memories": [_memory_dict(n) for n in nodes], "searched": False}

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
                    project_id=body.get("project_id") or "default",
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

    def forget(self, memory_id: str) -> Result:
        ok = self.manager.forget(memory_id)
        if not ok:
            return 404, {"error": f"No memory {memory_id}"}
        return 200, {"forgotten": memory_id}


def route(api: MemoryAPI, method: str, path: str, params: Dict[str, str], body: Dict[str, Any]) -> Result:
    """Map a request onto MemoryAPI. Returns (status, payload)."""
    parts = [p for p in path.strip("/").split("/") if p]
    if not parts or parts[0] != "api":
        return 404, {"error": "not found"}
    parts = parts[1:]

    if parts == ["status"] and method == "GET":
        return api.status()

    if parts == ["memories"]:
        if method == "GET":
            try:
                limit = int(params.get("limit", 50))
            except ValueError:
                return 400, {"error": "limit must be an integer"}
            return api.list_memories(params.get("project", "default"), params.get("q", ""), limit)
        if method == "POST":
            return api.create(body)

    if len(parts) == 2 and parts[0] == "memories":
        if method == "PATCH":
            return api.update(parts[1], body)
        if method == "DELETE":
            return api.forget(parts[1])

    if len(parts) == 3 and parts[0] == "memories" and parts[2] == "versions" and method == "GET":
        return api.get_versions(parts[1])

    if len(parts) == 2 and parts[0] == "entities" and method == "GET":
        return api.get_entity(parts[1])

    return 404, {"error": f"no route for {method} {path}"}
