"""Project metadata, kept in the graph store itself.

Each project's metadata is one `project` entity whose id is `project:` plus
the exact project id. The id is written verbatim through restore_entity, not
derived from the name, so `my-app` and `my_app` stay two projects and an
entity that extraction created for the same name (id `my_app`) is never
overwritten. Living in the store means export, import and migrate carry it
with no side file.
"""

import re
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from collivind.models.entity import EntityNode, EntityType
from collivind.storage.interfaces import GraphStore

META_PREFIX = "project:"
MAX_PROJECT_ID = 200
_COLOR = re.compile(r"^#[0-9a-fA-F]{6}$")
# field -> max length; tags are checked separately
_TEXT_FIELDS = {"display_name": 200, "description": 4000, "repo": 1000}
FIELDS = (*_TEXT_FIELDS, "color", "tags")


def meta_entity_id(project_id: str) -> str:
    return META_PREFIX + project_id


def is_meta_entity(entity: EntityNode) -> bool:
    return entity.id.startswith(META_PREFIX) and EntityType(entity.type) == EntityType.PROJECT


def validate_project_id(project_id: Any) -> str:
    if not isinstance(project_id, str) or not project_id.strip():
        raise ValueError("project id must be a non-empty string")
    if project_id == "*":
        raise ValueError('"*" means every project and cannot be used as a project id')
    if len(project_id) > MAX_PROJECT_ID:
        raise ValueError(f"project id is longer than {MAX_PROJECT_ID} characters")
    return project_id


def _validate_repo(value: str) -> str:
    """An http(s) URL or a filesystem path. The UI renders URLs as links, so
    any other scheme (javascript:, data:) is refused here, not escaped later."""
    if not value:
        return value
    scheme = urlparse(value).scheme.lower()
    if scheme in ("http", "https"):
        return value
    if scheme and len(scheme) > 1:  # "c" in C:\\repo is a drive letter, not a scheme
        raise ValueError("repo must be an http(s) URL or a filesystem path")
    return value


def validate_meta(body: Any) -> Dict[str, Any]:
    """The editable fields of a project, validated. Raises ValueError."""
    if not isinstance(body, dict):
        raise ValueError("project metadata must be an object")
    unknown = sorted(set(body) - set(FIELDS))
    if unknown:
        raise ValueError(f"unknown project fields: {', '.join(unknown)}")
    meta: Dict[str, Any] = {}
    for key, limit in _TEXT_FIELDS.items():
        if key in body:
            value = body[key]
            if not isinstance(value, str) or len(value) > limit:
                raise ValueError(f"{key} must be a string of at most {limit} characters")
            meta[key] = value.strip()
    if "repo" in meta:
        meta["repo"] = _validate_repo(meta["repo"])
    if "color" in body:
        color = body["color"]
        if not isinstance(color, str) or (color and not _COLOR.match(color)):
            raise ValueError("color must be #rrggbb or empty")
        meta["color"] = color.lower()
    if "tags" in body:
        tags = body["tags"]
        if not isinstance(tags, list) or not all(isinstance(t, str) and 0 < len(t.strip()) <= 100 for t in tags):
            raise ValueError("tags must be a list of non-empty strings")
        if len(tags) > 50:
            raise ValueError("at most 50 tags")
        meta["tags"] = list(dict.fromkeys(t.strip() for t in tags))
    return meta


def _meta_of(entity: Optional[EntityNode]) -> Dict[str, Any]:
    props = entity.properties if entity else {}
    return {key: props.get(key, [] if key == "tags" else "") for key in FIELDS}


def _meta_entities(graph: GraphStore) -> Dict[str, EntityNode]:
    return {e.id[len(META_PREFIX) :]: e for e in graph.iter_entities() if is_meta_entity(e)}


def list_projects(graph: GraphStore) -> List[Dict[str, Any]]:
    """Every project that has memories or metadata, with counts, busiest first."""
    total: Counter = Counter()
    live: Counter = Counter()
    for m in graph.iter_memories():
        total[m.project_id] += 1
        if not m.superseded_by and not m.valid_to:
            live[m.project_id] += 1
    metas = _meta_entities(graph)
    ids = set(total) | set(metas)
    rows = [{"project_id": pid, "memories": total[pid], "live": live[pid], **_meta_of(metas.get(pid))} for pid in ids]
    return sorted(rows, key=lambda r: (-r["live"], r["project_id"]))


def set_project_meta(graph: GraphStore, project_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
    """Merge validated `updates` into the project's metadata entity and return the new metadata."""
    project_id = validate_project_id(project_id)
    updates = validate_meta(updates)
    existing = _meta_entities(graph).get(project_id)
    now = datetime.now(timezone.utc)
    props = {**(existing.properties if existing else {}), **updates, "project_id": project_id}
    graph.restore_entity(
        EntityNode(
            id=meta_entity_id(project_id),
            name=project_id,
            type=EntityType.PROJECT,
            properties=props,
            created_at=existing.created_at if existing else now,
            updated_at=now,
        )
    )
    return {
        "project_id": project_id,
        **_meta_of(EntityNode(name=project_id, type=EntityType.PROJECT, properties=props)),
    }
