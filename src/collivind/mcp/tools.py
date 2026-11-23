import json
from typing import Any, Dict, List, Optional

from collivind import capture
from collivind.engine.conflicts import CONFLICT_INSTRUCTION
from collivind.engine.extractor import build_extraction_prompt, parse_extraction_response
from collivind.engine.memory_manager import DEGRADED_NOTE, MemoryManager
from collivind.exceptions import BatchPartiallyStoredError, CollivindError
from collivind.models import (
    EntityCreate,
    EntityType,
    MemoryCategory,
    MemoryCreate,
    RelationshipCreate,
    RelType,
    SearchQuery,
    SearchResult,
)
from collivind.storage.interfaces import EmbeddingProvider, GraphStore, VectorStore

BACKEND_HINT = (
    "A storage backend is failing. Run `collivind status` (or the collivind_status tool) "
    "to see which one; in docker mode check `docker ps` for a restarting container."
)


def _conflict_view(conflicts: List[SearchResult]) -> List[Dict[str, Any]]:
    return [
        {
            "id": c.memory.id,
            "summary": c.memory.summary,
            "content": c.memory.content[:400],
            "created_at": c.memory.created_at.isoformat() if c.memory.created_at else None,
            "similarity": round(c.vector_score, 3),
        }
        for c in conflicts
    ]


def _with_conflicts(result: Dict[str, Any], conflicts: Dict[str, List[SearchResult]]) -> Dict[str, Any]:
    """Attach conflict candidates, keyed by the new memory's id, plus what to do about them."""
    if not conflicts:
        return result
    return {
        **result,
        "possible_conflicts": {mid: _conflict_view(found) for mid, found in conflicts.items()},
        "conflict_instruction": CONFLICT_INSTRUCTION,
    }


class CollivindTools:
    def __init__(self, memory_manager: MemoryManager, session_id: Optional[str] = None):
        self.memory_manager = memory_manager
        self.session_id = session_id

    def _backend_health(self) -> Dict[str, Any]:
        backends: Dict[str, VectorStore | GraphStore | EmbeddingProvider] = {
            "vector_store": self.memory_manager.vector_store,
            "graph_store": self.memory_manager.graph_store,
            "embedding_provider": self.memory_manager.embedding_provider,
        }
        status: Dict[str, Any] = {"mode": self.memory_manager.config.mode}
        for name, backend in backends.items():
            try:
                status[name] = backend.health_check()
            except Exception as e:
                status[name] = {"status": "error", "message": str(e)}
        return status

    def _batch_add(self, args: Dict[str, Any]) -> str:
        """Store a batch and record the outcome against its checkpoint (#35).

        The hook that asked for the save never sees this call, so the
        checkpoint history is the only place "a save actually landed" exists.
        """
        checkpoint_id = args.get("checkpoint_id") or None
        memories = []
        for mem in args.get("memories") or []:
            normalized = dict(mem)
            if self.session_id:
                normalized.setdefault("session_id", self.session_id)
            memories.append(normalized)
        try:
            ids, conflicts = self.memory_manager.batch_add_checked(memories) if memories else ([], {})
        except BatchPartiallyStoredError as e:
            self._record(checkpoint_id, False, e.stored_ids, error=str(e), failed_index=e.failed_index)
            raise
        except Exception as e:
            self._record(checkpoint_id, False, [], error=str(e))
            raise
        self._record(checkpoint_id, True, ids)
        result: Dict[str, Any] = {"status": "success", "memory_ids": ids}
        if checkpoint_id:
            result["checkpoint_id"] = checkpoint_id
            result["checkpoint"] = "confirmed" if ids else "checked, nothing new"
        return json.dumps(_with_conflicts(result, conflicts))

    def _record(self, checkpoint_id, ok, ids, **kw) -> None:
        try:
            capture.record(checkpoint_id, ok, ids, agent_session=self.session_id or "", **kw)
        except OSError:
            pass  # history is diagnostics; never fail a write because of it

    @staticmethod
    def get_tool_list() -> List[Dict[str, Any]]:
        return [
            {
                "name": "collivind_status",
                "description": "Check health status of Collivind services.",
                "inputSchema": {"type": "object", "properties": {}},
            },
            {
                "name": "collivind_add_memory",
                "description": (
                    "Store a single extracted fact, decision, pattern, or error. If the response has "
                    "possible_conflicts, follow its conflict_instruction: retire the stale or wrong memory "
                    "with collivind_invalidate, or ask the user when you cannot tell which is right."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string"},
                        "summary": {"type": "string"},
                        "category": {"type": "string", "enum": [c.value for c in MemoryCategory]},
                        "project_id": {"type": "string", "default": "default"},
                        "entities": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "type": {"type": "string", "enum": [t.value for t in EntityType]},
                                    "properties": {"type": "object"},
                                },
                                "required": ["name", "type"],
                            },
                        },
                        "relationships": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "target_id": {"type": "string"},
                                    "type": {"type": "string", "enum": [r.value for r in RelType]},
                                    "properties": {"type": "object"},
                                },
                                "required": ["target_id", "type"],
                            },
                        },
                        "tags": {"type": "array", "items": {"type": "string"}},
                        "confidence": {"type": "number", "default": 1.0},
                    },
                    "required": ["content", "summary", "category"],
                },
            },
            {
                "name": "collivind_search",
                "description": "Semantic + graph hybrid search with optional filters and pagination.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "category": {"type": "string"},
                        "project_id": {"type": "string", "default": "default"},
                        "limit": {"type": "integer", "default": 10},
                        "offset": {"type": "integer", "default": 0},
                        "include_graph": {"type": "boolean", "default": True},
                        "tags": {"type": "array", "items": {"type": "string"}},
                        "entity_names": {"type": "array", "items": {"type": "string"}},
                        "session_id": {"type": "string", "description": "Filter by session"},
                        "user_id": {"type": "string", "description": "Filter by contributing user"},
                        "date_from": {"type": "string", "description": "ISO datetime"},
                        "date_to": {"type": "string", "description": "ISO datetime"},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "collivind_get_context",
                "description": "Get relevant context as formatted text.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "project_id": {"type": "string", "default": "default"},
                        "limit": {"type": "integer", "default": 10},
                        "max_tokens": {"type": "integer", "description": "Approximate token budget for the block"},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "collivind_mark_conflict",
                "description": (
                    "Flag two memories as genuinely contradicting each other when neither the source of truth "
                    "nor the user can settle it now. Draws a CONTRADICTS edge for later review; resolve it "
                    "afterwards with collivind_invalidate."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {"memory_id": {"type": "string"}, "other_id": {"type": "string"}},
                    "required": ["memory_id", "other_id"],
                },
            },
            {
                "name": "collivind_invalidate",
                "description": (
                    "Retire a memory that is outdated or wrong: it leaves search but stays in the version "
                    "history, with superseded_by pointing at the memory that replaces or corrects it. Use this, "
                    "not collivind_forget, to resolve a conflict."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "memory_id": {"type": "string"},
                        "superseded_by": {"type": "string"},
                        "reason": {"type": "string"},
                    },
                    "required": ["memory_id", "superseded_by", "reason"],
                },
            },
            {
                "name": "collivind_forget",
                "description": "Delete a specific memory permanently.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"memory_id": {"type": "string"}},
                    "required": ["memory_id"],
                },
            },
            {
                "name": "collivind_get_entity",
                "description": "Get an entity and its related memories.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"entity_name": {"type": "string"}},
                    "required": ["entity_name"],
                },
            },
            {
                "name": "collivind_get_timeline",
                "description": "Get a chronological timeline of memories for a project.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "project_id": {"type": "string", "default": "default"},
                        "entity_name": {"type": "string"},
                        "session_id": {"type": "string", "description": "Filter by session"},
                        "limit": {"type": "integer", "default": 50},
                        "offset": {"type": "integer", "default": 0},
                    },
                    "required": ["project_id"],
                },
            },
            {
                "name": "collivind_find_contradictions",
                "description": (
                    "List live memories close enough to a given memory to need checking against it (a newer "
                    "version, a real conflict, or just a neighbour). Similarity cannot tell these apart; read "
                    "them and follow conflict_instruction."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {"memory_id": {"type": "string", "description": "ID of the memory to check against"}},
                    "required": ["memory_id"],
                },
            },
            {
                "name": "collivind_batch_add",
                "description": (
                    "Add multiple memories in batch (all-or-nothing validation). possible_conflicts, when "
                    "present, maps each new memory id to existing memories to check; follow conflict_instruction."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "memories": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "content": {"type": "string"},
                                    "summary": {"type": "string"},
                                    "category": {"type": "string", "enum": [c.value for c in MemoryCategory]},
                                    "project_id": {"type": "string", "default": "default"},
                                    "entities": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "name": {"type": "string"},
                                                "type": {"type": "string", "enum": [t.value for t in EntityType]},
                                                "properties": {"type": "object"},
                                            },
                                            "required": ["name", "type"],
                                        },
                                    },
                                    "relationships": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "target_id": {"type": "string"},
                                                "type": {"type": "string", "enum": [r.value for r in RelType]},
                                                "properties": {"type": "object"},
                                            },
                                            "required": ["target_id", "type"],
                                        },
                                    },
                                    "tags": {"type": "array", "items": {"type": "string"}},
                                    "confidence": {"type": "number", "default": 1.0},
                                    "session_id": {
                                        "type": "string",
                                        "description": "Session that produced this memory",
                                    },
                                    "user_id": {"type": "string", "default": "local"},
                                    "source": {"type": "string", "default": "manual"},
                                },
                                "required": ["content", "summary", "category"],
                            },
                        },
                        "checkpoint_id": {
                            "type": "string",
                            "description": (
                                "Pass the checkpoint id from a <collivind_extraction> request so the save "
                                "is recorded as that checkpoint's confirmation. An empty memories list with "
                                "a checkpoint_id records 'checked, nothing new'."
                            ),
                        },
                    },
                    "required": ["memories"],
                },
            },
            {
                "name": "collivind_get_version_chain",
                "description": "Get the full version history of a memory.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"memory_id": {"type": "string"}},
                    "required": ["memory_id"],
                },
            },
            {
                "name": "collivind_extract",
                "description": (
                    "Extract structured memories from raw text using LLM. "
                    "Returns the extraction prompt — send it to the LLM, "
                    "then pass the response to collivind_extract_save."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "Raw text to extract from"},
                        "project_id": {"type": "string", "default": "default"},
                    },
                    "required": ["text"],
                },
            },
            {
                "name": "collivind_extract_save",
                "description": "Save LLM extraction results as memories.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "llm_response": {"type": "string", "description": "JSON from LLM extraction"},
                        "project_id": {"type": "string", "default": "default"},
                    },
                    "required": ["llm_response"],
                },
            },
        ]

    def handle_call(self, name: str, args: Dict[str, Any]) -> str:
        try:
            if name == "collivind_status":
                return json.dumps(self._backend_health(), indent=2)

            elif name == "collivind_add_memory":
                # Parse entities
                entities = []
                for e in args.get("entities", []):
                    entities.append(
                        EntityCreate(name=e["name"], type=EntityType(e["type"]), properties=e.get("properties", {}))
                    )

                # Parse relationships
                relationships = []
                for r in args.get("relationships", []):
                    relationships.append(
                        RelationshipCreate(
                            source_id="",  # Will be filled by memory_manager
                            target_id=r["target_id"],
                            type=RelType(r["type"]),
                            properties=r.get("properties", {}),
                        )
                    )

                mem_create = MemoryCreate(
                    content=args["content"],
                    summary=args["summary"],
                    category=MemoryCategory(args["category"]),
                    project_id=args.get("project_id", "default"),
                    confidence=args.get("confidence", 1.0),
                    tags=args.get("tags", []),
                    session_id=args.get("session_id", self.session_id),
                )

                memory, found = self.memory_manager.add_memory_checked(
                    mem_create, entities=entities, relationships=relationships
                )
                result: Dict[str, Any] = {"status": "success", "memory_id": memory.id}
                return json.dumps(_with_conflicts(result, {memory.id: found} if found else {}))

            elif name == "collivind_search":
                date_from = None
                date_to = None
                if args.get("date_from"):
                    from datetime import datetime

                    date_from = datetime.fromisoformat(args["date_from"])
                if args.get("date_to"):
                    from datetime import datetime

                    date_to = datetime.fromisoformat(args["date_to"])
                offset = args.get("offset", 0)
                fetch_limit = args.get("limit", 10) + offset
                q = SearchQuery(
                    query=args["query"],
                    category=args.get("category"),
                    project_id=args.get("project_id", "default"),
                    limit=fetch_limit,
                    include_graph=args.get("include_graph", True),
                    tags=args.get("tags"),
                    entity_names=args.get("entity_names"),
                    session_id=args.get("session_id"),
                    user_id=args.get("user_id"),
                    date_from=date_from,
                    date_to=date_to,
                )
                results, degraded = self.memory_manager.search_with_status(q)
                limit = args.get("limit", 10)
                page = results[offset : offset + limit]
                response = {"degraded": degraded, **({"warning": DEGRADED_NOTE} if degraded else {})}
                return json.dumps(
                    {
                        **response,
                        "results": [
                            {
                                "id": r.memory.id,
                                "content": r.memory.content,
                                "score": r.score,
                                "version": r.memory.version,
                                "session_id": r.memory.session_id,
                                "related_entities": r.related_entities,
                            }
                            for r in page
                        ],
                        "total": len(results),
                        "offset": offset,
                        "has_more": len(results) > offset + limit,
                    },
                    indent=2,
                )

            elif name == "collivind_get_context":
                context = self.memory_manager.get_context(
                    query_str=args["query"],
                    project_id=args.get("project_id", "default"),
                    limit=args.get("limit", 10),
                    max_tokens=args.get("max_tokens"),
                )
                return context

            elif name == "collivind_mark_conflict":
                try:
                    self.memory_manager.link(args["memory_id"], args["other_id"], RelType.CONTRADICTS)
                except LookupError as e:
                    return json.dumps({"error": str(e)})
                return json.dumps({"status": "success", "message": "Conflict flagged"})

            elif name == "collivind_invalidate":
                self.memory_manager.invalidate(
                    memory_id=args["memory_id"], superseded_by=args["superseded_by"], reason=args["reason"]
                )
                return json.dumps({"status": "success", "message": "Memory invalidated"})

            elif name == "collivind_forget":
                if not self.memory_manager.forget(args["memory_id"]):
                    return json.dumps({"error": "Memory not found"})
                return json.dumps({"status": "success", "message": "Memory forgotten"})

            elif name == "collivind_get_entity":
                entity_data = self.memory_manager.get_entity(args["entity_name"])
                if not entity_data:
                    return json.dumps({"error": "Entity not found"})
                return json.dumps(entity_data, indent=2)

            elif name == "collivind_get_timeline":
                offset = args.get("offset", 0)
                limit = args.get("limit", 50)
                timeline = self.memory_manager.get_timeline(
                    project_id=args.get("project_id", "default"),
                    entity=args.get("entity_name"),
                    limit=limit + offset,
                )
                sid = args.get("session_id")
                if sid:
                    timeline = [m for m in timeline if m.session_id == sid]
                timeline_page = timeline[offset : offset + limit]
                return json.dumps(
                    {
                        "results": [m.to_dict() for m in timeline_page],
                        "total": len(timeline),
                        "offset": offset,
                        "has_more": len(timeline) > offset + limit,
                    },
                    indent=2,
                )

            elif name == "collivind_find_contradictions":
                candidates = self.memory_manager.find_conflicts(args["memory_id"])
                if candidates is None:
                    return json.dumps({"error": "Memory not found"})
                return json.dumps(
                    {"possible_conflicts": _conflict_view(candidates), "conflict_instruction": CONFLICT_INSTRUCTION},
                    indent=2,
                )

            elif name == "collivind_batch_add":
                return self._batch_add(args)

            elif name == "collivind_get_version_chain":
                chain = self.memory_manager.get_version_chain(args["memory_id"])
                return json.dumps(
                    [
                        {
                            "id": m.id,
                            "content": m.content,
                            "version": m.version,
                            "valid_from": m.valid_from.isoformat() if m.valid_from else None,
                            "valid_to": m.valid_to.isoformat() if m.valid_to else None,
                            "superseded_by": m.superseded_by,
                        }
                        for m in chain
                    ],
                    indent=2,
                )

            elif name == "collivind_extract":
                prompt = build_extraction_prompt(
                    args["text"],
                    project_id=args.get("project_id", "default"),
                )
                return json.dumps(
                    {
                        "status": "prompt_ready",
                        "prompt": prompt,
                        "instructions": (
                            "Send this prompt to your LLM, then pass the JSON response "
                            "to collivind_extract_save to store the extracted memories."
                        ),
                    },
                    indent=2,
                )

            elif name == "collivind_extract_save":
                extracted = parse_extraction_response(args["llm_response"])
                if not extracted:
                    return json.dumps({"status": "no_memories", "count": 0})
                project_id = args.get("project_id", "default")
                prepared: List[tuple] = []
                for r in extracted:
                    entities = [EntityCreate(name=e["name"], type=EntityType(e["type"])) for e in r.entities]
                    mem_create = MemoryCreate(
                        content=r.content,
                        summary=r.summary,
                        category=MemoryCategory(r.category),
                        project_id=project_id,
                        confidence=r.confidence,
                        tags=r.tags,
                        session_id=self.session_id,
                    )
                    prepared.append((mem_create, entities, []))
                # Validate all, then store: same all-or-report contract as batch_add (#37).
                ids, conflicts = self.memory_manager.store_batch_checked(prepared)
                result = {"status": "success", "count": len(ids), "memory_ids": ids}
                return json.dumps(_with_conflicts(result, conflicts))

            else:
                return json.dumps({"error": f"Unknown tool: {name}"})

        except BatchPartiallyStoredError as e:
            return json.dumps(
                {
                    "error": str(e),
                    "partial": True,
                    "stored_memory_ids": e.stored_ids,
                    "failed_index": e.failed_index,
                    "hint": BACKEND_HINT,
                }
            )
        except CollivindError as e:
            # Backend failures used to surface as a bare driver trace with no
            # pointer at the cause (#39).
            return json.dumps({"error": str(e), "hint": BACKEND_HINT})
        except (KeyError, TypeError, ValueError) as e:
            return json.dumps({"error": str(e)})  # bad arguments, not a backend
        except Exception as e:
            # Store drivers raise their own types (a dead Neo4j surfaces as
            # neo4j.exceptions.ServiceUnavailable from most graph reads), so
            # anything that is not an argument error gets the pointer too.
            return json.dumps({"error": str(e), "hint": BACKEND_HINT})
