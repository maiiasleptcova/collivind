import json
from typing import Any, Dict, List, Optional

from collivind.engine.extractor import build_extraction_prompt, parse_extraction_response
from collivind.engine.memory_manager import MemoryManager
from collivind.models import (
    EntityCreate,
    EntityType,
    MemoryCategory,
    MemoryCreate,
    RelationshipCreate,
    RelType,
    SearchQuery,
)


class CollivindTools:
    def __init__(self, memory_manager: MemoryManager, session_id: Optional[str] = None):
        self.memory_manager = memory_manager
        self.session_id = session_id

    def _backend_health(self) -> Dict[str, Any]:
        backends = {
            "vector_store": self.memory_manager.vector_store,
            "graph_store": self.memory_manager.graph_store,
            "embedding_provider": self.memory_manager.embedding_provider,
        }
        status = {"mode": self.memory_manager.config.mode}
        for name, backend in backends.items():
            try:
                status[name] = backend.health_check()
            except Exception as e:
                status[name] = {"status": "error", "message": str(e)}
        return status

    @staticmethod
    def get_tool_list() -> List[Dict[str, Any]]:
        return [
            {
                "name": "collivind_status",
                "description": "Check health status of Collivind services.",
                "inputSchema": {
                    "type": "object",
                    "properties": {}
                }
            },
            {
                "name": "collivind_add_memory",
                "description": "Store a single extracted fact, decision, pattern, or error.",
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
                                    "properties": {"type": "object"}
                                },
                                "required": ["name", "type"]
                            }
                        },
                        "relationships": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "target_id": {"type": "string"},
                                    "type": {"type": "string", "enum": [r.value for r in RelType]},
                                    "properties": {"type": "object"}
                                },
                                "required": ["target_id", "type"]
                            }
                        },
                        "tags": {"type": "array", "items": {"type": "string"}},
                        "confidence": {"type": "number", "default": 1.0}
                    },
                    "required": ["content", "summary", "category"]
                }
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
                        "date_from": {"type": "string", "description": "ISO datetime"},
                        "date_to": {"type": "string", "description": "ISO datetime"}
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "collivind_get_context",
                "description": "Get relevant context as formatted text.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "project_id": {"type": "string", "default": "default"},
                        "limit": {"type": "integer", "default": 10}
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "collivind_invalidate",
                "description": "Mark a memory as outdated/superseded.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "memory_id": {"type": "string"},
                        "superseded_by": {"type": "string"},
                        "reason": {"type": "string"}
                    },
                    "required": ["memory_id", "superseded_by", "reason"]
                }
            },
            {
                "name": "collivind_forget",
                "description": "Delete a specific memory permanently.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "memory_id": {"type": "string"}
                    },
                    "required": ["memory_id"]
                }
            },
            {
                "name": "collivind_get_entity",
                "description": "Get an entity and its related memories.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "entity_name": {"type": "string"}
                    },
                    "required": ["entity_name"]
                }
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
                        "offset": {"type": "integer", "default": 0}
                    },
                    "required": ["project_id"]
                }
            },
            {
                "name": "collivind_find_contradictions",
                "description": "Find memories that potentially contradict a given memory.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "memory_id": {"type": "string", "description": "ID of the memory to check against"}
                    },
                    "required": ["memory_id"]
                }
            },
            {
                "name": "collivind_batch_add",
                "description": "Add multiple memories in batch.",
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
                                                "properties": {"type": "object"}
                                            },
                                            "required": ["name", "type"]
                                        }
                                    },
                                    "relationships": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "target_id": {"type": "string"},
                                                "type": {"type": "string", "enum": [r.value for r in RelType]},
                                                "properties": {"type": "object"}
                                            },
                                            "required": ["target_id", "type"]
                                        }
                                    },
                                    "tags": {"type": "array", "items": {"type": "string"}},
                                    "confidence": {"type": "number", "default": 1.0},
                                    "session_id": {
                                        "type": "string",
                                        "description": "Session that produced this memory",
                                    },
                                    "user_id": {"type": "string", "default": "local"},
                                    "source": {"type": "string", "default": "manual"}
                                },
                                "required": ["content", "summary", "category"]
                            }
                        }
                    },
                    "required": ["memories"]
                }
            },
            {
                "name": "collivind_get_version_chain",
                "description": "Get the full version history of a memory.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "memory_id": {"type": "string"}
                    },
                    "required": ["memory_id"]
                }
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
                        "project_id": {"type": "string", "default": "default"}
                    },
                    "required": ["text"]
                }
            },
            {
                "name": "collivind_extract_save",
                "description": "Save LLM extraction results as memories.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "llm_response": {"type": "string", "description": "JSON from LLM extraction"},
                        "project_id": {"type": "string", "default": "default"}
                    },
                    "required": ["llm_response"]
                }
            }
        ]

    def handle_call(self, name: str, args: Dict[str, Any]) -> str:
        try:
            if name == "collivind_status":
                return json.dumps(self._backend_health(), indent=2)

            elif name == "collivind_add_memory":
                # Parse entities
                entities = []
                for e in args.get("entities", []):
                    entities.append(EntityCreate(
                        name=e["name"],
                        type=EntityType(e["type"]),
                        properties=e.get("properties", {})
                    ))
                
                # Parse relationships
                relationships = []
                for r in args.get("relationships", []):
                    relationships.append(RelationshipCreate(
                        source_id="", # Will be filled by memory_manager
                        target_id=r["target_id"],
                        type=RelType(r["type"]),
                        properties=r.get("properties", {})
                    ))

                mem_create = MemoryCreate(
                    content=args["content"],
                    summary=args["summary"],
                    category=MemoryCategory(args["category"]),
                    project_id=args.get("project_id", "default"),
                    confidence=args.get("confidence", 1.0),
                    tags=args.get("tags", []),
                    session_id=args.get("session_id", self.session_id),
                )
                
                memory = self.memory_manager.add_memory(mem_create, entities=entities, relationships=relationships)
                return json.dumps({"status": "success", "memory_id": memory.id})

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
                    date_from=date_from,
                    date_to=date_to,
                )
                results = self.memory_manager.search(q)
                limit = args.get("limit", 10)
                page = results[offset:offset + limit]
                return json.dumps({
                    "results": [
                        {
                            "id": r.memory.id,
                            "content": r.memory.content,
                            "score": r.score,
                            "version": r.memory.version,
                            "session_id": r.memory.session_id,
                            "related_entities": r.related_entities,
                        } for r in page
                    ],
                    "total": len(results),
                    "offset": offset,
                    "has_more": len(results) > offset + limit,
                }, indent=2)

            elif name == "collivind_get_context":
                context = self.memory_manager.get_context(
                    query_str=args["query"],
                    project_id=args.get("project_id", "default"),
                    limit=args.get("limit", 10)
                )
                return context

            elif name == "collivind_invalidate":
                self.memory_manager.invalidate(
                    memory_id=args["memory_id"],
                    superseded_by=args["superseded_by"],
                    reason=args["reason"]
                )
                return json.dumps({"status": "success", "message": "Memory invalidated"})

            elif name == "collivind_forget":
                mem_id = args["memory_id"]
                self.memory_manager.graph_store.delete_memory(mem_id)
                self.memory_manager.vector_store.delete(mem_id)
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
                page = timeline[offset:offset + limit]
                return json.dumps({
                    "results": [m.to_dict() for m in page],
                    "total": len(timeline),
                    "offset": offset,
                    "has_more": len(timeline) > offset + limit,
                }, indent=2)

            elif name == "collivind_find_contradictions":
                memory = self.memory_manager.graph_store.get_memory(args["memory_id"])
                if not memory:
                    return json.dumps({"error": "Memory not found"})
                contradictions = self.memory_manager.search_engine.find_contradictions(memory)
                return json.dumps([{
                    "id": c.memory.id,
                    "content": c.memory.content,
                    "category": c.memory.category.value,
                    "similarity": c.vector_score
                } for c in contradictions], indent=2)

            elif name == "collivind_batch_add":
                memories = []
                for mem in args["memories"]:
                    normalized = dict(mem)
                    if self.session_id:
                        normalized.setdefault("session_id", self.session_id)
                    memories.append(normalized)
                ids = self.memory_manager.batch_add_memories(memories)
                return json.dumps({"status": "success", "memory_ids": ids})

            elif name == "collivind_get_version_chain":
                chain = self.memory_manager.get_version_chain(args["memory_id"])
                return json.dumps([
                    {
                        "id": m.id,
                        "content": m.content,
                        "version": m.version,
                        "valid_from": m.valid_from.isoformat() if m.valid_from else None,
                        "valid_to": m.valid_to.isoformat() if m.valid_to else None,
                        "superseded_by": m.superseded_by,
                    } for m in chain
                ], indent=2)

            elif name == "collivind_extract":
                prompt = build_extraction_prompt(
                    args["text"],
                    project_id=args.get("project_id", "default"),
                )
                return json.dumps({
                    "status": "prompt_ready",
                    "prompt": prompt,
                    "instructions": (
                        "Send this prompt to your LLM, then pass the JSON response "
                        "to collivind_extract_save to store the extracted memories."
                    ),
                }, indent=2)

            elif name == "collivind_extract_save":
                results = parse_extraction_response(args["llm_response"])
                if not results:
                    return json.dumps({"status": "no_memories", "count": 0})
                project_id = args.get("project_id", "default")
                ids = []
                for r in results:
                    entities = [
                        EntityCreate(name=e["name"], type=EntityType(e["type"]))
                        for e in r.entities
                    ]
                    mem_create = MemoryCreate(
                        content=r.content,
                        summary=r.summary,
                        category=MemoryCategory(r.category),
                        project_id=project_id,
                        confidence=r.confidence,
                        tags=r.tags,
                        session_id=self.session_id,
                    )
                    node = self.memory_manager.add_memory(mem_create, entities=entities)
                    ids.append(node.id)
                return json.dumps({"status": "success", "count": len(ids), "memory_ids": ids})

            else:
                return json.dumps({"error": f"Unknown tool: {name}"})

        except Exception as e:
            return json.dumps({"error": str(e)})
