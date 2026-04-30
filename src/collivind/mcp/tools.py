import json
from typing import Dict, Any, List

from collivind.engine.memory_manager import MemoryManager
from collivind.docker.health import check_all_services
from collivind.models import (
    MemoryCreate, MemoryCategory,
    EntityCreate, EntityType,
    RelationshipCreate, RelType,
    SearchQuery
)

class CollivindTools:
    def __init__(self, memory_manager: MemoryManager):
        self.memory_manager = memory_manager

    def get_tool_list(self) -> List[Dict[str, Any]]:
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
                "description": "Semantic + graph hybrid search.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "category": {"type": "string"},
                        "project_id": {"type": "string", "default": "default"},
                        "limit": {"type": "integer", "default": 10},
                        "include_graph": {"type": "boolean", "default": True}
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
                        "limit": {"type": "integer", "default": 50}
                    },
                    "required": ["project_id"]
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
                                    "confidence": {"type": "number", "default": 1.0}
                                },
                                "required": ["content", "summary", "category"]
                            }
                        }
                    },
                    "required": ["memories"]
                }
            }
        ]

    def handle_call(self, name: str, args: Dict[str, Any]) -> str:
        try:
            if name == "collivind_status":
                status = check_all_services(self.memory_manager.config)
                return json.dumps(status, indent=2)

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
                    tags=args.get("tags", [])
                )
                
                memory = self.memory_manager.add_memory(mem_create, entities=entities, relationships=relationships)
                return json.dumps({"status": "success", "memory_id": memory.id})

            elif name == "collivind_search":
                q = SearchQuery(
                    query=args["query"],
                    category=args.get("category"),
                    project_id=args.get("project_id", "default"),
                    limit=args.get("limit", 10),
                    include_graph=args.get("include_graph", True)
                )
                results = self.memory_manager.search(q)
                return json.dumps([
                    {
                        "id": r.memory.id,
                        "content": r.memory.content,
                        "score": r.score,
                        "related_entities": r.related_entities
                    } for r in results
                ], indent=2)

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
                timeline = self.memory_manager.get_timeline(
                    project_id=args.get("project_id", "default"),
                    entity=args.get("entity_name"),
                    limit=args.get("limit", 50)
                )
                return json.dumps([m.to_dict() for m in timeline], indent=2)

            elif name == "collivind_batch_add":
                ids = self.memory_manager.batch_add_memories(args["memories"])
                return json.dumps({"status": "success", "memory_ids": ids})

            else:
                return json.dumps({"error": f"Unknown tool: {name}"})

        except Exception as e:
            return json.dumps({"error": str(e)})
