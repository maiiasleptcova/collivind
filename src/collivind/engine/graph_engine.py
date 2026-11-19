from typing import Any, Dict, List, Optional

from collivind.storage.interfaces import GraphStore


class GraphEngine:
    def __init__(self, graph_store: GraphStore):
        self.graph_store = graph_store

    def _neighbor_identity(self, neighbor: Dict[str, Any]) -> tuple[Optional[str], Optional[str]]:
        node = neighbor.get("node") if isinstance(neighbor.get("node"), dict) else {}
        ent_id = neighbor.get("id") or node.get("id")
        ent_name = neighbor.get("name") or node.get("name")
        if not ent_name and ent_id:
            ent_name = ent_id.replace("_", " ")
        return ent_id, ent_name

    def get_expanded_memories(self, seed_memory_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Find memories connected to the seed memories via shared entities.
        Returns a dictionary mapping memory_id to {"memory": MemoryNode, "shared_entities": [entity names]}

        Two batched store calls (seed -> entities, entities -> memories) rather
        than one per seed and one per entity per seed. The walk order (seeds,
        then each seed's entities, then each entity's memories) is the old
        per-item order, so dict insertion order, and with it the stable sort's
        tie-breaking in SearchEngine, is unchanged.
        """
        if not seed_memory_ids:
            return {}
        seeds = set(seed_memory_ids)

        neighbors = self.graph_store.get_neighbors_batch(seed_memory_ids, rel_types=["ABOUT", "MENTIONS"])
        entity_names: List[str] = []
        for mem_id in seed_memory_ids:
            for n in neighbors.get(mem_id, []):
                ent_id, ent_name = self._neighbor_identity(n)
                if ent_id:
                    entity_names.append(ent_name or ent_id.replace("_", " "))
        # a later seed repeating an entity added nothing new on the per-item path
        entity_names = list(dict.fromkeys(entity_names))

        related = self.graph_store.find_related_memories_batch(entity_names, limit=20)
        expanded: Dict[str, Dict[str, Any]] = {}
        for ent_name in entity_names:
            for r_mem in related.get(ent_name, []):
                if r_mem.id in seeds:
                    continue
                entry = expanded.setdefault(r_mem.id, {"memory": r_mem, "shared_entities": {}})
                entry["shared_entities"][ent_name] = None  # ordered set

        return {
            mem_id: {"memory": entry["memory"], "shared_entities": list(entry["shared_entities"])}
            for mem_id, entry in expanded.items()
        }
