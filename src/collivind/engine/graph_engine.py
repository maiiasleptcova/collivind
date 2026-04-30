from typing import List, Dict, Any, Set
from collivind.storage.interfaces import GraphStore
from collivind.models import MemoryNode

class GraphEngine:
    def __init__(self, graph_store: GraphStore):
        self.graph_store = graph_store

    def get_expanded_memories(self, seed_memory_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Find memories connected to the seed memories via shared entities.
        Returns a dictionary mapping memory_id to {"memory": MemoryNode, "shared_entities": [entity names]}
        """
        expanded = {}
        if not seed_memory_ids:
            return expanded

        # We'll use the Neo4j GraphStore to execute a specific query for this expansion.
        # However, to keep it modular, we assume Neo4jGraphStore has a way to run custom queries 
        # or we use the existing methods. 
        # For pure interface-based implementation, we can use `get_neighbors` on each seed.
        
        for mem_id in seed_memory_ids:
            # 1. Get entities for this memory
            neighbors = self.graph_store.get_neighbors(mem_id, rel_types=["ABOUT", "MENTIONS"], direction="OUT", depth=1)
            
            for n in neighbors:
                entity = n["node"]
                ent_name = entity.get("name")
                if not ent_name:
                    continue
                    
                # 2. Get memories for this entity
                related_mems = self.graph_store.find_related_memories(ent_name, limit=20)
                for r_mem in related_mems:
                    if r_mem.id in seed_memory_ids:
                        continue
                        
                    if r_mem.id not in expanded:
                        expanded[r_mem.id] = {"memory": r_mem, "shared_entities": set()}
                    expanded[r_mem.id]["shared_entities"].add(ent_name)

        # Convert sets to lists for output
        for mem_id in expanded:
            expanded[mem_id]["shared_entities"] = list(expanded[mem_id]["shared_entities"])
            
        return expanded
