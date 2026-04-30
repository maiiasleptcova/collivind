import sys
import json
from typing import Dict, Any, Optional

from collivind.config import load_config
from collivind.storage.qdrant_store import QdrantVectorStore
from collivind.storage.neo4j_store import Neo4jGraphStore
from collivind.storage.embedding_service import HttpEmbeddingProvider
from collivind.engine.memory_manager import MemoryManager
from collivind.mcp.tools import CollivindTools
from collivind.exceptions import CollivindError

class MCPServer:
    def __init__(self):
        try:
            self.config = load_config()
            self.vector_store = QdrantVectorStore(self.config.qdrant, self.config.embeddings.dimension)
            self.graph_store = Neo4jGraphStore(self.config.neo4j)
            self.embedding_provider = HttpEmbeddingProvider(self.config.embeddings)
            
            self.manager = MemoryManager(
                vector_store=self.vector_store,
                graph_store=self.graph_store,
                embedding_provider=self.embedding_provider,
                config=self.config
            )
            
            self.tools = CollivindTools(self.manager)
            self.initialized = False
        except Exception as e:
            # Output error to stderr so as not to break JSON-RPC
            print(f"Failed to initialize components: {e}", file=sys.stderr)
            sys.exit(1)

    def serve(self):
        """Main stdio loop for JSON-RPC 2.0."""
        for line in sys.stdin:
            if not line.strip():
                continue
                
            try:
                request = json.loads(line)
                response = self.handle_request(request)
                if response:
                    sys.stdout.write(json.dumps(response) + "\n")
                    sys.stdout.flush()
            except json.JSONDecodeError:
                self.send_error(None, -32700, "Parse error")
            except Exception as e:
                self.send_error(request.get("id"), -32603, f"Internal error: {e}")

    def handle_request(self, request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        req_id = request.get("id")
        method = request.get("method")
        params = request.get("params", {})

        if method == "initialize":
            self.initialized = True
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {}
                    },
                    "serverInfo": {
                        "name": "collivind",
                        "version": "0.1.0"
                    }
                }
            }
            
        elif method == "notifications/initialized":
            return None # Notifications don't get responses
            
        elif method == "tools/list":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "tools": self.tools.get_tool_list()
                }
            }
            
        elif method == "tools/call":
            tool_name = params.get("name")
            tool_args = params.get("arguments", {})
            
            try:
                content_str = self.tools.handle_call(tool_name, tool_args)
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": content_str
                            }
                        ]
                    }
                }
            except Exception as e:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "isError": True,
                        "content": [
                            {
                                "type": "text",
                                "text": f"Error executing tool: {e}"
                            }
                        ]
                    }
                }
                
        else:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {
                    "code": -32601,
                    "message": "Method not found"
                }
            }

    def send_error(self, req_id: Any, code: int, message: str):
        err = {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {
                "code": code,
                "message": message
            }
        }
        sys.stdout.write(json.dumps(err) + "\n")
        sys.stdout.flush()

if __name__ == "__main__":
    server = MCPServer()
    server.serve()
