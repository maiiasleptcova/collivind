import json
import logging
import sys
import uuid
from typing import Any, Dict, Optional

from collivind.config import load_config
from collivind.engine.memory_manager import MemoryManager
from collivind.mcp.tools import CollivindTools
from collivind.storage.factory import create_all_backends

logger = logging.getLogger(__name__)


class MCPServer:
    def __init__(self):
        self.initialized = False
        self.backends_available = False
        self.manager = None
        self.tools = None
        self._init_error = None
        self.session_id = str(uuid.uuid4())

        try:
            self.config = load_config()
            self.vector_store, self.graph_store, self.embedding_provider = create_all_backends(self.config)

            self.manager = MemoryManager(
                vector_store=self.vector_store,
                graph_store=self.graph_store,
                embedding_provider=self.embedding_provider,
                config=self.config,
            )

            self.tools = CollivindTools(self.manager, session_id=self.session_id)
            self.backends_available = True
        except Exception as e:
            self._init_error = str(e)
            logger.warning(f"Server starting in degraded mode: {e}")
            print(f"Warning: backends unavailable, starting in degraded mode: {e}", file=sys.stderr)

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
            server_info = {
                "name": "collivind",
                "version": "0.1.0"
            }
            if not self.backends_available:
                server_info["status"] = "degraded"
                server_info["degraded_reason"] = self._init_error
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {}
                    },
                    "serverInfo": server_info
                }
            }
            
        elif method == "notifications/initialized":
            return None # Notifications don't get responses
            
        elif method == "tools/list":
            # Tool schemas are static — advertise them even in degraded mode
            # so clients can see what exists; calls return isError instead.
            tools_list = self.tools.get_tool_list() if self.tools else CollivindTools.get_tool_list()
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "tools": tools_list
                }
            }
            
        elif method == "tools/call":
            tool_name = params.get("name")
            tool_args = params.get("arguments", {})

            if not self.backends_available:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "isError": True,
                        "content": [
                            {
                                "type": "text",
                                "text": f"Storage backends unavailable: {self._init_error}. "
                                        f"Tool '{tool_name}' cannot execute in degraded mode."
                            }
                        ]
                    }
                }

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
