from typing import Any, Dict

import httpx

from collivind.config import CollivindConfig


def check_qdrant_health(config: CollivindConfig) -> Dict[str, Any]:
    """Checks if Qdrant is healthy via its HTTP API."""
    url = f"http://{config.qdrant.host}:{config.qdrant.port}/healthz"
    try:
        resp = httpx.get(url, timeout=2.0)
        if resp.status_code == 200:
            return {"status": "ok", "message": "Qdrant is healthy"}
        return {"status": "error", "message": f"Qdrant returned {resp.status_code}"}
    except httpx.RequestError as e:
        return {"status": "error", "message": f"Connection failed: {e}"}


def check_neo4j_health(config: CollivindConfig) -> Dict[str, Any]:
    """Checks Neo4j connectivity over bolt using the configured URI."""
    from neo4j import GraphDatabase

    try:
        driver = GraphDatabase.driver(
            config.neo4j.uri,
            auth=(config.neo4j.user, config.neo4j.password),
            connection_timeout=2.0,
        )
        try:
            driver.verify_connectivity()
        finally:
            driver.close()
        return {"status": "ok", "message": "Neo4j is healthy"}
    except Exception as e:
        return {"status": "error", "message": f"Connection failed: {e}"}


def check_embeddings_health(config: CollivindConfig) -> Dict[str, Any]:
    """Checks if the embedding service is healthy."""
    url = f"{config.embeddings.service_url}/health"
    try:
        resp = httpx.get(url, timeout=2.0)
        if resp.status_code == 200:
            return {"status": "ok", "message": "Embeddings service is healthy"}
        return {"status": "error", "message": f"Embeddings returned {resp.status_code}"}
    except httpx.RequestError as e:
        return {"status": "error", "message": f"Connection failed: {e}"}


def check_all_services(config: CollivindConfig) -> Dict[str, Dict[str, Any]]:
    """Checks health of all Docker services."""
    return {
        "qdrant": check_qdrant_health(config),
        "neo4j": check_neo4j_health(config),
        "embeddings": check_embeddings_health(config),
    }
