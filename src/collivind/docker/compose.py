import hashlib
import re
import string
import subprocess
from importlib import resources
from pathlib import Path

from collivind.config import CollivindConfig
from collivind.exceptions import DockerExecutionError

_NEO4J_AUTH = re.compile(r"NEO4J_AUTH=neo4j/(\S+)")


def copy_templates(data_dir: Path, config: CollivindConfig) -> bool:
    """Write the Docker templates into the data directory, substituting variables.

    Safe to re-run on an existing install, which is how template fixes reach
    it (#39): the Neo4j password already in docker-compose.yml is kept, since
    NEO4J_AUTH only takes effect on a fresh volume and the database still
    expects the original one, and a compose file that changes is first saved
    as docker-compose.yml.bak so hand edits are recoverable.

    Returns "replaced" (backed up), "kept-edits" (hand-edited file left
    alone), or "" (new or unchanged).
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    compose_path = data_dir / "docker-compose.yml"
    existing = compose_path.read_text() if compose_path.exists() else None
    kept = _NEO4J_AUTH.search(existing) if existing else None

    # Read templates using importlib.resources
    pkg = "collivind.docker.templates"

    # Process docker-compose.yml
    compose_tmpl = resources.files(pkg).joinpath("docker-compose.yml.template").read_text()

    # Substitute variables
    compose_content = string.Template(compose_tmpl).safe_substitute(
        COMPOSE_PROJECT=config.docker.compose_project,
        NEO4J_PASSWORD=kept.group(1) if kept else config.neo4j.password,
    )

    # A hash of what we last wrote tells an untouched file (safe to refresh)
    # from one the user edited (kept, with a note). Installs older than the
    # marker have no hash and are refreshed with a backup.
    marker = data_dir / ".docker-compose.sha256"
    if existing is not None and existing != compose_content and marker.exists():
        if hashlib.sha256(existing.encode()).hexdigest() != marker.read_text().strip():
            return "kept-edits"
    replaced = existing is not None and existing != compose_content
    if replaced:
        (data_dir / "docker-compose.yml.bak").write_text(existing)
    compose_path.write_text(compose_content)
    marker.write_text(hashlib.sha256(compose_content.encode()).hexdigest())

    # Copy Dockerfile.embeddings
    dockerfile_content = resources.files(pkg).joinpath("Dockerfile.embeddings").read_text()
    (data_dir / "Dockerfile.embeddings").write_text(dockerfile_content)

    # Copy embedding_server.py
    server_content = resources.files(pkg).joinpath("embedding_server.py").read_text()
    (data_dir / "embedding_server.py").write_text(server_content)
    return "replaced" if replaced else ""


def check_docker_running():
    """Checks if Docker is running by executing 'docker info'."""
    try:
        subprocess.run(["docker", "info"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        raise DockerExecutionError("Docker is not running or not installed. Please start Docker first.")


def docker_compose_up(data_dir: Path):
    """Runs docker-compose up -d --build in the data directory."""
    try:
        subprocess.run(["docker-compose", "up", "-d", "--build"], cwd=str(data_dir), check=True)
    except subprocess.CalledProcessError as e:
        raise DockerExecutionError(f"Failed to start Docker containers: {e}")


def docker_compose_down(data_dir: Path):
    """Runs docker-compose down in the data directory."""
    try:
        subprocess.run(["docker-compose", "down"], cwd=str(data_dir), check=True)
    except subprocess.CalledProcessError as e:
        raise DockerExecutionError(f"Failed to stop Docker containers: {e}")
