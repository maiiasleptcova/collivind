import os
import shutil
import string
import subprocess
from pathlib import Path
from importlib import resources

from collivind.config import CollivindConfig
from collivind.exceptions import DockerExecutionError

def copy_templates(data_dir: Path, config: CollivindConfig):
    """Copies Docker templates to the data directory, substituting variables."""
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Read templates using importlib.resources
    pkg = "collivind.docker.templates"
    
    # Process docker-compose.yml
    compose_tmpl = resources.files(pkg).joinpath("docker-compose.yml.template").read_text()
    
    # Substitute variables
    compose_content = string.Template(compose_tmpl).safe_substitute(
        COMPOSE_PROJECT=config.docker.compose_project,
        NEO4J_PASSWORD=config.neo4j.password,
    )
    
    (data_dir / "docker-compose.yml").write_text(compose_content)
    
    # Copy Dockerfile.embeddings
    dockerfile_content = resources.files(pkg).joinpath("Dockerfile.embeddings").read_text()
    (data_dir / "Dockerfile.embeddings").write_text(dockerfile_content)
    
    # Copy embedding_server.py
    server_content = resources.files(pkg).joinpath("embedding_server.py").read_text()
    (data_dir / "embedding_server.py").write_text(server_content)


def check_docker_running():
    """Checks if Docker is running by executing 'docker info'."""
    try:
        result = subprocess.run(
            ["docker", "info"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        raise DockerExecutionError("Docker is not running or not installed. Please start Docker first.")


def docker_compose_up(data_dir: Path):
    """Runs docker-compose up -d --build in the data directory."""
    try:
        subprocess.run(
            ["docker-compose", "up", "-d", "--build"],
            cwd=str(data_dir),
            check=True
        )
    except subprocess.CalledProcessError as e:
        raise DockerExecutionError(f"Failed to start Docker containers: {e}")

def docker_compose_down(data_dir: Path):
    """Runs docker-compose down in the data directory."""
    try:
        subprocess.run(
            ["docker-compose", "down"],
            cwd=str(data_dir),
            check=True
        )
    except subprocess.CalledProcessError as e:
        raise DockerExecutionError(f"Failed to stop Docker containers: {e}")
