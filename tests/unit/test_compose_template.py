"""The compose template is the whole deployment contract for docker mode.

Issue #39: `/logs` was never declared, so Docker backed it with an anonymous
volume. When that volume could not be written, Neo4j treated the failed log
append as fatal and exited; `restart: unless-stopped` then crash-looped it 67
times over 46 hours. Nothing surfaced it, because no service declared a
healthcheck, so `docker ps` reported "Restarting" rather than "unhealthy" and
the outage was invisible until someone ran a status check by hand.

These assert the two properties that failure needed, so neither can be
removed silently.
"""

import re
from pathlib import Path

import pytest
import yaml  # dev dependency; a skipped guard here would guard nothing

TEMPLATE = Path(__file__).parents[2] / "src/collivind/docker/templates/docker-compose.yml.template"


@pytest.fixture(scope="module")
def compose():
    """The template with its shell placeholders filled, parsed as YAML."""
    text = TEMPLATE.read_text()
    text = re.sub(r"\$\{(\w+)\}", r"placeholder-\1", text)
    return yaml.safe_load(text)


SERVICES = ("qdrant", "neo4j", "embeddings")


def test_the_template_still_parses(compose):
    assert set(compose["services"]) == set(SERVICES)


@pytest.mark.parametrize("service", SERVICES)
def test_every_service_declares_a_healthcheck(compose, service):
    """Without one a crash loop shows as "Restarting", which no tooling and no
    human notices until the product has already been unusable for days."""
    hc = compose["services"][service].get("healthcheck")
    assert hc, f"{service} has no healthcheck; a crash loop would be invisible"
    assert hc.get("test"), f"{service} healthcheck has no test command"
    assert hc.get("start_period"), f"{service} needs a start_period or boot counts as failure"


@pytest.mark.parametrize("service", SERVICES)
def test_healthchecks_use_a_tool_the_image_actually_has(compose, service):
    """A probe binary that is absent exits 127 every time, which is a
    permanently-unhealthy service rather than a working check. Verified
    against the real images: qdrant and python:3.11-slim carry neither curl
    nor wget; neo4j carries wget but not curl.
    """
    test = compose["services"][service]["healthcheck"]["test"]
    joined = " ".join(test) if isinstance(test, list) else test
    assert "curl" not in joined, f"{service}'s image has no curl"
    if service != "neo4j":
        assert "wget" not in joined, f"only the neo4j image carries wget, not {service}'s"


def test_neo4j_logs_are_not_an_anonymous_volume(compose):
    """The root cause of #39. Undeclared, /logs gets an unmanaged anonymous
    volume, and a write failure inside it is fatal to the database."""
    neo4j = compose["services"]["neo4j"]
    targets = {v.split(":")[1] for v in neo4j["volumes"] if ":" in v}
    assert "/logs" in targets, "/logs is undeclared and would fall back to an anonymous volume"

    source = next(v.split(":")[0] for v in neo4j["volumes"] if v.endswith(":/logs"))
    assert source in compose["volumes"], f"{source} is used but never declared as a named volume"


def test_no_stdout_logging_env_setting(compose):
    """#39 suggests `NEO4J_server_logs_user_stdout__enabled=true` so a failed
    log write cannot kill the database. That setting does not exist in 5.26 —
    logging is configured by conf/user-logs.xml (log4j2) — and Neo4j's strict
    config validation rejects unknown keys:

        Failed to read config: Unrecognized setting. No declared setting
        with name: server.logs.user.stdout_enabled

    Verified against neo4j:5.26-community: the container refuses to boot and
    crash-loops, turning a disk-pressure outage into an unconditional one on
    every install. This guards the plausible-looking fix from coming back.
    """
    env = compose["services"]["neo4j"]["environment"]
    offending = [e for e in env if "logs_user_stdout" in e]
    assert not offending, f"not a valid 5.26 setting; breaks startup: {offending}"


def test_every_named_volume_is_used(compose):
    """A declared-but-unmounted volume is a typo in a mount path."""
    mounted = {
        v.split(":")[0]
        for svc in compose["services"].values()
        for v in svc.get("volumes", [])
        if ":" in v and not v.startswith((".", "/"))
    }
    assert set(compose["volumes"]) == mounted, "declared volumes and mounted volumes disagree"
