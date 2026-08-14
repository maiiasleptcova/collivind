"""The compose template is the whole deployment contract for docker mode.

Issue #39: `/logs` was never declared, so Docker backed it with an anonymous
volume. When that volume could not be written, Neo4j treated the failed log
append as fatal and exited; `restart: unless-stopped` then crash-looped it 67
times over 46 hours, and collivind was unusable for the whole period.

Neither change here fixes that. Being precise about what they do, because
two rounds of review were spent on claims that outran the diff:

- The named `/logs` volume stops each down/up cycle leaking another dangling
  anonymous volume and makes the logs findable. It does NOT prevent the crash
  loop: named and anonymous volumes share a filesystem, so a full disk is
  equally fatal. That needs ignoreExceptions="true" in conf/user-logs.xml.
- The healthchecks do NOT surface a crash loop either. Measured: a
  crash-looping container shows `Restarting (1)` in `docker ps`, not
  `(unhealthy)`, and its probe runs zero times, because probes only execute
  while the container is running. They cover up-but-not-serving — and note
  the neo4j probe only proves the JVM is up and Jetty is bound, not that the
  database can answer a query.

What is unambiguously fixed is the invalid stdout setting, below.
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
    """Covers the up-but-not-serving shape only — see the module docstring for
    why these do NOT surface a crash loop."""
    hc = compose["services"][service].get("healthcheck")
    assert hc, f"{service} declares no healthcheck"
    assert hc.get("test"), f"{service} healthcheck has no test command"
    assert hc.get("start_period"), f"{service} needs a start_period or boot counts as failure"


@pytest.mark.parametrize("service", SERVICES)
def test_healthchecks_use_a_tool_the_image_actually_has(compose, service):
    """A probe binary that is absent exits non-zero every time, which is a
    permanently-unhealthy service rather than a working check. Verified
    against the real images: qdrant and python:3.11-slim carry neither curl
    nor wget; neo4j carries wget but not curl.
    """
    test = compose["services"][service]["healthcheck"]["test"]
    joined = " ".join(test) if isinstance(test, list) else test
    assert "curl" not in joined, f"{service}'s image has no curl"
    if service != "neo4j":
        assert "wget" not in joined, f"only the neo4j image carries wget, not {service}'s"


@pytest.mark.parametrize("service", SERVICES)
def test_dev_tcp_probes_invoke_bash_explicitly(compose, service):
    """`/dev/tcp` is a bash builtin, not a file and not POSIX. `CMD-SHELL`
    runs `/bin/sh -c`, and /bin/sh is dash in these images, so a /dev/tcp
    probe under CMD-SHELL fails on every run:

        sh: 1: cannot create /dev/tcp/127.0.0.1/6333: Directory nonexistent

    That is worse than having no healthcheck — the service reports unhealthy
    forever while serving fine, which teaches everyone to ignore the field.
    Verified against qdrant/qdrant:v1.12.6, where /bin/sh -> dash and
    /bin/bash exists separately.
    """
    test = compose["services"][service]["healthcheck"]["test"]
    joined = " ".join(test) if isinstance(test, list) else test
    if "/dev/tcp" not in joined:
        pytest.skip(f"{service} does not use /dev/tcp")
    assert isinstance(test, list) and test[0] == "CMD", (
        f"{service} uses /dev/tcp under {test[0]}; CMD-SHELL is `sh -c` and dash has no /dev/tcp"
    )
    assert test[1] == "bash", f"{service} must invoke bash explicitly for /dev/tcp, got {test[1]!r}"


def test_neo4j_logs_are_not_an_anonymous_volume(compose):
    """Undeclared, /logs gets a fresh anonymous volume that `docker-compose
    down` leaves dangling, so each down/up cycle leaks another one and the
    incident logs are unfindable.

    Naming it does not stop #39 recurring — same filesystem, so a full disk is
    equally fatal. See the module docstring."""
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
