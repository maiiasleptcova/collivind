"""`collivind export` / `collivind import`: lossless backup bundles (format v2).

Export is never gated: a user's memory is theirs to take elsewhere.
"""

import json
import sys
from pathlib import Path

import click

from collivind.cli.commands.memory import _manager
from collivind.cli.commands.migrate import _open, _space
from collivind.config import load_config
from collivind.engine.bundle import BundleError, export_bundle, import_bundle, is_bundle_header, read_bundle
from collivind.models.memory import MemoryCategory, validate_confidence

_PLURAL = {"memory": "memories", "entity": "entities", "relationship": "relationships", "vector": "vectors"}


def _fail(message: str):
    click.secho(message, fg="red", err=True)
    sys.exit(1)


def _close(stores) -> None:
    stores.vector.close()
    stores.graph.close()


@click.command(name="export")
@click.option("--project", "-p", default=None, help="Only this project (default: every project)")
@click.option("--output", "-o", type=click.File("w"), default="-", help="File path or - for stdout")
@click.option("--vectors", is_flag=True, help="Include embedding vectors (large; re-embedded on import if omitted)")
@click.option(
    "--okf",
    "okf_dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Instead of a backup, write the memory wiki as Open Knowledge Format Markdown into this directory",
)
def export_cmd(project, output, vectors, okf_dir):
    """Back up memories, entities, relationships and version history as JSONL.

    With --okf DIR, write one Markdown page per topic (entity, summary, project)
    instead: readable, diffable, and openable in OKF tools such as `openwiki visualize`.
    """
    config = load_config()
    if okf_dir is not None:
        return _export_okf(config, okf_dir, project)
    stores = _open(config, "Export")  # strict: a backup taken with the graph down would be partial
    try:
        counts, skipped = export_bundle(stores, output, space=_space(config, stores), project=project, vectors=vectors)
    except Exception as e:
        _fail(f"Export failed: {e}")
    finally:
        _close(stores)
    scope = f"project '{project}'" if project else "all projects"
    summary = ", ".join(f"{n} {_PLURAL[kind] if n != 1 else kind}" for kind, n in counts.items())
    click.secho(f"Exported {scope}: {summary}", fg="green", err=True)
    if project and not counts["memory"]:
        click.secho(f"Project '{project}' has no memories; the backup is empty. Typo?", fg="yellow", err=True)
    if skipped:
        click.secho(
            f"Left out {skipped} edge(s) whose other end is outside the export (another project or deleted).",
            fg="yellow",
            err=True,
        )


def _open_graph(config):
    """The graph store alone: the wiki reads no vectors, so no embedding model is loaded or required."""
    from collivind.storage.factory import create_graph_store

    try:
        graph = create_graph_store(config)
        graph.initialize()
        return graph
    except Exception as e:
        _fail(f"Export mode {config.mode!r} is not reachable; nothing was written. {e}")


def _export_okf(config, okf_dir: Path, project) -> None:
    from collivind.web.okf import export_okf

    graph = _open_graph(config)
    try:
        counts = export_okf(graph, okf_dir, project=project)
    except FileExistsError as e:
        _fail(str(e))
        return
    except Exception as e:
        _fail(f"Export failed: {e}")
        return
    finally:
        graph.close()
    click.secho(
        f"Wrote {counts['files']} files to {okf_dir}: {counts['entity']} topics, {counts['summary']} summaries, "
        f"{counts['project']} projects.",
        fg="green",
        err=True,
    )
    if project and not counts["project"]:
        click.secho(f"Project '{project}' has no memories; the wiki is empty. Typo?", fg="yellow", err=True)


@click.command(name="import")
@click.argument("source", type=click.File("r"))
@click.option("--dry-run", is_flag=True, help="Validate the file and count what would be added; write nothing")
def import_cmd(source, dry_run):
    """Restore a backup made by `collivind export`.

    Bundles (v2) restore by id and only add what is missing, so re-importing
    is safe and never reverts later edits. Older plain-memory exports (v1)
    are re-added through the normal pipeline (deduplication applies).
    """
    lines = source.readlines()
    first = next((line for line in lines if line.strip()), None)
    try:
        header = json.loads(first) if first else None
    except json.JSONDecodeError as e:
        _fail(f"Line 1 is not valid JSON: {e}")
    if is_bundle_header(header):
        _import_bundle(lines, dry_run)
    else:
        _import_v1(lines, dry_run)


def _import_bundle(lines, dry_run: bool) -> None:
    try:
        bundle = read_bundle(lines)
    except BundleError as e:
        _fail(f"Import aborted, nothing written: {e}")
    config = load_config()
    stores = _open(config, "Target")
    try:
        space = _space(config, stores)
        report = import_bundle(bundle, stores, space=space, dry_run=dry_run)
    except Exception as e:
        _fail(f"Import failed: {e}")
    finally:
        _close(stores)
    if report.reembedded:
        why = "no vectors in the bundle" if not bundle.has_vectors else f"bundle space {bundle.space} != {space}"
        click.echo(f"Re-embedding with this install's model ({why}).")
    for edge in report.dangling:
        click.secho(f"skipped dangling edge {edge.id}: an endpoint is not in the bundle", fg="yellow")
    click.echo(f"{'kind':<14}{'bundle':>8}{'target':>8}{'missing':>9}")
    for c in report.counts:
        click.echo(f"{c.kind:<14}{c.source:>8}{c.target:>8}{c.missing:>9}")
    if dry_run:
        click.secho("Dry run: nothing written. 'missing' is what a real import would add.", fg="green")
    elif report.ok:
        click.secho("Every bundle record is present in the target.", fg="green")
    else:
        _fail("Verification failed: some bundle records are missing from the target (see 'missing').")


def _import_v1(lines, dry_run: bool) -> None:
    records = []
    for n, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as e:
            _fail(f"Line {n} is not valid JSON: {e}")
    for n, rec in enumerate(records, 1):
        problem = _v1_problem(rec)
        if problem:
            _fail(f"Import aborted, nothing written: record {n} {problem}")
    if dry_run:
        click.secho(f"Dry run: {len(records)} v1 memories would be re-added (deduplication applies).", fg="green")
        return
    try:
        count = _manager().import_memories(records)
    except Exception as e:
        _fail(f"Import failed: {e}")
    click.secho(f"Processed {count} memories (near-duplicates merged, exact ones skipped)", fg="green")


def _v1_problem(rec) -> str:
    """Why `import_memories` would reject this v1 record, checked before any write."""
    if not isinstance(rec, dict):
        return "is not a JSON object"
    if not isinstance(rec.get("content"), str) or not rec["content"]:
        return "has no content"
    try:
        MemoryCategory(rec.get("category", "fact"))
        if rec.get("confidence") is not None:
            validate_confidence(rec["confidence"])
    except ValueError as e:
        return f"is invalid: {e}"
    return ""
