"""Direct memory verbs: store, retrieve, and manage memories from the terminal.

Every command supports --json for scripting and agent use.
"""
import json
import sys

import click

from collivind.config import load_config
from collivind.engine.memory_manager import MemoryManager
from collivind.models.memory import MemoryCategory, MemoryCreate
from collivind.storage.factory import create_all_backends

CATEGORIES = [c.value for c in MemoryCategory]


def _manager() -> MemoryManager:
    config = load_config()
    vector_store, graph_store, embedding_provider = create_all_backends(config)
    return MemoryManager(vector_store, graph_store, embedding_provider, config)


def _echo_memory(mem, as_json: bool):
    if as_json:
        click.echo(json.dumps(mem.to_dict(), indent=2))
        return
    cat = mem.category.value if hasattr(mem.category, "value") else mem.category
    click.secho(f"[{cat}] ", fg="yellow", nl=False)
    click.echo(mem.content)
    if mem.tags:
        click.secho(f"       tags: {', '.join(mem.tags)}", fg="bright_black")
    click.secho(f"       id: {mem.id} | project: {mem.project_id} | v{mem.version}", fg="bright_black")


def _fail(message: str):
    click.secho(message, fg="red", err=True)
    sys.exit(1)


@click.command()
@click.argument("content")
@click.option("--summary", "-s", default=None, help="Short summary (defaults to content)")
@click.option("--category", "-c", type=click.Choice(CATEGORIES), default="fact")
@click.option("--project", "-p", default="default")
@click.option("--tags", "-t", default="", help="Comma-separated tags")
@click.option("--confidence", default=1.0, type=float)
@click.option("--json", "as_json", is_flag=True, help="JSON output")
def add(content, summary, category, project, tags, confidence, as_json):
    """Store a memory."""
    try:
        manager = _manager()
        memory = manager.add_memory(MemoryCreate(
            content=content,
            summary=summary or content[:120],
            category=MemoryCategory(category),
            project_id=project,
            confidence=confidence,
            tags=[t.strip() for t in tags.split(",") if t.strip()],
        ))
    except Exception as e:
        _fail(f"Add failed: {e}")
    if as_json:
        click.echo(json.dumps(memory.to_dict(), indent=2))
    else:
        click.secho(f"Stored {memory.id}", fg="green")


@click.command()
@click.argument("memory_id")
@click.option("--json", "as_json", is_flag=True, help="JSON output")
def get(memory_id, as_json):
    """Show one memory by id."""
    try:
        memory = _manager().graph_store.get_memory(memory_id)
    except Exception as e:
        _fail(f"Get failed: {e}")
    if not memory:
        _fail(f"No memory with id {memory_id}")
    _echo_memory(memory, as_json)


@click.command()
@click.argument("query")
@click.option("--project", "-p", default="default")
@click.option("--limit", "-l", default=10)
@click.option("--max-tokens", default=None, type=int, help="Approximate token budget")
def context(query, project, limit, max_tokens):
    """Formatted context block for a query (pipe into prompts/scripts)."""
    try:
        click.echo(_manager().get_context(query, project_id=project, limit=limit, max_tokens=max_tokens))
    except Exception as e:
        _fail(f"Context failed: {e}")


@click.command()
@click.argument("memory_id")
@click.option("--content", default=None)
@click.option("--summary", "-s", default=None)
@click.option("--tags", "-t", default=None, help="Comma-separated tags (replaces existing)")
@click.option("--confidence", default=None, type=float)
@click.option("--json", "as_json", is_flag=True, help="JSON output")
def update(memory_id, content, summary, tags, confidence, as_json):
    """Update fields on a memory (re-embeds when text changes)."""
    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags is not None else None
    try:
        memory = _manager().update_memory(
            memory_id, content=content, summary=summary, tags=tag_list, confidence=confidence
        )
    except Exception as e:
        _fail(f"Update failed: {e}")
    if not memory:
        _fail(f"No memory with id {memory_id}")
    _echo_memory(memory, as_json)


@click.command()
@click.argument("memory_id")
@click.option("--superseded-by", default="", help="Id of the replacing memory")
@click.option("--reason", "-r", default="outdated")
def invalidate(memory_id, superseded_by, reason):
    """Mark a memory as outdated (kept in history, excluded from search)."""
    try:
        _manager().invalidate(memory_id, superseded_by, reason)
    except Exception as e:
        _fail(f"Invalidate failed: {e}")
    click.secho(f"Invalidated {memory_id} ({reason})", fg="green")


@click.command()
@click.argument("memory_id")
@click.confirmation_option(prompt="Permanently delete this memory?")
def forget(memory_id):
    """Delete a memory permanently."""
    try:
        deleted = _manager().forget(memory_id)
    except Exception as e:
        _fail(f"Forget failed: {e}")
    if not deleted:
        _fail(f"No memory with id {memory_id}")
    click.secho(f"Deleted {memory_id}", fg="green")


@click.command(name="export")
@click.option("--project", "-p", default="default")
@click.option("--output", "-o", type=click.File("w"), default="-", help="File path or - for stdout")
def export_cmd(project, output):
    """Export a project's memories as JSONL."""
    try:
        records = _manager().export_memories(project_id=project)
    except Exception as e:
        _fail(f"Export failed: {e}")
    for rec in records:
        output.write(json.dumps(rec) + "\n")
    click.secho(f"Exported {len(records)} memories from '{project}'", fg="green", err=True)


@click.command(name="import")
@click.argument("source", type=click.File("r"))
def import_cmd(source):
    """Import memories from a JSONL export (deduplication applies)."""
    records = []
    for n, line in enumerate(source, 1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as e:
            _fail(f"Line {n} is not valid JSON: {e}")
    try:
        count = _manager().import_memories(records)
    except Exception as e:
        _fail(f"Import failed: {e}")
    click.secho(f"Processed {count} memories (near-duplicates merged, exact ones skipped)", fg="green")
