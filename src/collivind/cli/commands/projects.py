"""`collivind projects`: see which project the current directory maps to, and relabel memories."""

import re
import sys

import click

from collivind.engine.project_labels import rename_project, resolve_project, selected_memories
from collivind.engine.projects import list_projects, validate_project_id


def _fail(message: str):
    click.secho(message, fg="red", err=True)
    sys.exit(1)


@click.group()
def projects():
    """List projects and relabel memories between them."""


@projects.command(name="list")
def list_cmd():
    """Projects with memory counts, and the one this directory resolves to."""
    from collivind.cli.commands.memory import _manager

    try:
        rows = list_projects(_manager().graph_store)
    except Exception as e:
        _fail(f"List failed: {e}")
    current = resolve_project()
    for r in rows:
        mark = "*" if r["project_id"] == current else " "
        click.echo(f"{mark} {r['project_id']:<30} {r['live']:>6} live {r['memories']:>6} total")
    click.secho(f"* this directory resolves to '{current}'", fg="bright_black")


@projects.command()
@click.argument("old")
@click.argument("new")
@click.option("--match", "-m", default=None, help="Only memories whose content, summary or tags match this regex")
@click.option("--apply", "apply_", is_flag=True, help="Write the change (without it: dry run)")
def rename(old, new, match, apply_):
    """Relabel OLD's memories as project NEW. Dry run unless --apply.

    Every version of a memory moves together. Ids, links and vectors are kept,
    and an interrupted run is finished by running it again.
    """
    from collivind.cli.commands.memory import _manager

    try:
        new = validate_project_id(new).strip()
        if match:
            re.compile(match)
    except (ValueError, re.error) as e:
        _fail(f"Invalid argument: {e}")
    if old == new:
        _fail("OLD and NEW are the same project")
    try:
        manager = _manager()
        chosen = selected_memories(manager.graph_store, old, match)
    except Exception as e:
        _fail(f"Rename failed: {e}")

    live = sum(1 for m in chosen if not m.superseded_by and not m.valid_to)
    scope = f" matching /{match}/" if match else ""
    click.echo(f"{len(chosen)} memories ({live} live) in '{old}'{scope} -> '{new}'")
    for m in chosen[:10]:
        click.secho(f"  {m.id[:8]}  {(m.summary or m.content)[:90]}", fg="bright_black")
    if len(chosen) > 10:
        click.secho(f"  ... and {len(chosen) - 10} more", fg="bright_black")
    if not apply_:
        click.secho("Dry run: nothing written. Add --apply to relabel.", fg="yellow")
        return
    if not chosen:
        return
    try:
        done = rename_project(manager.graph_store, manager.vector_store, chosen, new)
    except Exception as e:
        _fail(f"Rename stopped part-way ({e}); run the same command again to finish it")
    click.secho(f"Relabelled {done['memories']} memories and {done['vectors']} vectors as '{new}'", fg="green")
