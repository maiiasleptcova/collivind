"""`collivind entities`: find entities that are one thing under several names, and merge them."""

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import click

from collivind.engine.entity_duplicates import merge_command
from collivind.engine.entity_merge import MergePlan


@click.group()
def entities():
    """Find and merge duplicate entities."""


@entities.command()
@click.option("--project", "-p", default=None, help="Only entities with memories in this project")
@click.option("--json", "as_json", is_flag=True, help="Machine-readable output")
@click.option("--no-embeddings", is_flag=True, help="Skip the name-embedding comparison")
@click.option("--threshold", default=0.92, show_default=True, type=click.FloatRange(0.5, 1.0))
@click.option("--limit", default=30, show_default=True, type=click.IntRange(1), help="Groups shown (text output)")
def duplicates(project, as_json, no_embeddings, threshold, limit):
    """List groups of entities that may be duplicates, with the evidence. Never merges."""
    from collivind.cli.commands.memory import _fail, _manager

    try:
        groups = _manager().find_duplicate_entities(project, not no_embeddings, threshold)
    except Exception as e:
        _fail(f"Duplicate search failed: {e}")
        return
    for g in groups:
        g["command"] = merge_command(g)
    if as_json:
        click.echo(json.dumps(groups, indent=2))
        return
    if not groups:
        click.echo("No likely duplicate entities found.")
        return
    for g in groups[:limit]:
        click.secho(f"\n{g['memories']} memories across {len(g['members'])} entities", bold=True)
        for m in g["members"]:
            click.echo(f"  {m['name']!r:<40} {m['type']:<8} {m['memories']:>5} memories {m['edges']:>5} edges")
        for p in g["pairs"]:
            why = ", ".join(e["reason"] + (f" {e['similarity']}" if "similarity" in e else "") for e in p["evidence"])
            click.secho(f"    {p['a']!r} ~ {p['b']!r}: {why}; {p['shared_memories']} shared", fg="bright_black")
        click.secho(f"  dry run: {g['command']}", fg="cyan")
    if len(groups) > limit:
        click.secho(f"\n... and {len(groups) - limit} more groups (--limit, --json)", fg="bright_black")


def _show_plan(plan: MergePlan) -> None:
    click.echo(f"Keep {plan.keep.name!r} ({plan.keep.id}); merge away:")
    for other in plan.others:
        click.echo(f"  {other.name!r} ({other.id})")
    c = plan.counts()
    click.echo(
        f"{c['edges_removed']} edges touch them: {c['memory_edges_moved']} memory links and "
        f"{c['entity_edges_moved']} entity links move to {plan.keep.name!r}; "
        f"{c['duplicates_dropped']} duplicates and {c['self_loops_dropped']} self-loops are dropped."
    )
    by_type = Counter(getattr(e.type, "value", e.type) for e in plan.moved)
    if by_type:
        click.echo("  moved by type: " + ", ".join(f"{t} {n}" for t, n in sorted(by_type.items())))
    for e in [x for x in plan.moved if getattr(x.type, "value", x.type) not in ("ABOUT", "MENTIONS")]:
        click.secho(f"    {e.source_id} -{getattr(e.type, 'value', e.type)}-> {e.target_id}", fg="bright_black")
    for e, why in plan.dropped:
        click.secho(
            f"    drop ({why}) {e.source_id} -{getattr(e.type, 'value', e.type)}-> {e.target_id}", fg="bright_black"
        )
    click.echo(f"Aliases on {plan.keep.name!r} afterwards: {', '.join(plan.aliases)}")
    if plan.topic_summaries:
        click.secho(
            f"{len(plan.topic_summaries)} topic summaries are keyed on a merged name and keep that key; "
            "re-run `collivind summarize --topic` for the kept name if you want one there.",
            fg="yellow",
        )


@entities.command()
@click.argument("keep")
@click.argument("others", nargs=-1, required=True)
@click.option("--apply", "apply_", is_flag=True, help="Write the change (without it: dry run)")
@click.option("--backup", type=click.Path(dir_okay=False, path_type=Path), default=None, help="Backup JSONL path")
def merge(keep, others, apply_, backup):
    """Merge OTHERS into KEEP. Dry run unless --apply.

    Every memory link and entity link moves to KEEP; self-loops and duplicate
    links are dropped; OTHERS' names become KEEP's aliases, so later saves
    naming them land on KEEP. A JSONL backup of every touched row is written
    first. Interrupted? Run the same command again.
    """
    from collivind.cli.commands.memory import _fail, _manager

    try:
        manager = _manager()
        plan = manager.plan_entity_merge(keep, others)
    except (LookupError, ValueError) as e:
        _fail(f"Error: {e.args[0]}")
        return
    except Exception as e:
        _fail(f"Merge plan failed: {e}")
        return
    _show_plan(plan)
    if not apply_:
        click.secho("Dry run: nothing written. Add --apply to merge.", fg="yellow")
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    backup = backup or manager.config.expanded_data_dir / "backups" / f"entity-merge-{plan.keep.id}-{stamp}.jsonl"
    try:
        done = manager.merge_entities(plan, backup)
    except FileExistsError:
        _fail(f"Backup {backup} already exists; pass another --backup path")
        return
    except Exception as e:
        _fail(f"Merge stopped part-way ({e}); run the same command again to finish it. Backup: {backup}")
        return
    click.secho(
        f"Merged {done['entities_removed']} entities into {plan.keep.name!r}: {done['edges_moved']} edges moved, "
        f"{done['duplicates_dropped'] + done['self_loops_dropped']} dropped. Backup: {backup}",
        fg="green",
    )
