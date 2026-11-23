"""`collivind contradictions review`: triage the CONTRADICTS edges older versions wrote on similarity alone."""

import json
from collections import Counter

import click

from collivind.cli.commands.memory import _fail, _manager
from collivind.engine.conflicts import DUPLICATE, RESOLVED, REVIEW, UNRELATED, review_backlog

ACTIONS = {
    RESOLVED: "re-label RELATES_TO (one side already retired)",
    UNRELATED: "re-label RELATES_TO",
    DUPLICATE: "invalidate the older memory (superseded by the newer)",
    REVIEW: "keep CONTRADICTS: read it and resolve by hand or with an agent",
}


@click.group()
def contradictions():
    """Inspect and clean up CONTRADICTS edges."""


@contradictions.command()
@click.option("--project", "-p", default=None, help="Only edges touching this project (default: all)")
@click.option("--apply", "do_apply", is_flag=True, help="Act on the buckets (default: dry run, nothing written)")
@click.option("--show", default=20, show_default=True, help="Pairs to print per bucket (0 = none)")
@click.option("--json", "as_json", is_flag=True, help="Every verdict as JSON")
def review(project, do_apply, show, as_json):
    """Bucket auto-detected CONTRADICTS edges; --apply re-labels or invalidates, never deletes a memory.

    Hand-drawn CONTRADICTS edges are never touched.
    """
    try:
        manager = _manager()
        verdicts = review_backlog(manager, project_id=project, apply=do_apply)
    except Exception as e:
        _fail(f"Review failed: {e}")
    if as_json:
        click.echo(json.dumps([v.__dict__ for v in verdicts], indent=2))
        return
    counts = Counter(v.bucket for v in verdicts)
    click.secho(f"{len(verdicts)} auto-detected CONTRADICTS edges", bold=True)
    for bucket in (RESOLVED, UNRELATED, DUPLICATE, REVIEW):
        click.echo(f"  {bucket:<10} {counts[bucket]:>6}  -> {ACTIONS[bucket]}")
    graph = manager.graph_store
    for bucket in (DUPLICATE, REVIEW):
        picked = [v for v in verdicts if v.bucket == bucket][:show]
        if picked:
            click.secho(f"\n{bucket}:", fg="yellow")
        for v in picked:
            a, b = graph.get_memory(v.source_id), graph.get_memory(v.target_id)
            click.echo(f"  {v.source_id} <> {v.target_id}  ({v.reason})")
            for m in (a, b):
                if m is not None:
                    click.secho(f"      {m.summary or m.content[:100]}", fg="bright_black")
    if do_apply:
        click.secho("\nApplied.", fg="green")
    else:
        click.secho("\nDry run: nothing changed. Re-run with --apply to act on the buckets.", fg="cyan")
