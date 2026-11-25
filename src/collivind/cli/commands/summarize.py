"""`collivind summarize`: read a group's members, or store the summary an agent wrote for it.

Collivind never writes summary text; this is the terminal twin of the
`collivind_summarize` MCP tool.
"""

import json
import sys

import click

from collivind.engine import summaries
from collivind.engine.project_labels import resolve_project


@click.command()
@click.option("--session", "session_id", default=None, help="Session group: memories saved with this session id")
@click.option("--topic", default=None, help="Topic group, named by you (needs --member)")
@click.option("--member", "member_ids", multiple=True, help="Member memory id (repeatable)")
@click.option("--text", default=None, help='Summary text, one "- " fact per line; "-" reads stdin')
@click.option("--project", "-p", default=None, help="Project (default: resolved from the current directory)")
@click.option("--json", "as_json", is_flag=True, help="JSON output")
def summarize(session_id, topic, member_ids, text, project, as_json):
    """Show a group's members (no --text), or store its summary (--text)."""
    from collivind.cli.commands.memory import _fail, _manager

    project = project or resolve_project()
    try:
        group = summaries.group_for(session_id=session_id, topic=topic)
        manager = _manager()
        ids = list(member_ids) or None
        if text is None:
            view = summaries.group_view(manager, project, group, ids)
            if as_json:
                click.echo(json.dumps(view, indent=2))
                return
            current = view["current_summary"]
            click.secho(f"{group.kind} {group.key}: {len(view['members'])} members", bold=True)
            if current:
                click.secho(f"current summary (v{current['version']}, {current['id']}):", fg="cyan")
                click.echo(current["content"])
            for m in view["members"]:
                flag = click.style(" new", fg="green") if m["new"] else ""
                click.echo(f"- [{m['date']}] {m['content']}  ({m['id']}){flag}")
            return
        if text == "-":
            text = sys.stdin.read()
        node = summaries.save_summary(manager, project, group, text, ids)
    except (LookupError, ValueError) as e:
        _fail(f"Error: {e}")
        return
    except Exception as e:
        _fail(f"Error: {e}\nRun `collivind status` to check the backends.")
        return
    count = len(summaries.members_of(manager, [node.id])[node.id])
    if as_json:
        click.echo(json.dumps({"summary_id": node.id, "version": node.version, "members": count}))
    else:
        click.secho(f"Saved summary {node.id} (v{node.version}) over {count} memories.", fg="green")
