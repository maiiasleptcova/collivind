"""Install in-session slash commands (/mem-save, /mem-recall) for agents.

Claude Code and Codex get user-level commands; Copilot prompt files are
workspace-level, so those install into the current repo's .github/prompts.
"""

from pathlib import Path

import click

SAVE_BODY = """Store the following knowledge in Collivind memory: $ARGUMENTS

Use the `collivind_add_memory` MCP tool if available, otherwise run
`collivind add "<content>" --category <category>` via shell.
Pick the right category (fact, decision, pattern, error, architecture,
preference, snippet), extract entities, and write a one-line summary.
If no text was given, extract and store the noteworthy knowledge from the
current conversation instead. Confirm with the stored memory id.
"""

RECALL_BODY = """Retrieve stored knowledge relevant to: $ARGUMENTS

Use the `collivind_get_context` MCP tool if available, otherwise run
`collivind context "$ARGUMENTS"` via shell. Briefly present what was found
and apply it to the ongoing task. If nothing relevant is stored, say so.
"""

COMMANDS = {"mem-save": SAVE_BODY, "mem-recall": RECALL_BODY}

DESCRIPTIONS = {
    "mem-save": "Save knowledge to Collivind memory",
    "mem-recall": "Recall relevant knowledge from Collivind memory",
}


def _claude_file(name: str) -> str:
    return f"---\ndescription: {DESCRIPTIONS[name]}\n---\n{COMMANDS[name]}"


def _copilot_file(name: str) -> str:
    return f"---\ndescription: {DESCRIPTIONS[name]}\n---\n{COMMANDS[name]}"


def install_commands(tool: str, project_dir: Path | None = None) -> list[Path]:
    """Write the command files for one tool. Returns the paths written."""
    if tool == "claude":
        target = Path.home() / ".claude" / "commands"
        files = {f"{n}.md": _claude_file(n) for n in COMMANDS}
    elif tool == "codex":
        target = Path.home() / ".codex" / "prompts"
        files = {f"{n}.md": COMMANDS[n] for n in COMMANDS}
    elif tool == "copilot":
        target = (project_dir or Path.cwd()) / ".github" / "prompts"
        files = {f"{n}.prompt.md": _copilot_file(n) for n in COMMANDS}
    else:
        raise click.ClickException(f"Unknown tool: {tool}")

    target.mkdir(parents=True, exist_ok=True)
    written = []
    for filename, content in files.items():
        path = target / filename
        path.write_text(content)
        written.append(path)
    return written


def install_all_commands() -> dict:
    """Claude Code always, Codex when ~/.codex exists. Copilot is
    workspace-level and only installs when explicitly requested."""
    results = {"claude": install_commands("claude")}
    if (Path.home() / ".codex").exists():
        results["codex"] = install_commands("codex")
    return results


@click.group()
def commands():
    """Manage in-session agent commands (/mem-save, /mem-recall)."""


@commands.command()
@click.option(
    "--tool",
    type=click.Choice(["claude", "codex", "copilot", "auto"]),
    default="auto",
    help="auto = Claude Code + Codex if present; copilot writes into ./.github/prompts",
)
def install(tool):
    """Install /mem-save and /mem-recall for your agents."""
    if tool == "auto":
        results = install_all_commands()
    else:
        results = {tool: install_commands(tool)}
    for t, paths in results.items():
        click.secho(f"{t}: installed {', '.join(p.name for p in paths)} in {paths[0].parent}", fg="green")
