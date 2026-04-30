import os
import json
import click
from pathlib import Path

EXTRACTION_PROMPT = """
<collivind_extraction>
You are connected to the Collivind memory graph. It is time to extract and save important knowledge from this session so far.
Please use the `collivind_batch_add` tool to store the following:
1. Noteworthy facts, decisions, patterns, errors, or architecture choices you have learned or made.
2. Ensure each memory is self-contained and makes sense without the conversation context.
3. Classify each memory into one of: fact, decision, pattern, error, architecture, preference, snippet.
4. Identify all relevant entities (project, file, service, concept, person, library, tool).
5. Identify any relationships between entities or to previous memories if applicable.
6. Skip trivial, common-sense, or highly ephemeral facts. Focus on knowledge valuable for future sessions.

Once you have successfully invoked the tool and saved the memories, acknowledge it briefly and continue the conversation normally.
</collivind_extraction>
"""

PRECOMPACT_PROMPT = """
<collivind_urgent_extraction>
WARNING: Your context window is about to be compressed or cleared!
You MUST extract ALL relevant facts, context, architecture details, and project state from the current session immediately.
Use the `collivind_batch_add` tool to store everything that might be useful when you resume.
- Be aggressive in saving context.
- Classify correctly (fact, decision, pattern, error, architecture, preference, snippet).
- Extract entities and relationships.

Save the data using the tool right now.
</collivind_urgent_extraction>
"""

def get_state_file() -> Path:
    return Path.home() / ".collivind" / "hook_state.json"

def read_state() -> dict:
    state_file = get_state_file()
    if state_file.exists():
        try:
            with open(state_file, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {"message_count": 0}

def write_state(state: dict):
    state_file = get_state_file()
    state_file.parent.mkdir(parents=True, exist_ok=True)
    with open(state_file, "w") as f:
        json.dump(state, f)

@click.group()
def hook():
    """Claude Code hook commands."""
    pass

@hook.command()
@click.option('--threshold', default=15, help="Number of messages before extraction")
def stop(threshold):
    """Periodic stop hook."""
    state = read_state()
    count = state.get("message_count", 0) + 1
    
    if count >= threshold:
        click.echo(EXTRACTION_PROMPT)
        state["message_count"] = 0
    else:
        state["message_count"] = count
        
    write_state(state)

@hook.command()
def precompact():
    """Urgent pre-compact extraction hook."""
    click.echo(PRECOMPACT_PROMPT)
    # Reset count since we are extracting now
    state = read_state()
    state["message_count"] = 0
    write_state(state)
