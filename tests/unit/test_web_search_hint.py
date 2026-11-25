"""Regression for "search needs Enter twice" (2026-11-25).

Cause: the first search after a start waits seconds for the embedding model.
Meanwhile the 120 ms text filter repainted the rail with "Nothing matches.
Press Enter to search by meaning instead.", so the reader pressed Enter again.
The rail's wording is plain JS with no DOM (static/search.js), checked with
node like the overview; skipped where node is not installed.
"""

import shutil
import subprocess

import pytest

from collivind.web.server import STATIC

NODE = shutil.which("node")

SCRIPT = """
const assert = (ok, msg) => { if (!ok) { console.error(msg); process.exit(1); } };

// typed, Enter pressed, the answer not back yet: the filter has text too
const pending = railHint({ search: null, searching: "docker exec", text: "docker exec" });
assert(pending.heading.startsWith("Searching by meaning"), "pending heading: " + pending.heading);
assert(!/Press Enter/.test(pending.empty), "pending must not ask for another Enter: " + pending.empty);

// a newer search in flight outranks the results of the last one
const again = railHint({ search: { query: "old" }, searching: "new", text: "new" });
assert(again.heading.includes("new") && !again.heading.includes("old"), "newer search: " + again.heading);

// once answered: results, and an empty answer says to rephrase, not to press Enter
const done = railHint({ search: { query: "docker exec" }, searching: null, text: "docker exec" });
assert(done.heading === "Search results for “docker exec”", "done heading: " + done.heading);
assert(!/Press Enter/.test(done.empty), "done hint: " + done.empty);

// only typing (no Enter yet) still offers the search
assert(/Press Enter/.test(railHint({ search: null, searching: null, text: "dock" }).empty), "typing hint");
assert(railHint({ search: null, searching: null, text: "" }) === null, "nothing typed");
console.log("ok");
"""


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_pending_search_does_not_ask_for_a_second_enter():
    script = (STATIC / "search.js").read_text() + SCRIPT
    out = subprocess.run([NODE, "--input-type=module", "-e", script], capture_output=True, text=True, timeout=30)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "ok"


def test_enter_cancels_the_pending_text_filter():
    """The other half of the fix lives in app.js: Enter clears the debounce timer before searching."""
    app = (STATIC / "app.js").read_text()
    submit = app[app.index('$("#search-form").onsubmit') :]
    assert submit.index("clearTimeout(timer)") < submit.index("runSearch(q)")
