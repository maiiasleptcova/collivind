"""The overview's clustering is plain JS with no DOM, so node checks it directly.
Skipped where node is not installed; the Python side has no copy of this logic.
The module's source is inlined rather than imported, so any node that runs ES
modules from -e will do, not only the ones that import a bare .js file as ESM."""

import shutil
import subprocess

import pytest

from collivind.web.server import STATIC

NODE = shutil.which("node")

SCRIPT = """
const assert = (ok, msg) => { if (!ok) { console.error(msg); process.exit(1); } };

// 30 memories about "hub", every third also about "side", one about nothing
const mems = Array.from({ length: 31 }, (_, i) => ({
  id: `m${i}`, kind: "memory", project_id: i < 20 ? "a" : "b", category: i % 2 ? "fact" : "error",
}));
const edges = [];
for (let i = 0; i < 30; i++) {
  edges.push({ source: `m${i}`, target: "hub", type: "ABOUT" });
  if (i % 3 === 0) edges.push({ source: `m${i}`, target: "side", type: "ABOUT" });
}
const ents = new Map([["hub", { id: "hub", label: "Hub" }], ["side", { id: "side", label: "Side" }]]);
const links = entityLinks(new Set(mems.map((m) => m.id)), new Set(ents.keys()), edges);

const top = entityClusters(mems, links, ents, "a");
const count = Object.fromEntries(top.nodes.map((n) => [n.key, n.count]));
assert(count.hub === 30 && count.side === 10 && count[OTHER] === 1, "entity counts " + JSON.stringify(count));
assert(top.edges.length === 1 && top.edges[0].weight === 10, "co-occurrence link");

// opening hub: side is the only sub-bubble, plus the rest "only about hub"
const { members, opened } = membersAt(["hub"], mems, links);
assert(members.length === 30 && opened.has("hub"), "members of hub");
const sub = entityClusters(members, links, ents, "a", opened);
assert(sub.nodes.every((n) => n.key !== "hub"), "opened entity is not its own sub-bubble");
assert(membersAt(["hub", OTHER], mems, links).members.length === 20, "only-hub members");

// drawn view leaves the opened entity out, keeps shared ones
const ids = openIds(members, links, opened);
assert(!ids.has("hub") && ids.has("side") && ids.size === 31, "open ids");

// memories that share nothing left to split by fall back to their categories
const onlyHub = membersAt(["hub", OTHER], mems, links).members;
const cats = categoryClusters(onlyHub, "a");
assert(cats.nodes.length === 2 && cats.nodes.reduce((t, n) => t + n.count, 0) === 20, "category bubbles");
const fact = cats.nodes.find((n) => n.label === "fact");
assert(membersAt(["hub", OTHER, fact.key], mems, links).members.length === fact.count, "category path");

const projects = projectClusters(mems, links, (p) => p.toUpperCase());
assert(projects.nodes.length === 2 && projects.edges.length === 1, "project bubbles share hub");

// packed bubbles never overlap
const packed = pack(Array.from({ length: 40 }, (_, i) => ({ id: `c${i}`, count: (i * 37) % 500 + 1 })));
for (const a of packed) for (const b of packed) {
  if (a !== b) assert(Math.hypot(a.x - b.x, a.y - b.y) >= a.r + b.r, `overlap ${a.id} ${b.id}`);
}
console.log("ok");
"""


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_overview_clustering():
    script = (STATIC / "overview.js").read_text() + SCRIPT
    out = subprocess.run([NODE, "--input-type=module", "-e", script], capture_output=True, text=True, timeout=30)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "ok"
