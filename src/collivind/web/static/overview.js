// The overview: the store as a few dozen labelled bubbles instead of every
// node. Level 1 is projects (skipped when there is only one), level 2 the
// entities a project's memories are about, and opening one shows that
// entity's memories with the other entities they share. Pure functions over
// the /api/graph payload, already narrowed by the rail's filters. A bubble
// too big to draw node by node opens into bubbles of its own (the entities
// its memories share), so a hub with a thousand memories is two clicks deep,
// not a thousand dots.

export const TOP_ENTITIES = 40;
export const OTHER = "__other";
const STRONGEST = 3;      // links kept per bubble; more and the overview is a hairball again
const MIN_SHARED = 2;     // an entity opened alongside must be shared by this many memories
export const OPEN_LIMIT = 220; // more members than this and a bubble opens into bubbles

const pairKey = (a, b) => (a < b ? `${a}\u0000${b}` : `${b}\u0000${a}`);

// memory id -> entity ids it links to, over the edges the rail lets through
export function entityLinks(memIds, entityIds, edges) {
  const out = new Map();
  for (const e of edges) {
    for (const [m, x] of [[e.source, e.target], [e.target, e.source]]) {
      if (!memIds.has(m) || !entityIds.has(x)) continue;
      if (!out.has(m)) out.set(m, new Set());
      out.get(m).add(x);
    }
  }
  return out;
}

// Weighted links between bubbles that share members, each bubble keeping
// only its strongest few so the picture stays readable.
function strongest(weights) {
  const byNode = new Map();
  for (const [key, w] of weights) {
    const [a, b] = key.split("\u0000");
    for (const n of [a, b]) {
      if (!byNode.has(n)) byNode.set(n, []);
      byNode.get(n).push([key, w]);
    }
  }
  const keep = new Set();
  for (const list of byNode.values()) {
    list.sort((x, y) => y[1] - x[1]);
    for (const [key] of list.slice(0, STRONGEST)) keep.add(key);
  }
  return [...keep].map((key) => {
    const [source, target] = key.split("\u0000");
    return { source, target, type: "CLUSTER", weight: weights.get(key) };
  });
}

function addPairs(weights, ids) {
  const list = [...ids];
  for (let i = 0; i < list.length; i++) {
    for (let j = i + 1; j < list.length; j++) {
      const key = pairKey(list[i], list[j]);
      weights.set(key, (weights.get(key) || 0) + 1);
    }
  }
}

const cluster = (id, label, count, extra) => ({ id, kind: "cluster", label, count, ...extra });

// Level 1: one bubble per project, linked by the entities they share.
export function projectClusters(memories, links, nameOf) {
  const counts = new Map();
  for (const m of memories) counts.set(m.project_id, (counts.get(m.project_id) || 0) + 1);
  const projectsOf = new Map();
  for (const m of memories) {
    for (const x of links.get(m.id) || []) {
      if (!projectsOf.has(x)) projectsOf.set(x, new Set());
      projectsOf.get(x).add(`p:${m.project_id}`);
    }
  }
  const weights = new Map();
  for (const ps of projectsOf.values()) if (ps.size > 1) addPairs(weights, ps);
  const nodes = [...counts].map(([pid, n]) => cluster(`p:${pid}`, nameOf(pid), n, { level: "project", key: pid, project_id: pid }));
  return { nodes: pack(nodes), edges: strongest(weights) };
}

// Level 2: the entities most memories are about, plus one bubble for the rest.
// `exclude` are the entities already opened: every member links to them.
export function entityClusters(memories, links, entityById, projectId, exclude = new Set(), restLabel = "Not about a top entity") {
  const counts = new Map();
  for (const m of memories) for (const x of links.get(m.id) || []) if (!exclude.has(x)) counts.set(x, (counts.get(x) || 0) + 1);
  const top = [...counts].sort((a, b) => b[1] - a[1]).slice(0, TOP_ENTITIES);
  const topIds = new Set(top.map(([x]) => `e:${x}`));
  const weights = new Map();
  let rest = 0;
  for (const m of memories) {
    const mine = [...(links.get(m.id) || [])].map((x) => `e:${x}`).filter((x) => topIds.has(x));
    if (!mine.length) rest++;
    addPairs(weights, mine);
  }
  const nodes = top.map(([x, n]) => cluster(`e:${x}`, entityById.get(x)?.label || x, n,
    { level: "entity", key: x, type: entityById.get(x)?.type, project_id: projectId }));
  if (rest) nodes.push(cluster(`e:${OTHER}`, restLabel, rest, { level: "entity", key: OTHER, project_id: projectId }));
  return { nodes: pack(nodes), edges: strongest(weights) };
}

// Follow a path of opened bubbles down to the memories in the last one.
// OTHER at a step means "about none of the bubbles shown at that step".
export function membersAt(path, memories, links) {
  let members = memories;
  const opened = new Set();
  for (const key of path) {
    if (key === OTHER) {
      const top = topKeys(members, links, opened);
      members = members.filter((m) => ![...(links.get(m.id) || [])].some((x) => top.has(x)));
    } else {
      members = members.filter((m) => links.get(m.id)?.has(key));
      opened.add(key);
    }
  }
  return { members, opened };
}

function topKeys(memories, links, exclude) {
  const counts = new Map();
  for (const m of memories) for (const x of links.get(m.id) || []) if (!exclude.has(x)) counts.set(x, (counts.get(x) || 0) + 1);
  return new Set([...counts].sort((a, b) => b[1] - a[1]).slice(0, TOP_ENTITIES).map(([x]) => x));
}

// The ids to draw once a bubble is small enough: its memories and the
// entities they share. Opened entities are left out: every member links to
// them, so drawn they are one giant star.
export function openIds(members, links, opened) {
  const shared = new Map();
  for (const m of members) for (const x of links.get(m.id) || []) if (!opened.has(x)) shared.set(x, (shared.get(x) || 0) + 1);
  const ids = new Set(members.map((m) => m.id));
  for (const [x, n] of shared) if (n >= MIN_SHARED) ids.add(x);
  return ids;
}

export const bubbleRadius = (count) => 10 + Math.sqrt(count) * 2.4;

// Biggest bubble in the middle, the rest on an outward spiral at the first
// spot that overlaps nothing. Deterministic, so the overview never wobbles.
// ponytail: O(n^2) per spiral step; fine for the ~40 bubbles drawn here.
export function pack(nodes) {
  const sorted = [...nodes].sort((a, b) => b.count - a.count);
  const placed = [];
  for (const n of sorted) {
    const r = bubbleRadius(n.count);
    let x = 0, y = 0;
    for (let a = 0, d = 0; placed.some((p) => Math.hypot(p.x - x, p.y - y) < p.r + r + 22); a += 0.3, d += 1.5) {
      x = d * Math.cos(a); y = d * Math.sin(a);
    }
    placed.push({ ...n, x, y, r });
  }
  return placed;
}
