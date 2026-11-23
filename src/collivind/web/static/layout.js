// One step of a force layout: Barnes-Hut repulsion (O(n log n)), springs on
// edges, and a weak pull to the origin so disconnected islands stay in view.
// The same model as d3-force's manyBody/link/x/y, at a fraction of the size.

const THETA2 = 0.81;        // Barnes-Hut opening criterion squared (theta 0.9)
const CHARGE = -38;
const LINK_DIST = 34;
const GRAVITY = 0.035;
const FRICTION = 0.58;
const MAX_DEPTH = 24;

function quad(x0, y0, size) { return { x0, y0, size, point: null, kids: null, mass: 0, cx: 0, cy: 0 }; }

function insert(q, p, depth) {
  if (!q.kids && !q.point) { q.point = p; return; }
  if (!q.kids) {
    if (depth > MAX_DEPTH) return; // coincident points: the first one stands in for both
    const old = q.point; q.point = null; q.kids = [null, null, null, null];
    place(q, old, depth);
  }
  place(q, p, depth);
}

function place(q, p, depth) {
  const h = q.size / 2;
  const i = (p.x >= q.x0 + h ? 1 : 0) + (p.y >= q.y0 + h ? 2 : 0);
  if (!q.kids[i]) q.kids[i] = quad(q.x0 + (i & 1 ? h : 0), q.y0 + (i & 2 ? h : 0), h);
  insert(q.kids[i], p, depth + 1);
}

function accumulate(q) {
  if (!q) return;
  if (!q.kids) {
    if (q.point) { q.mass = 1; q.cx = q.point.x; q.cy = q.point.y; }
    return;
  }
  let m = 0, x = 0, y = 0;
  for (const k of q.kids) {
    if (!k) continue;
    accumulate(k);
    m += k.mass; x += k.cx * k.mass; y += k.cy * k.mass;
  }
  q.mass = m; q.cx = m ? x / m : 0; q.cy = m ? y / m : 0;
}

function repel(q, p, strength) {
  if (!q || !q.mass || (!q.kids && q.point === p)) return;
  const dx = q.cx - p.x, dy = q.cy - p.y;
  let d2 = dx * dx + dy * dy;
  if (!q.kids || (q.size * q.size) / THETA2 < d2) {
    if (d2 < 1) d2 = 1;
    const w = (strength * q.mass) / d2;
    p.vx += dx * w; p.vy += dy * w;
    return;
  }
  for (const k of q.kids) repel(k, p, strength);
}

export function buildTree(nodes) {
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const n of nodes) { x0 = Math.min(x0, n.x); y0 = Math.min(y0, n.y); x1 = Math.max(x1, n.x); y1 = Math.max(y1, n.y); }
  const root = quad(x0, y0, Math.max(x1 - x0, y1 - y0, 1) * 1.0001);
  for (const n of nodes) insert(root, n, 0);
  accumulate(root);
  return root;
}

export function layoutTick(nodes, links, alpha, pinned) {
  if (!nodes.length) return;
  const root = buildTree(nodes);
  const charge = CHARGE * alpha;
  for (const n of nodes) repel(root, n, charge);

  for (const l of links) {
    const s = l.source, t = l.target;
    let dx = t.x + t.vx - s.x - s.vx, dy = t.y + t.vy - s.y - s.vy;
    const len = Math.hypot(dx, dy) || 1;
    // stiffer for low-degree ends, so hubs do not get yanked around
    const strength = 1 / Math.min(s.degree || 1, t.degree || 1);
    const f = ((len - LINK_DIST) / len) * alpha * strength;
    dx *= f; dy *= f;
    const bias = (s.degree || 1) / ((s.degree || 1) + (t.degree || 1));
    t.vx -= dx * bias; t.vy -= dy * bias;
    s.vx += dx * (1 - bias); s.vy += dy * (1 - bias);
  }

  for (const n of nodes) {
    n.vx -= n.x * GRAVITY * alpha; n.vy -= n.y * GRAVITY * alpha;
    if (n === pinned) { n.vx = n.vy = 0; continue; }
    n.vx *= FRICTION; n.vy *= FRICTION;
    n.x += n.vx; n.y += n.vy;
  }
}
