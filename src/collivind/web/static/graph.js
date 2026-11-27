// Force-directed graph on a single <canvas>. Written here rather than vendored:
// a velocity-Verlet layout with Barnes-Hut repulsion is ~150 lines, and one
// canvas (not an element per node) keeps a few thousand nodes interactive.

import { layoutTick } from "./layout.js";

const MIN_K = 0.04, MAX_K = 8;
const EDGE_STYLE = {
  ABOUT: "entity", MENTIONS: "entity",
  CONTRADICTS: "contradicts", SUPERSEDES: "supersedes",
  // cause and fix: drawn heavier, coloured and always with an arrow, so a chain reads at a glance
  RESOLVES: "resolves", CAUSED_BY: "causes",
  // how entities fit together: the structural edges, drawn heavier
  USES: "relation", PART_OF: "relation", IMPLEMENTS: "relation", INTERACTS_WITH: "relation", RELATED_TO: "relation",
  SUMMARIZES: "summary",
  TOPIC: "summary", // the wiki's topic map: the reason is in the reader, not on the line
};
const isSummary = (d) => d.kind === "memory" && d.category === "summary";
const LABEL_BUDGET = 220;
const DIRECTED = new Set(["supersedes", "memory", "resolves", "causes"]);
const HOVER_SLOP = 6; // px around a node that still counts as on it
const EDGE_LABEL_BUDGET = 24;

const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

export class GraphView {
  // contextLabels (the wiki's topic map, after OpenWiki's visualizer): no
  // labels at rest, only the hovered node, or the selected one and its neighbours.
  constructor(canvas, { onSelect, onConnect, colorOf, contextLabels = false, edgeLabels = true }) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.onSelect = onSelect;
    this.onConnect = onConnect;
    this.colorOf = colorOf;
    this.contextLabels = contextLabels;
    this.edgeLabels = edgeLabels;
    this.byId = new Map();      // every node ever loaded, keeps positions across reloads
    this.nodes = [];            // visible
    this.links = [];            // visible, with node refs
    this.adj = new Map();
    this.k = 1; this.tx = 0; this.ty = 0;
    this.alpha = 0;
    this.hover = null; this.selected = null; this.highlight = null; this.meanings = {};
    this.userMoved = false;
    this.litAll = false;        // true: the selection does not dim the rest (the local graph)
    this.pointer = null;
    this.reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
    this.gen = 0; this.inFlight = false;
    try {
      this.worker = new Worker("/layout-worker.js", { type: "module" });
      this.worker.onmessage = (e) => this.onLayout(e.data);
      this.worker.onerror = () => { this.worker = null; this.inFlight = false; }; // tick inline instead
    } catch { this.worker = null; }
    this.readTheme();
    this.resize();
    new ResizeObserver(() => this.resize()).observe(canvas);
    this.bind();
    this.loop = this.loop.bind(this);
    requestAnimationFrame(this.loop);
  }

  readTheme() {
    this.theme = {
      bg: cssVar("--canvas"), ink: cssVar("--ink"), steel: cssVar("--steel"), line: cssVar("--edge"),
      entityEdge: cssVar("--edge-soft"), danger: cssVar("--danger"), entity: cssVar("--entity"),
      fix: cssVar("--green"), cause: cssVar("--cause"),
      halo: cssVar("--halo"), label: cssVar("--ink-2"), labelBg: cssVar("--canvas"),
    };
    this.dirty = true;
  }

  resize() {
    const r = this.canvas.getBoundingClientRect();
    if (!r.width || !r.height) return; // hidden view: keep the last real size
    const dpr = window.devicePixelRatio || 1;
    // keep the world point under the centre where it was (the panel opening
    // narrows the stage; the graph should not slide away underneath it)
    if (this.w) { this.tx += (r.width - this.w) / 2; this.ty += (r.height - this.h) / 2; }
    this.w = r.width; this.h = r.height; this.dpr = dpr;
    this.canvas.width = Math.max(1, Math.round(r.width * dpr));
    this.canvas.height = Math.max(1, Math.round(r.height * dpr));
    this.dirty = true;
  }

  // nodes/edges: the subset to draw. Positions persist by id, so filtering
  // does not throw the reader's mental map away.
  // fixed: nodes carry their own x/y/r (the overview's packed bubbles), so
  // nothing simulates and the view fits them at once.
  setGraph(nodes, edges, { reheat = 0.5, fixed = false } = {}) {
    const spread = 12 * Math.sqrt(nodes.length + 1);
    this.nodes = nodes.map((n, i) => {
      let p = this.byId.get(n.id);
      if (!p) {
        // phyllotaxis start: no two nodes coincide and the first ticks are calm
        const r = spread * Math.sqrt((i + 0.5) / (nodes.length + 1)), a = i * 2.39996;
        p = { x: r * Math.cos(a), y: r * Math.sin(a), vx: 0, vy: 0 };
        this.byId.set(n.id, p);
      }
      if (fixed) Object.assign(p, { x: n.x, y: n.y, vx: 0, vy: 0 });
      return Object.assign(p, { data: n, id: n.id, degree: 0 });
    });
    const visible = new Map(this.nodes.map((n) => [n.id, n]));
    this.links = [];
    this.adj = new Map(this.nodes.map((n) => [n.id, new Set()]));
    for (const e of edges) {
      const s = visible.get(e.source), t = visible.get(e.target);
      if (!s || !t) continue;
      this.links.push({ source: s, target: t, type: e.type, style: e.type === "CLUSTER" ? "cluster" : EDGE_STYLE[e.type] || "memory", data: e });
      s.degree++; t.degree++;
      this.adj.get(s.id).add(t.id); this.adj.get(t.id).add(s.id);
    }
    for (const n of this.nodes) {
      n.r0 = n.data.size !== undefined ? Math.min(16, 3 + Math.sqrt(n.data.size) * 0.9) // a topic, by its memories
        : n.data.kind === "cluster" ? n.data.r : n.data.kind === "entity" ? (n.data.hub ? 7 : 3.6)
        : isSummary(n.data) ? Math.min(18, 8 + Math.sqrt(n.data.members || 0) * 1.4)
        : Math.min(11, 4 + Math.sqrt(n.degree) * 1.1);
    }
    this.scaleNodes(true);
    if (this.selected && !visible.has(this.selected)) this.selected = null;
    // A click that re-centres leaves the pointer resting on a node of the new
    // graph; that hover would hold the fresh layout still. The next move re-hits.
    if (!fixed) this.hover = null;
    this.alpha = fixed ? 0 : Math.max(this.alpha, reheat);
    // Thousands of edges at full strength are a solid smear; the more there
    // are, the fainter each one, so structure shows through the density.
    this.edgeAlpha = Math.min(0.9, 45 / Math.sqrt(this.links.length || 1));
    this.dirty = true;
    this.gen++; this.inFlight = false;
    if (fixed) { this.userMoved = false; this.fit(60); return; }
    this.postGraph();
  }

  // Hand the worker this graph's positions; it answers one request at a time.
  postGraph() {
    if (!this.worker) return;
    const index = new Map(this.nodes.map((n, i) => [n.id, i]));
    const pos = new Float64Array(this.nodes.length * 2);
    this.nodes.forEach((n, i) => { pos[i * 2] = n.x; pos[i * 2 + 1] = n.y; });
    const links = new Int32Array(this.links.length * 2);
    this.links.forEach((l, i) => { links[i * 2] = index.get(l.source.id); links[i * 2 + 1] = index.get(l.target.id); });
    const degrees = Float64Array.from(this.nodes, (n) => n.degree);
    this.worker.postMessage({ type: "graph", gen: this.gen, pos, links, degrees }, [pos.buffer, links.buffer, degrees.buffer]);
  }

  onLayout({ gen, pos, alpha }) {
    if (gen !== this.gen) return;
    this.inFlight = false;
    const dragged = this.pointer?.dragging;
    this.nodes.forEach((n, i) => { if (n !== dragged) { n.x = pos[i * 2]; n.y = pos[i * 2 + 1]; } });
    // a drag may have reheated the layout while this tick was out
    this.alpha = this.pointer?.dragging ? Math.max(alpha, this.alpha) : alpha;
    if (!this.userMoved && (!this.reduced || this.alpha <= 0.004)) this.fit();
    this.dirty = true;
  }

  requestTick(ticks) {
    const d = this.pointer?.dragging;
    const pin = d ? { i: this.nodes.indexOf(d), x: d.x, y: d.y } : null;
    this.inFlight = true;
    this.worker.postMessage({ type: "tick", gen: this.gen, alpha: this.alpha, ticks, pin });
  }

  // Zoomed in past 2x, nodes stop growing on screen: a fitted 4-node graph is
  // zoomed in to spread the nodes apart, not to draw them as dinner plates.
  scaleNodes(force = false) {
    const f = Math.min(1, 2 / this.k);
    if (!force && f === this.nodeScale) return;
    this.nodeScale = f;
    for (const n of this.nodes) n.r = n.data.kind === "cluster" ? n.r0 : n.r0 * f;
  }

  setHighlight(ids) { this.highlight = ids && ids.size ? ids : null; this.dirty = true; }
  select(id, { center = false } = {}) {
    this.selected = id;
    if (center) this.centerOn(id);
    this.dirty = true;
  }
  neighbours(id) { return this.adj.get(id) || new Set(); }

  // ---- view transform -------------------------------------------------
  toWorld(sx, sy) { return [(sx - this.tx) / this.k, (sy - this.ty) / this.k]; }
  zoomAt(sx, sy, factor) {
    const k = Math.min(MAX_K, Math.max(MIN_K, this.k * factor));
    const [wx, wy] = this.toWorld(sx, sy);
    this.k = k; this.tx = sx - wx * k; this.ty = sy - wy * k;
    this.userMoved = true; this.dirty = true;
  }
  ensureVisible(id, margin = 60) {
    const n = this.byId.get(id);
    // while the view still fits itself it keeps every node in sight; moving it
    // here would mark it user-moved and freeze a half-settled layout off-fit
    if (!n || !this.w) return;
    if (!this.userMoved) { if (this.alpha <= 0.004) this.fit(); return; }
    const sx = n.x * this.k + this.tx, sy = n.y * this.k + this.ty;
    if (sx < margin || sx > this.w - margin || sy < margin || sy > this.h - margin) this.centerOn(id, false);
  }
  centerOn(id, zoom = true) {
    const n = this.byId.get(id);
    if (!n) return;
    if (zoom) this.k = Math.max(this.k, 1.4);
    this.tx = this.w / 2 - n.x * this.k; this.ty = this.h / 2 - n.y * this.k;
    this.userMoved = true; this.dirty = true;
  }
  // A small graph is zoomed in until its nodes sit far enough apart for
  // their labels (fixed 12px, drawn to the right of each node) not to collide;
  // room is kept on the right for those labels.
  fit(pad = 40) {
    if (!this.nodes.length || !this.w) return;
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    for (const n of this.nodes) { x0 = Math.min(x0, n.x); y0 = Math.min(y0, n.y); x1 = Math.max(x1, n.x); y1 = Math.max(y1, n.y); }
    const n = this.nodes.length, labelled = n <= 300 && this.nodes[0].data.kind !== "cluster";
    const right = labelled ? Math.min(240, this.w * 0.3) : pad;
    const cap = n <= 12 ? 6 : n <= 60 ? 4 : 2;
    const k = Math.min(MAX_K, Math.max(MIN_K, Math.min((this.w - pad - right) / (x1 - x0 || 1), (this.h - pad * 2) / (y1 - y0 || 1), cap)));
    this.k = k; this.tx = pad + (this.w - pad - right) / 2 - ((x0 + x1) / 2) * k; this.ty = this.h / 2 - ((y0 + y1) / 2) * k;
    this.dirty = true;
  }

  // ---- loop -----------------------------------------------------------
  loop() {
    // With reduced motion the layout still settles, it just is not watched:
    // many ticks per frame, then one fitted paint.
    const ticks = this.reduced ? 40 : this.nodes.length > 1500 ? 1 : 2;
    // Hold still while the pointer is on a node, so a click lands where the
    // node was drawn: the layout and the auto-fit both move nodes every frame.
    const held = this.hover && !this.pointer?.dragging && !this.pointer?.connecting;
    if (held) {
      // the layout resumes (alpha is kept) once the pointer leaves the node
    } else if (this.alpha > 0.004 && this.worker) {
      if (!this.inFlight) this.requestTick(ticks);
    } else if (this.alpha > 0.004) {
      for (let i = 0; i < ticks && this.alpha > 0.004; i++) {
        layoutTick(this.nodes, this.links, this.alpha, this.pointer?.dragging);
        this.alpha *= 0.985;
      }
      if (!this.userMoved && (!this.reduced || this.alpha <= 0.004)) this.fit();
      this.dirty = this.dirty || !this.reduced || this.alpha <= 0.004;
    }
    if (this.dirty) { this.draw(); this.dirty = false; }
    requestAnimationFrame(this.loop);
  }

  // The set the reader is looking at: hover beats selection beats search.
  focusSet() {
    const anchor = this.hover || (this.litAll ? null : this.selected);
    if (anchor) return new Set([anchor, ...this.neighbours(anchor)]);
    return this.highlight;
  }

  draw() {
    this.scaleNodes();
    const { ctx, theme: t, k, tx, ty, dpr } = this;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.fillStyle = t.bg;
    ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
    ctx.setTransform(dpr * k, 0, 0, dpr * k, dpr * tx, dpr * ty);
    const focus = this.focusSet();
    const lit = (n) => !focus || focus.has(n.id);

    // edges, batched by style and by lit/dim so there are a handful of strokes
    const styles = {
      entity: [t.entityEdge, 0.8, []], memory: [t.line, 1.1, []],
      contradicts: [t.danger, 1.3, [4, 3]], supersedes: [t.steel, 1.2, [1.5, 2.5]],
      relation: [t.ink, 1.5, []], summary: [t.steel, 1.2, []],
      resolves: [t.fix, 2.4, []], causes: [t.cause, 2.4, [6, 3]],
    };
    this.drawClusterLinks(focus);
    for (const pass of [false, true]) {
      for (const [style, [color, width, dash]] of Object.entries(styles)) {
        ctx.beginPath();
        let any = false;
        for (const l of this.links) {
          if (l.style !== style) continue;
          const a = this.hover || (this.litAll ? null : this.selected);
          const on = !focus || (focus.has(l.source.id) && focus.has(l.target.id) && (!a || l.source.id === a || l.target.id === a));
          if (on !== pass) continue;
          ctx.moveTo(l.source.x, l.source.y); ctx.lineTo(l.target.x, l.target.y); any = true;
        }
        if (!any) continue;
        ctx.globalAlpha = pass ? (focus ? 1 : this.edgeAlpha) : 0.12;
        ctx.strokeStyle = color; ctx.lineWidth = (pass && focus ? width * 1.6 : width) / Math.sqrt(k);
        ctx.setLineDash(dash.map((d) => d / Math.sqrt(k)));
        ctx.stroke();
      }
    }
    ctx.setLineDash([]);
    this.drawArrows(focus);

    // nodes
    for (const pass of [false, true]) {
      for (const n of this.nodes) {
        if (lit(n) !== pass) continue;
        ctx.globalAlpha = pass ? 1 : 0.18;
        this.drawNode(n);
      }
    }
    ctx.globalAlpha = 1;
    for (const id of [this.selected, this.hover]) {
      const n = id && this.byId.get(id);
      if (!n || !this.adj.has(id)) continue;
      ctx.beginPath(); ctx.arc(n.x, n.y, n.r + 4 / k, 0, Math.PI * 2);
      ctx.strokeStyle = t.ink; ctx.lineWidth = 2 / k; ctx.stroke();
    }
    if (this.pointer?.connecting) {
      const s = this.byId.get(this.pointer.connecting);
      ctx.beginPath(); ctx.moveTo(s.x, s.y); ctx.lineTo(this.pointer.wx, this.pointer.wy);
      ctx.strokeStyle = t.ink; ctx.lineWidth = 1.5 / k; ctx.setLineDash([5 / k, 4 / k]); ctx.stroke(); ctx.setLineDash([]);
    }
    this.drawLabels(focus);
    this.drawEdgeLabels();
  }

  setMeanings(meanings) { this.meanings = meanings || {}; this.dirty = true; }

  // What each edge of the hovered or selected node means, at its midpoint:
  // "uses", "is part of", plus the note when one was given. Only for that
  // node, so 13k edges never all speak at once.
  drawEdgeLabels() {
    const anchor = this.hover || this.selected;
    if (!this.edgeLabels || !anchor || !this.links.length || this.nodes[0]?.data.kind === "cluster") return;
    const { ctx, k, tx, ty, dpr, theme: t } = this;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.font = `500 11px "Instrument Sans", system-ui, sans-serif`;
    ctx.textBaseline = "middle"; ctx.textAlign = "center";
    // a summary's member links all say the same thing; the panel lists them instead
    // "is about" says nothing the square at its end does not; only its note is worth a label
    const links = this.links.filter((l) => l.style !== "summary" && (l.style !== "entity" || l.data?.note)
      && (l.source.id === anchor || l.target.id === anchor)).slice(0, EDGE_LABEL_BUDGET);
    const taken = [];
    for (const l of links) {
      const m = this.meanings?.[l.type];
      let text = m ? m.forward : l.type.toLowerCase().replace(/_/g, " ");
      if (l.data?.note) text += ` · ${l.data.note.length > 40 ? l.data.note.slice(0, 39) + "…" : l.data.note}`;
      const sx = ((l.source.x + l.target.x) / 2) * k + tx, sy = ((l.source.y + l.target.y) / 2) * k + ty;
      if (sx < -80 || sx > this.w + 80 || sy < -20 || sy > this.h + 20) continue;
      const w = ctx.measureText(text).width + 10;
      if (taken.some(([x, y, tw]) => Math.abs(sx - x) < (w + tw) / 2 && Math.abs(sy - y) < 16)) continue;
      taken.push([sx, sy, w]);
      ctx.globalAlpha = 0.92; ctx.fillStyle = t.labelBg;
      ctx.fillRect(sx - w / 2, sy - 8, w, 16);
      ctx.globalAlpha = 1;
      ctx.fillStyle = l.style === "resolves" ? t.fix : l.style === "causes" ? t.cause : l.style === "relation" ? t.ink : t.label;
      ctx.fillText(text, sx, sy);
    }
    ctx.textAlign = "start";
  }

  drawClusterLinks(focus) {
    const { ctx, k, theme: t } = this;
    ctx.strokeStyle = t.line;
    for (const l of this.links) {
      if (l.style !== "cluster") continue;
      const on = !focus || (focus.has(l.source.id) && focus.has(l.target.id));
      ctx.globalAlpha = on ? (focus ? 0.7 : 0.3) : 0.06;
      ctx.lineWidth = (0.75 + Math.log2(l.data.weight || 1) * 0.6) / k;
      ctx.beginPath(); ctx.moveTo(l.source.x, l.source.y); ctx.lineTo(l.target.x, l.target.y); ctx.stroke();
    }
    ctx.globalAlpha = 1;
  }

  drawNode(n) {
    const { ctx, theme: t } = this;
    const d = n.data;
    if (d.kind === "cluster") {
      const color = d.level === "entity" && d.type ? t.entity : this.colorOf(d.project_id);
      const a = ctx.globalAlpha;
      ctx.beginPath(); ctx.arc(n.x, n.y, n.r, 0, Math.PI * 2);
      ctx.fillStyle = t.bg; ctx.fill();
      ctx.globalAlpha = a * 0.2; ctx.fillStyle = color; ctx.fill();
      ctx.globalAlpha = a; ctx.strokeStyle = color; ctx.lineWidth = 1.5 / this.k; ctx.stroke();
      return;
    }
    if (d.kind === "entity") {
      const s = n.r * 1.7;
      ctx.fillStyle = t.entity;
      if (d.hub) {
        // a collapsed hub: an outlined square, its count in the label
        ctx.fillStyle = t.bg; ctx.fillRect(n.x - s / 2, n.y - s / 2, s, s);
        ctx.strokeStyle = t.entity; ctx.lineWidth = 2 / this.k; ctx.setLineDash([3 / this.k, 2 / this.k]);
        ctx.strokeRect(n.x - s / 2, n.y - s / 2, s, s); ctx.setLineDash([]);
        return;
      }
      ctx.fillRect(n.x - s / 2, n.y - s / 2, s, s);
      return;
    }
    const color = this.colorOf(d.project_id);
    if (isSummary(d)) {
      // a hexagon with a ring: a summary stands for a group, not one fact
      ctx.beginPath();
      for (let i = 0; i < 6; i++) {
        const a = Math.PI / 6 + (i * Math.PI) / 3;
        ctx[i ? "lineTo" : "moveTo"](n.x + n.r * Math.cos(a), n.y + n.r * Math.sin(a));
      }
      ctx.closePath();
      ctx.fillStyle = d.superseded ? t.bg : color; ctx.fill();
      ctx.strokeStyle = t.ink; ctx.lineWidth = 2 / this.k; ctx.stroke();
      return;
    }
    ctx.beginPath(); ctx.arc(n.x, n.y, n.r, 0, Math.PI * 2);
    if (d.superseded) {
      // superseded versions are drawn hollow: history, not current knowledge
      ctx.fillStyle = t.bg; ctx.fill();
      ctx.strokeStyle = color; ctx.lineWidth = 1.6 / this.k; ctx.stroke();
    } else {
      ctx.fillStyle = color; ctx.fill();
    }
  }

  drawArrows(focus) {
    const { ctx, k, theme: t } = this;
    for (const l of this.links) {
      if (!DIRECTED.has(l.style)) continue;
      // plain arrows only once zoomed in; cause and fix always show their direction
      const strong = l.style === "resolves" || l.style === "causes";
      if (!strong && k <= 0.7) continue;
      if (focus && !(focus.has(l.source.id) && focus.has(l.target.id))) continue;
      ctx.fillStyle = l.style === "resolves" ? t.fix : l.style === "causes" ? t.cause : t.steel;
      const dx = l.target.x - l.source.x, dy = l.target.y - l.source.y, len = Math.hypot(dx, dy) || 1;
      const ux = dx / len, uy = dy / len, tipX = l.target.x - ux * (l.target.r + 2 / k), tipY = l.target.y - uy * (l.target.r + 2 / k);
      const a = (strong ? 10 : 6) / k, b = (strong ? 5 : 3) / k;
      ctx.beginPath(); ctx.moveTo(tipX, tipY);
      ctx.lineTo(tipX - ux * a - uy * b, tipY - uy * a + ux * b);
      ctx.lineTo(tipX - ux * a + uy * b, tipY - uy * a - ux * b);
      ctx.fill();
    }
  }

  // Labels in screen space so they stay crisp; only as many as can be read.
  drawLabels(focus) {
    const { ctx, k, tx, ty, dpr, theme: t } = this;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.font = `500 12px "Instrument Sans", system-ui, sans-serif`;
    ctx.textBaseline = "middle";
    const anchor = this.hover || this.selected;
    let pool;
    if (this.nodes.length && this.nodes[0].data.kind === "cluster") { this.drawClusterLabels(anchor); return; }
    if (this.contextLabels) {
      const shown = new Set([this.hover, this.selected, ...(this.selected ? this.neighbours(this.selected) : [])]);
      pool = this.nodes.filter((n) => shown.has(n.id));
    } else if (focus) pool = this.nodes.filter((n) => focus.has(n.id));
    else if (k >= 1.1 || this.nodes.length <= 300) pool = this.nodes; // small sets: label what fits at any zoom
    else pool = [];
    pool = pool
      .map((n) => ({ n, sx: n.x * k + tx, sy: n.y * k + ty }))
      .filter((p) => p.sx > -50 && p.sx < this.w + 50 && p.sy > -20 && p.sy < this.h + 20)
      .sort((a, b) => (b.n.id === anchor) - (a.n.id === anchor) || b.n.degree - a.n.degree)
      .slice(0, LABEL_BUDGET);
    ctx.textAlign = "center";
    ctx.font = `700 11px "Instrument Sans", system-ui, sans-serif`;
    for (const n of this.nodes) {
      if (!isSummary(n.data) || n.r * k < 8) continue;
      ctx.fillStyle = t.bg;
      ctx.fillText(String(n.data.members || 0), n.x * k + tx, n.y * k + ty);
    }
    ctx.textAlign = "start";
    ctx.font = `500 12px "Instrument Sans", system-ui, sans-serif`;
    const taken = [];
    for (const { n, sx, sy } of pool) {
      let text = n.data.label || n.id;
      if (text.length > 48) text = text.slice(0, 47) + "…";
      if (n.data.hub) text += ` · ${n.data.count} memories, collapsed`;
      const w = ctx.measureText(text).width, x = sx + n.r * k + 6, y = sy;
      // skip a label that would overprint one already placed
      if (n.id !== anchor && taken.some((r) => x < r[2] && x + w > r[0] && Math.abs(y - r[1]) < 14)) continue;
      taken.push([x, y, x + w]);
      ctx.fillStyle = t.labelBg; ctx.globalAlpha = 0.85;
      ctx.fillRect(x - 3, y - 8, w + 6, 16);
      ctx.globalAlpha = 1; ctx.fillStyle = n.id === anchor ? t.ink : t.label;
      ctx.fillText(text, x, y);
    }
  }

  // Every bubble gets its name and count; biggest first, so on a collision
  // the smaller one gives way.
  drawClusterLabels(anchor) {
    const { ctx, k, tx, ty, theme: t } = this;
    ctx.textAlign = "center";
    const taken = [];
    for (const n of [...this.nodes].sort((a, b) => (b.id === anchor) - (a.id === anchor) || b.data.count - a.data.count)) {
      const sx = n.x * k + tx, sy = n.y * k + ty, sr = n.r * k;
      let text = n.data.label || n.id;
      if (text.length > 30) text = text.slice(0, 29) + "…";
      const inside = sr > 34;
      const y = inside ? sy - 7 : sy + sr + 11;
      ctx.font = `600 12px "Instrument Sans", system-ui, sans-serif`;
      const w = Math.max(ctx.measureText(text).width, 30);
      if (n.id !== anchor && taken.some((r) => Math.abs(sx - r[0]) < (w + r[2]) / 2 && Math.abs(y - r[1]) < 26)) continue;
      taken.push([sx, y, w]);
      if (!inside) { ctx.fillStyle = t.labelBg; ctx.globalAlpha = 0.85; ctx.fillRect(sx - w / 2 - 3, y - 8, w + 6, 30); ctx.globalAlpha = 1; }
      ctx.fillStyle = n.id === anchor ? t.ink : t.label;
      ctx.fillText(text, sx, y);
      ctx.font = `500 11px "Instrument Sans", system-ui, sans-serif`;
      ctx.fillStyle = t.steel;
      ctx.fillText(`${n.data.count.toLocaleString()} ${n.data.count === 1 ? "memory" : "memories"}`, sx, y + 14);
    }
    ctx.textAlign = "start";
  }

  // ---- pointer --------------------------------------------------------
  hit(sx, sy) {
    const [wx, wy] = this.toWorld(sx, sy);
    let best = null, bestD = Infinity;
    const slop = HOVER_SLOP / this.k;
    for (const n of this.nodes) {
      const d = Math.hypot(n.x - wx, n.y - wy);
      if (d < n.r + slop && d < bestD) { best = n; bestD = d; }
    }
    return best;
  }

  bind() {
    const c = this.canvas;
    const pos = (e) => { const r = c.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; };
    c.addEventListener("pointerdown", (e) => {
      const [sx, sy] = pos(e);
      const n = this.hit(sx, sy);
      c.setPointerCapture(e.pointerId);
      this.pointer = { sx, sy, x0: sx, y0: sy, node: n?.id, moved: false,
        connecting: n && (e.shiftKey || e.altKey) ? n.id : null, dragging: null };
    });
    c.addEventListener("pointermove", (e) => {
      const [sx, sy] = pos(e);
      const p = this.pointer;
      if (!p) {
        const n = this.hit(sx, sy);
        const id = n ? n.id : null;
        if (id !== this.hover) { this.hover = id; c.style.cursor = id ? "pointer" : "grab"; this.dirty = true; }
        return;
      }
      if (Math.hypot(sx - p.x0, sy - p.y0) > 3) p.moved = true;
      const [wx, wy] = this.toWorld(sx, sy);
      p.wx = wx; p.wy = wy;
      if (p.connecting) {
        const over = this.hit(sx, sy);
        this.hover = over && over.id !== p.connecting ? over.id : null;
      } else if (p.node && p.moved && this.byId.get(p.node).data.kind !== "cluster") {
        const n = this.byId.get(p.node);
        p.dragging = n; n.x = wx; n.y = wy; n.vx = n.vy = 0;
        this.alpha = Math.max(this.alpha, 0.12);
      } else if (p.moved) { // empty canvas, or a bubble: bubbles stay where they were packed
        this.tx += sx - p.sx; this.ty += sy - p.sy; this.userMoved = true;
        c.style.cursor = "grabbing";
      }
      p.sx = sx; p.sy = sy;
      this.dirty = true;
    });
    const end = (e) => {
      const p = this.pointer;
      this.pointer = null;
      c.style.cursor = this.hover ? "pointer" : "grab";
      if (!p) return;
      if (p.connecting) {
        const [sx, sy] = pos(e);
        const target = this.hit(sx, sy);
        if (target && target.id !== p.connecting) this.onConnect(p.connecting, target.id);
        this.hover = null;
      } else if (!p.moved) {
        this.onSelect(p.node || null);
      }
      this.dirty = true;
    };
    c.addEventListener("pointerup", end);
    c.addEventListener("pointercancel", () => { this.pointer = null; });
    c.addEventListener("pointerleave", () => { if (!this.pointer && this.hover) { this.hover = null; this.dirty = true; } });
    c.addEventListener("wheel", (e) => {
      e.preventDefault();
      const [sx, sy] = pos(e);
      this.zoomAt(sx, sy, Math.exp(-e.deltaY * (e.ctrlKey ? 0.01 : 0.0022)));
    }, { passive: false });
    c.addEventListener("dblclick", (e) => { const [sx, sy] = pos(e); if (!this.hit(sx, sy)) { this.userMoved = false; this.fit(); } });
    c.addEventListener("keydown", (e) => {
      const step = 60;
      const keys = {
        "+": () => this.zoomAt(this.w / 2, this.h / 2, 1.25), "=": () => this.zoomAt(this.w / 2, this.h / 2, 1.25),
        "-": () => this.zoomAt(this.w / 2, this.h / 2, 0.8), "0": () => { this.userMoved = false; this.fit(); },
        ArrowLeft: () => { this.tx += step; }, ArrowRight: () => { this.tx -= step; },
        ArrowUp: () => { this.ty += step; }, ArrowDown: () => { this.ty -= step; },
      };
      if (keys[e.key]) { e.preventDefault(); keys[e.key](); this.userMoved = true; this.dirty = true; }
    });
  }
}
