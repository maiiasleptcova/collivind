// Entry point: state, the three views (graph, list, projects), filters and search.

import { $, $$, api, esc, toast, showError, clearError, projectColor, projectName } from "./util.js";
import { CATEGORIES, card, editorMarkup, readEditor } from "./memory.js";
import { GraphView } from "./graph.js";
import { initPanel, openNode, closePanel, currentId, refreshPanel } from "./panel.js";
import { initProjects, renderProjects } from "./projects.js";
import { OPEN_LIMIT, OTHER, entityClusters, entityLinks, membersAt, openIds, projectClusters } from "./overview.js";

// Off until asked for: on a real store CONTRADICTS is ~40% of all edges, and
// drawn by default it paints the whole graph red.
const DEFAULT_OFF = new Set(["CONTRADICTS"]);
const DAY = 86400000;
const PAGE = 50;

const state = {
  view: "graph",
  graph: { nodes: [], edges: [], total_memories: 0, truncated: false },
  byId: new Map(),
  projects: [],
  meta: {},
  linkable: {},
  filters: { project: "", categories: new Set(CATEGORIES), entityTypes: null, showEntities: true, from: "", to: "", hideSuperseded: false },
  text: "",
  search: null,          // { query, results } after Enter
  selected: null,
  local: false,
  depth: 1,
  visible: [],
  list: [], editingId: null, creating: false,
  knownTypes: new Set(),
  mode: "overview",      // "overview": bubbles that open; "all": every loaded node
  path: [],              // opened overview bubbles, outermost first: entity ids, or OTHER
  edgeTypes: new Set(),  // relationship types drawn and walked
  knownEdgeTypes: new Set(),
  focusGraph: null,      // the neighbourhood of the search results, from the server
  page: { sort: "newest", group: "", offset: 0, total: 0 },
};

// What the graph view draws from: the search neighbourhood when there is one.
const shown = () => state.focusGraph || state.graph;

const colorOf = (projectId) => projectColor(projectId, state.meta);

const view = new GraphView($("#canvas"), {
  colorOf,
  onSelect: (id) => select(id),
  onConnect: (source, target) => { select(source); openNode(source, { linkTo: target }); },
});

initPanel({
  graph: () => ({ byId: state.byId, nodes: state.visible, edges: shown().edges }),
  linkable: () => state.linkable,
  projects: () => state.meta,
  colorOf,
  select: (id, opts) => select(id, opts),
  reload: () => loadGraph(),
  local: () => state.local,
  toggleLocal: () => { state.local = !state.local; applyFilters(); refreshPanel(); },
  confirmForget: (id) => confirmForget(id),
});

initProjects({
  rows: () => state.projects,
  reload: async () => { await loadProjects(); renderProjects(); view.dirty = true; },
  openInGraph: (id) => { setProject(id); setView("graph"); },
  openInList: (id) => { setProject(id); setView("list"); },
});

// ---- data -------------------------------------------------------------

async function loadGraph() {
  try {
    // Scoped server-side, so a store past the cap shows a project's older
    // memories too, not just its share of the newest few thousand.
    const project = state.filters.project;
    const g = await api(project ? `/api/graph?project=${encodeURIComponent(project)}` : "/api/graph");
    state.graph = g;
    for (const e of g.edges) {
      if (state.knownEdgeTypes.has(e.type)) continue;
      state.knownEdgeTypes.add(e.type);
      if (!DEFAULT_OFF.has(e.type)) state.edgeTypes.add(e.type);
    }
    renderEdgeTypes();
    renderSince();
    if (state.search) await refocus({ apply: false });
    rebuildById();
    // Types never seen before arrive switched on; only an untick hides one.
    const types = g.nodes.filter((n) => n.kind === "entity").map((n) => n.type);
    state.filters.entityTypes ??= new Set();
    for (const t of types) if (!state.knownTypes.has(t)) { state.knownTypes.add(t); state.filters.entityTypes.add(t); }
    renderEntityTypes();
    applyFilters();
    if (state.selected && !state.byId.has(state.selected)) select(null);
  } catch (e) { showError(e); }
}

function rebuildById() {
  state.byId = new Map([...state.graph.nodes, ...(state.focusGraph?.nodes || [])].map((n) => [n.id, n]));
}

async function loadProjects() {
  const { projects } = await api("/api/projects");
  state.projects = projects;
  state.meta = Object.fromEntries(projects.map((p) => [p.project_id, p]));
  const sel = $("#f-project");
  const current = state.filters.project;
  sel.innerHTML = `<option value="">All projects</option>` + projects.map((p) =>
    `<option value="${esc(p.project_id)}">${esc(projectName(p.project_id, state.meta))}</option>`).join("");
  sel.value = state.meta[current] ? current : "";
}

// ---- filtering --------------------------------------------------------

function memoryPasses(n, f) {
  if (f.project && n.project_id !== f.project) return false;
  if (!f.categories.has(n.category) && CATEGORIES.includes(n.category)) return false;
  if (f.hideSuperseded && n.superseded) return false;
  const day = (n.created_at || "").slice(0, 10);
  if (f.from && day < f.from) return false;
  if (f.to && day > f.to) return false;
  return true;
}

// The overview's inputs: memories that pass the rail, the entities the rail
// lets through, and which of those each memory is linked to.
function overviewInputs(nodes, memIds, edges) {
  const f = state.filters;
  const memories = nodes.filter((n) => memIds.has(n.id));
  const entityById = new Map(f.showEntities
    ? nodes.filter((n) => n.kind === "entity" && f.entityTypes?.has(n.type)).map((n) => [n.id, n]) : []);
  const links = entityLinks(memIds, new Set(entityById.keys()), edges);
  const projects = new Set(memories.map((m) => m.project_id));
  const project = f.project || (projects.size === 1 ? [...projects][0] : "");
  return { memories, entityById, links, byProject: projects.size > 1, project };
}

const crumbLabel = (key, i) => key !== OTHER ? state.byId.get(key)?.label || key
  : i ? `Only ${state.byId.get(state.path[i - 1])?.label || "that"}` : "Not about a top entity";

// The bubbles for where the reader is; null when the opened bubble is small
// enough to draw node by node (or would only split into one bubble again).
function overviewBubbles(o) {
  if (o.byProject) return projectClusters(o.memories, o.links, (pid) => projectName(pid, state.meta));
  const { members, opened } = membersAt(state.path, o.memories, o.links);
  const last = state.path[state.path.length - 1];
  if (state.path.length && (members.length <= OPEN_LIMIT || last === OTHER)) return null;
  const restLabel = state.path.length ? `Only ${crumbLabel(last, state.path.length - 1)}` : "Not about a top entity";
  const bubbles = entityClusters(members, o.links, o.entityById, o.project, opened, restLabel);
  return bubbles.nodes.length > 1 || !state.path.length ? bubbles : null;
}

function renderOverview(o, bubbles) {
  const { nodes, edges } = bubbles;
  state.visible = [];
  view.select(null);
  view.setGraph(nodes, edges, { fixed: true });
  state.overview = { clusters: nodes.length, memories: o.memories.length, byProject: o.byProject };
}

function applyFilters() {
  const f = state.filters;
  const { nodes } = shown();
  const edges = shown().edges.filter((e) => state.edgeTypes.has(e.type));
  const memIds = new Set(nodes.filter((n) => n.kind === "memory" && memoryPasses(n, f)).map((n) => n.id));
  state.overview = null;
  const overview = state.mode === "overview" && !state.focusGraph;
  const o = overview ? overviewInputs(nodes, memIds, edges) : null;
  const bubbles = o && overviewBubbles(o);
  if (bubbles) {
    renderOverview(o, bubbles);
    $("#stage").classList.toggle("is-local", false);
    return finishRender();
  }
  const reached = new Set();
  if (f.showEntities) {
    for (const e of edges) {
      if (memIds.has(e.source)) reached.add(e.target);
      if (memIds.has(e.target)) reached.add(e.source);
    }
  }
  let ids = new Set(nodes.filter((n) => memIds.has(n.id) ||
    (n.kind === "entity" && reached.has(n.id) && f.entityTypes?.has(n.type))).map((n) => n.id));

  // An opened bubble: its memories and the entities they share, nothing else.
  if (o) {
    const { members, opened } = membersAt(state.path, o.memories, o.links);
    const open = openIds(members, o.links, opened);
    ids = new Set([...ids].filter((id) => open.has(id)));
  }

  // Local graph: n hops around the selection, over what the filters allow.
  if (state.local && state.selected && ids.has(state.selected)) {
    const near = new Set([state.selected]);
    let frontier = new Set([state.selected]);
    for (let hop = 0; hop < state.depth; hop++) {
      const next = new Set();
      for (const e of edges) {
        for (const [a, b] of [[e.source, e.target], [e.target, e.source]]) {
          if (frontier.has(a) && ids.has(b) && !near.has(b)) { near.add(b); next.add(b); }
        }
      }
      frontier = next;
    }
    ids = near;
  }
  $("#stage").classList.toggle("is-local", state.local);

  state.visible = nodes.filter((n) => ids.has(n.id));
  view.setGraph(state.visible, edges, { reheat: 0.4 });
  if (state.local || state.path.length || state.focusGraph) view.userMoved = false;
  finishRender();
}

function finishRender() {
  updateHighlight();
  renderNote();
  renderCrumbs();
}

// ---- overview navigation ------------------------------------------------

function setMode(mode) {
  state.mode = mode;
  state.path = [];
  state.page = { ...state.page, offset: 0 };
  for (const b of $$("[data-mode]")) b.setAttribute("aria-pressed", String(b.dataset.mode === mode));
  view.userMoved = false;
  applyFilters();
}

function openBubble(node) {
  if (node.level === "project") return setProject(node.key);
  state.path = [...state.path, node.key];
  view.userMoved = false;
  applyFilters();
}

function renderCrumbs() {
  const nav = $("#crumbs");
  const crumbs = [];
  if (state.focusGraph) {
    crumbs.push([`Around “${state.search.query.slice(0, 40)}”`, null], ["Clear search", "clear"]);
  } else if (state.mode === "overview") {
    const multi = state.projects.filter((p) => p.memories).length > 1;
    if (multi) crumbs.push(["All projects", state.filters.project || state.path.length ? "root" : null]);
    if (state.filters.project) crumbs.push([projectName(state.filters.project, state.meta), state.path.length ? "0" : null]);
    else if (!multi) crumbs.push(["Overview", state.path.length ? "0" : null]);
    state.path.forEach((key, i) => crumbs.push([crumbLabel(key, i), i < state.path.length - 1 ? String(i + 1) : null]));
    if (!crumbs.length) crumbs.push(["Overview", null]);
  }
  nav.innerHTML = crumbs.map(([label, action], i) => (i ? `<span class="sep" aria-hidden="true">›</span>` : "") +
    (action ? `<button data-crumb="${action}">${esc(label)}</button>` : `<span aria-current="page">${esc(label)}</span>`)).join("");
  $$("[data-crumb]", nav).forEach((b) => b.onclick = () => {
    const action = b.dataset.crumb;
    if (action === "clear") return clearSearch();
    if (action === "root") return setProject("");
    state.path = state.path.slice(0, Number(action)); view.userMoved = false; applyFilters();
  });
}

function renderNote() {
  const g = shown();
  if (state.overview) {
    const o = state.overview;
    $("#graph-note").textContent = `Overview: ${o.memories} memories in ${o.clusters} ${o.byProject ? "projects" : "clusters"}. ` +
      `Click a bubble to open it${g.truncated ? `; only the newest ${g.limit} memories are loaded` : ""}.`;
    $("#graph-empty").hidden = !!g.total_memories;
    return;
  }
  const shownMem = state.visible.filter((n) => n.kind === "memory").length;
  const shownEnt = state.visible.length - shownMem;
  const parts = [`Showing ${shownMem} of ${g.total_memories} memories and ${shownEnt} entities.`];
  if (state.focusGraph) parts.push(`Search results and what is within ${state.depth} hop${state.depth > 1 ? "s" : ""} of them.`);
  else if (g.truncated) {
    parts.push(state.filters.project
      ? `Only this project's newest ${g.limit} memories are loaded.`
      : `Only the newest ${g.limit} memories are loaded; pick a project to see the rest.`);
  }
  if (state.local) parts.push(`Local graph, ${state.depth} hop${state.depth > 1 ? "s" : ""} around the selection.`);
  $("#graph-note").textContent = parts.join(" ");
  const empty = !g.total_memories;
  $("#graph-empty").hidden = !empty;
}

// Text in the box filters labels live; Enter asks the store by meaning.
function updateHighlight() {
  const results = $("#results");
  let hits = [];
  let heading = "";
  if (state.search) {
    hits = state.search.results.map((m) => ({ id: m.id, label: m.summary || m.content || m.label, sub: m.score === undefined ? m.kind || "" : `match ${Number(m.score).toFixed(2)}`, kind: "memory", category: m.category }));
    heading = `Search results for “${state.search.query}”`;
  } else if (state.text) {
    const q = state.text.toLowerCase();
    // In the overview nothing is drawn node by node, so match everything loaded.
    hits = (state.visible.length ? state.visible : shown().nodes).filter((n) => n.label.toLowerCase().includes(q)).slice(0, 60)
      .map((n) => ({ id: n.id, label: n.label, sub: n.kind === "entity" ? n.type : n.category, kind: n.kind }));
    heading = `Matching “${state.text}”`;
  }
  view.setHighlight(hits.length ? new Set(hits.map((h) => h.id)) : null);
  if (!heading) { results.innerHTML = ""; return; }
  results.innerHTML = `<h3>${esc(heading)}</h3>` + (hits.length
    ? `<ul>${hits.map((h) => `<li><button class="result" data-node="${esc(h.id)}">
        <span class="result-label">${esc(h.label)}</span><span class="result-sub">${esc(h.sub || "")}</span></button></li>`).join("")}</ul>`
    : `<p class="hint">Nothing matches. ${state.search ? "Try other words, or clear the project filter." : "Press Enter to search by meaning instead."}</p>`)
    + (state.search ? `<button class="quiet" id="clear-search">Clear search</button>` : "");
  $$("[data-node]", results).forEach((b) => b.onclick = () => select(b.dataset.node, { center: true }));
  $("#clear-search")?.addEventListener("click", clearSearch);
}

async function runSearch(query) {
  const project = state.filters.project || "*";
  try {
    clearError();
    const data = await api(`/api/memories?limit=30&project=${encodeURIComponent(project)}&q=${encodeURIComponent(query)}`);
    state.search = { query, results: data.memories || [] };
    // a selection outranks the highlight, so results would stay hidden behind it
    if (state.selected) select(null);
    await refocus();
    if (state.view === "list") { state.list = state.search.results; renderList(); }
  } catch (e) { showError(e); }
}

// Search first: the results become the graph, with their neighbourhood
// fetched server-side, so a hit outside the loaded newest-N still shows.
async function refocus({ apply = true } = {}) {
  const seeds = (state.search?.results || []).map((m) => m.id);
  if (!seeds.length) { state.focusGraph = null; rebuildById(); if (apply) applyFilters(); return; }
  const params = new URLSearchParams({ seeds: seeds.join(","), depth: String(state.depth), types: [...state.edgeTypes].join(","), limit: "1500" });
  if (state.filters.project) params.set("project", state.filters.project);
  try {
    state.focusGraph = await api(`/api/graph?${params}`);
  } catch (e) { showError(e); return; }
  rebuildById();
  view.userMoved = false;
  if (apply) applyFilters();
}

// Show one node and its neighbourhood, wherever the view currently is.
function focusOn(id) {
  const n = state.byId.get(id);
  state.search = { query: n?.label || id, results: [{ id, label: n?.label || id, kind: n?.kind === "entity" ? n.type : n?.category }] };
  $("#q").value = "";
  // only once it is drawn: an unreachable id must not bounce back here
  refocus().then(() => { if (view.adj.has(id)) select(id, { center: true }); });
}

function clearSearch() {
  state.search = null; state.text = ""; $("#q").value = "";
  state.focusGraph = null;
  rebuildById();
  applyFilters();
  if (state.view === "list") loadList();
}

// ---- selection --------------------------------------------------------

function select(id, { center = false } = {}) {
  const drawn = id && view.byId.get(id);
  if (drawn?.data.kind === "cluster" && view.adj.has(id)) return openBubble(drawn.data);
  // Not on the canvas (the overview, or outside an opened bubble): bring it in.
  if (id && !view.adj.has(id) && state.byId.has(id)) return focusOn(id);
  state.selected = id;
  view.select(id, { center });
  if (!id) { closePanel(); if (state.local) { state.local = false; applyFilters(); } return; }
  if (state.local) applyFilters();
  if (currentId() !== id) openNode(id);
  // the panel narrows the stage on the next layout; keep the selection in view
  setTimeout(() => view.ensureVisible(id), 260);
}

// ---- views ------------------------------------------------------------

function setView(name) {
  state.view = name;
  for (const v of ["graph", "list", "projects"]) {
    $(`#view-${v}`).hidden = v !== name;
    const tab = $(`[data-view="${v}"]`);
    tab.setAttribute("aria-current", v === name ? "page" : "false");
  }
  if (name === "list") loadList();
  if (name === "projects") renderProjects();
  if (name === "graph") view.resize();
}

function setProject(id) {
  state.filters.project = id;
  state.path = [];
  state.page = { ...state.page, offset: 0 };
  $("#f-project").value = state.meta[id] ? id : "";
  view.userMoved = false;
  loadGraph();
}

// ---- list view --------------------------------------------------------

async function loadList() {
  const list = $("#list");
  list.setAttribute("aria-busy", "true");
  try {
    clearError();
    const project = state.filters.project || "*";
    const q = state.search?.query || "";
    const { sort, group, offset } = state.page;
    const params = new URLSearchParams({ limit: String(q ? 100 : PAGE), project, q, sort, group, offset: String(q ? 0 : offset) });
    const data = await api(`/api/memories?${params}`);
    state.list = data.memories || [];
    state.page = { ...state.page, total: data.total ?? state.list.length };
    renderList();
  } catch (e) {
    list.innerHTML = "";
    showError(e);
  } finally {
    list.setAttribute("aria-busy", "false");
  }
}

function renderList() {
  const list = $("#list");
  const scope = state.filters.project ? projectName(state.filters.project, state.meta) : "every project";
  $("#list-scope").textContent = state.search ? `Search results in ${scope}` : `Current memories in ${scope}`;
  const projectIds = [state.filters.project || "default", ...state.projects.map((p) => p.project_id)];
  if (!state.list.length && !state.creating) {
    // Empty state teaches the interface rather than saying "no results".
    list.innerHTML = state.search ? `<div class="empty"><h2>No memory matches that</h2><p>Search compares meaning, not words. Try describing the idea, or widen the project.</p></div>`
      : `<div class="empty"><h2>Nothing stored yet</h2>
      <p>Memories arrive when your agent's session ends, or you can write one now.
      From a terminal: <code>collivind add "…"</code></p></div>`;
    return;
  }
  renderPager();
  const { group } = state.page;
  const keyOf = (m) => group === "project" ? projectName(m.project_id, state.meta) : group === "category" ? m.category : "";
  // the server already orders a page by its group; search results are sorted here
  const rows = group && state.search ? [...state.list].sort((a, b) => keyOf(a).localeCompare(keyOf(b))) : state.list;
  let last = null;
  list.innerHTML = (state.creating ? `<article class="memory">${editorMarkup(null, [...new Set(projectIds)])}</article>` : "")
    + rows.map((m) => {
      const head = group && keyOf(m) !== last ? `<h2 class="group-head">${esc(keyOf(m))}<small>${rows.filter((r) => keyOf(r) === keyOf(m)).length} on this page</small></h2>` : "";
      last = keyOf(m);
      return head + (m.id === state.editingId ? `<article class="memory">${editorMarkup(m)}</article>` : card(m));
    }).join("");
  wireList();
}

function renderPager() {
  const { offset, total } = state.page;
  const paged = !state.search;
  $("#pager").hidden = !paged || !total;
  if (!paged) return;
  $("#l-range").textContent = `${(offset + 1).toLocaleString()}–${(offset + state.list.length).toLocaleString()} of ${total.toLocaleString()}`;
  $("#l-prev").disabled = offset <= 0;
  $("#l-next").disabled = offset + PAGE >= total;
}

function wireList() {
  const list = $("#list");
  $$("[data-edit]", list).forEach((b) => b.onclick = () => { state.editingId = b.dataset.edit; state.creating = false; renderList(); $("#f-summary")?.focus(); });
  $$("[data-forget]", list).forEach((b) => b.onclick = () => confirmForget(b.dataset.forget));
  $$("[data-show], [data-open]", list).forEach((b) => b.onclick = () => {
    const id = b.dataset.show || b.dataset.open;
    setView("graph");
    if (state.byId.has(id)) select(id, { center: true });
    else {
      const m = state.list.find((x) => x.id === id);
      state.search = { query: m?.summary || id, results: [{ id, label: m?.summary }] };
      refocus().then(() => { if (view.adj.has(id)) select(id, { center: true }); });
    }
  });
  const form = $("form.editor", list);
  if (form) {
    $("[data-cancel]", form).onclick = () => { state.editingId = null; state.creating = false; renderList(); };
    form.onsubmit = async (e) => {
      e.preventDefault();
      const id = form.dataset.editor;
      try {
        clearError();
        const body = readEditor(!id);
        if (id) await api(`/api/memories/${encodeURIComponent(id)}`, { method: "PATCH", body });
        else await api("/api/memories", { method: "POST", body });
        toast(id ? "Saved" : "Memory created");
        state.editingId = null; state.creating = false;
        await Promise.all([loadList(), loadGraph(), loadProjects()]);
      } catch (err) { showError(err); }
    };
  }
}

// Destructive actions are never one click (PRODUCT.md).
function confirmForget(id) {
  const dialog = $("#confirm");
  const node = state.byId.get(id);
  $("#confirm-body").textContent = `“${(node?.label || id).slice(0, 140)}” will be deleted from both stores with its links. This cannot be undone. Editing it keeps the history instead.`;
  dialog.showModal();
  $("#confirm-cancel").onclick = () => dialog.close();
  $("#confirm-ok").onclick = async () => {
    dialog.close();
    try {
      clearError();
      await api(`/api/memories/${encodeURIComponent(id)}`, { method: "DELETE" });
      toast("Memory forgotten");
      if (state.selected === id) select(null);
      await Promise.all([loadGraph(), loadProjects()]);
      if (state.view === "list") loadList();
    } catch (e) { showError(e); }
  };
}

// ---- filter controls --------------------------------------------------

function renderCategories() {
  $("#f-categories").innerHTML = CATEGORIES.map((c) =>
    `<label class="check"><input type="checkbox" value="${esc(c)}" checked> ${esc(c)}</label>`).join("");
  $$("#f-categories input").forEach((i) => i.onchange = () => {
    i.checked ? state.filters.categories.add(i.value) : state.filters.categories.delete(i.value);
    applyFilters();
  });
}

function renderEdgeTypes() {
  const counts = new Map();
  for (const e of state.graph.edges) counts.set(e.type, (counts.get(e.type) || 0) + 1);
  const box = $("#f-edge-types");
  box.innerHTML = counts.size ? [...counts].sort((a, b) => b[1] - a[1]).map(([t, n]) =>
    `<label class="check"><input type="checkbox" value="${esc(t)}"${state.edgeTypes.has(t) ? " checked" : ""}> ${esc(t.toLowerCase().replace(/_/g, " "))} <span class="count">${n.toLocaleString()}</span></label>`).join("")
    : `<p class="hint">No links yet.</p>`;
  $$("input", box).forEach((i) => i.onchange = () => {
    const next = new Set(state.edgeTypes);
    if (i.checked) next.add(i.value); else next.delete(i.value);
    state.edgeTypes = next;
    if (state.search) refocus(); else applyFilters();
  });
}

// "Recent": days back from today, over the span the loaded memories cover.
function renderSince() {
  const times = state.graph.nodes.filter((n) => n.created_at).map((n) => Date.parse(n.created_at)).filter(Number.isFinite);
  const slider = $("#f-since");
  if (!times.length) { slider.disabled = true; return; }
  slider.disabled = false;
  const span = Math.max(1, Math.ceil((Date.now() - Math.min(...times)) / DAY));
  slider.max = String(span);
  syncSince();
}

function syncSince() {
  const slider = $("#f-since");
  const from = state.filters.from;
  const days = from ? Math.max(0, Math.ceil((Date.now() - Date.parse(from)) / DAY)) : Number(slider.max);
  slider.value = String(Math.min(days, Number(slider.max)));
  $("#f-since-label").textContent = !from ? "any time" : days <= 1 ? "since yesterday" : `last ${days} days`;
}

function renderEntityTypes() {
  const types = [...new Set(state.graph.nodes.filter((n) => n.kind === "entity").map((n) => n.type))].sort();
  const box = $("#f-entity-types");
  box.innerHTML = types.length ? types.map((t) =>
    `<label class="check"><input type="checkbox" value="${esc(t)}"${state.filters.entityTypes.has(t) ? " checked" : ""}> ${esc(t)}</label>`).join("")
    : `<p class="hint">No entities yet. Extraction links memories to the files, services and libraries they mention.</p>`;
  $$("input", box).forEach((i) => i.onchange = () => {
    i.checked ? state.filters.entityTypes.add(i.value) : state.filters.entityTypes.delete(i.value);
    applyFilters();
  });
}

function wireControls() {
  $("#f-project").onchange = (e) => { state.search = null; state.focusGraph = null; setProject(e.target.value); if (state.view === "list") loadList(); };
  $("#f-entities").onchange = (e) => { state.filters.showEntities = e.target.checked; $("#f-entity-types").hidden = !e.target.checked; applyFilters(); };
  $("#f-superseded").onchange = (e) => { state.filters.hideSuperseded = e.target.checked; applyFilters(); };
  $("#f-from").onchange = (e) => { state.filters.from = e.target.value; syncSince(); applyFilters(); };
  let sinceTimer;
  $("#f-since").addEventListener("input", (e) => {
    const days = Number(e.target.value);
    const from = days >= Number(e.target.max) ? "" : new Date(Date.now() - days * DAY).toISOString().slice(0, 10);
    state.filters.from = from;
    $("#f-from").value = from;
    syncSince();
    clearTimeout(sinceTimer);
    sinceTimer = setTimeout(applyFilters, 60);
  });
  $$("[data-mode]").forEach((b) => b.onclick = () => setMode(b.dataset.mode));
  $("#l-sort").onchange = (e) => { state.page = { ...state.page, sort: e.target.value, offset: 0 }; loadList(); };
  $("#l-group").onchange = (e) => { state.page = { ...state.page, group: e.target.value, offset: 0 }; loadList(); };
  $("#l-prev").onclick = () => { state.page = { ...state.page, offset: Math.max(0, state.page.offset - PAGE) }; loadList(); };
  $("#l-next").onclick = () => { state.page = { ...state.page, offset: state.page.offset + PAGE }; loadList(); };
  $("#f-to").onchange = (e) => { state.filters.to = e.target.value; applyFilters(); };
  $("#f-depth").onchange = (e) => {
    state.depth = Number(e.target.value) || 1;
    if (state.search) refocus(); else if (state.local) applyFilters();
  };
  $("#f-reset").onclick = () => {
    state.filters = { ...state.filters, project: "", categories: new Set(CATEGORIES), from: "", to: "", hideSuperseded: false, showEntities: true,
      entityTypes: new Set(state.knownTypes) }; // every type seen, not just the current project's
    $("#f-project").value = ""; $("#f-from").value = ""; $("#f-to").value = ""; $("#f-superseded").checked = false; $("#f-entities").checked = true;
    $("#f-entity-types").hidden = false;
    state.edgeTypes = new Set([...state.knownEdgeTypes].filter((t) => !DEFAULT_OFF.has(t)));
    renderCategories(); loadGraph();
  };

  let timer;
  $("#q").addEventListener("input", (e) => {
    clearTimeout(timer);
    timer = setTimeout(() => {
      state.text = e.target.value.trim();
      if (!state.text) state.search = null;
      updateHighlight();
    }, 120);
  });
  $("#search-form").onsubmit = (e) => {
    e.preventDefault();
    const q = $("#q").value.trim();
    if (q) runSearch(q); else clearSearch();
  };
  $$("[data-view]").forEach((b) => b.onclick = () => setView(b.dataset.view));
  $("#new").onclick = () => { setView("list"); state.creating = true; state.editingId = null; renderList(); $("#f-summary")?.focus(); };
  $("#compact").onclick = (e) => {
    const on = document.body.classList.toggle("compact");
    e.currentTarget.setAttribute("aria-pressed", String(on));
  };
  $("#zoom-in").onclick = () => view.zoomAt(view.w / 2, view.h / 2, 1.3);
  $("#zoom-out").onclick = () => view.zoomAt(view.w / 2, view.h / 2, 1 / 1.3);
  $("#zoom-fit").onclick = () => { view.userMoved = false; view.fit(); };
  $("#error-close").onclick = clearError;
  $("#theme").onclick = cycleTheme;

  // Keyboard first, borrowed from the Linear/Raycast reference.
  document.addEventListener("keydown", (e) => {
    const typing = /^(INPUT|TEXTAREA|SELECT)$/.test(document.activeElement?.tagName) || document.activeElement?.isContentEditable;
    if (e.key === "Escape") {
      if ($("#confirm").open) return;
      if (typing && document.activeElement === $("#q")) { $("#q").blur(); return; }
      if (!typing && state.selected) { select(null); $("#canvas").focus(); }
      return;
    }
    if (typing || e.metaKey || e.ctrlKey || e.altKey) return;
    const keys = {
      "/": () => $("#q").focus(),
      g: () => setView("graph"), l: () => setView("list"), p: () => setView("projects"),
      v: () => { setView("graph"); setMode("overview"); }, e: () => { setView("graph"); setMode("all"); },
      o: () => { if (state.selected) { state.local = !state.local; applyFilters(); refreshPanel(); } },
    };
    if (keys[e.key]) { e.preventDefault(); keys[e.key](); }
  });
}

// ---- theme ------------------------------------------------------------

const THEMES = ["auto", "light", "dark"];
function applyTheme(name) {
  if (name === "auto") delete document.documentElement.dataset.theme;
  else document.documentElement.dataset.theme = name;
  $("#theme").setAttribute("aria-label", `Theme: ${name}. Change theme`);
  $("#theme-label").textContent = name === "auto" ? "Auto" : name === "light" ? "Light" : "Dark";
  requestAnimationFrame(() => view.readTheme());
}
function cycleTheme() {
  const now = document.documentElement.dataset.theme || "auto";
  const next = THEMES[(THEMES.indexOf(now) + 1) % THEMES.length];
  try { localStorage.setItem("collivind-theme", next); } catch { /* private mode: theme just isn't remembered */ }
  applyTheme(next);
}
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => view.readTheme());

// ---- boot -------------------------------------------------------------

(async function boot() {
  let saved = "auto";
  try { saved = localStorage.getItem("collivind-theme") || "auto"; } catch { /* storage blocked */ }
  applyTheme(THEMES.includes(saved) ? saved : "auto");
  renderCategories();
  wireControls();
  try {
    const s = await api("/api/status");
    state.linkable = s.linkable || {};
    $("#store").textContent = `${s.mode} store`;
  } catch { $("#store").textContent = "store unknown"; }
  try { await loadProjects(); } catch (e) { showError(e); }
  await loadGraph();
})();


