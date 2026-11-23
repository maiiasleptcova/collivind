// Entry point: state, the three views (graph, list, projects), filters and search.

import { $, $$, api, esc, toast, showError, clearError, projectColor, projectName } from "./util.js";
import { CATEGORIES, card, editorMarkup, readEditor } from "./memory.js";
import { GraphView } from "./graph.js";
import { initPanel, openNode, closePanel, currentId, refreshPanel } from "./panel.js";
import { initProjects, renderProjects } from "./projects.js";

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
};

const colorOf = (projectId) => projectColor(projectId, state.meta);

const view = new GraphView($("#canvas"), {
  colorOf,
  onSelect: (id) => select(id),
  onConnect: (source, target) => { select(source); openNode(source, { linkTo: target }); },
});

initPanel({
  graph: () => ({ byId: state.byId, nodes: state.visible, edges: state.graph.edges }),
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
    state.byId = new Map(g.nodes.map((n) => [n.id, n]));
    // Types never seen before arrive switched on; only an untick hides one.
    const types = g.nodes.filter((n) => n.kind === "entity").map((n) => n.type);
    state.filters.entityTypes ??= new Set();
    for (const t of types) if (!state.knownTypes.has(t)) { state.knownTypes.add(t); state.filters.entityTypes.add(t); }
    renderEntityTypes();
    applyFilters();
    if (state.selected && !state.byId.has(state.selected)) select(null);
  } catch (e) { showError(e); }
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

function applyFilters() {
  const f = state.filters;
  const { nodes, edges } = state.graph;
  const memIds = new Set(nodes.filter((n) => n.kind === "memory" && memoryPasses(n, f)).map((n) => n.id));
  const reached = new Set();
  if (f.showEntities) {
    for (const e of edges) {
      if (memIds.has(e.source)) reached.add(e.target);
      if (memIds.has(e.target)) reached.add(e.source);
    }
  }
  let ids = new Set(nodes.filter((n) => memIds.has(n.id) ||
    (n.kind === "entity" && reached.has(n.id) && f.entityTypes?.has(n.type))).map((n) => n.id));

  // Local graph: n hops around the selection, over what the filters allow.
  if (state.local && state.selected && ids.has(state.selected)) {
    const near = new Set([state.selected]);
    let frontier = [state.selected];
    for (let hop = 0; hop < state.depth; hop++) {
      const next = [];
      for (const e of edges) {
        for (const [a, b] of [[e.source, e.target], [e.target, e.source]]) {
          if (frontier.includes(a) && ids.has(b) && !near.has(b)) { near.add(b); next.push(b); }
        }
      }
      frontier = next;
    }
    ids = near;
  }
  $("#stage").classList.toggle("is-local", state.local);

  state.visible = nodes.filter((n) => ids.has(n.id));
  view.setGraph(state.visible, edges, { reheat: 0.4 });
  if (state.local) view.userMoved = false;
  updateHighlight();
  renderNote();
}

function renderNote() {
  const g = state.graph;
  const shownMem = state.visible.filter((n) => n.kind === "memory").length;
  const shownEnt = state.visible.length - shownMem;
  const parts = [`Showing ${shownMem} of ${g.total_memories} memories and ${shownEnt} entities.`];
  if (g.truncated) {
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
    hits = state.search.results.map((m) => ({ id: m.id, label: m.summary || m.content, sub: `match ${Number(m.score).toFixed(2)}`, kind: "memory", category: m.category }));
    heading = `Search results for “${state.search.query}”`;
  } else if (state.text) {
    const q = state.text.toLowerCase();
    hits = state.visible.filter((n) => n.label.toLowerCase().includes(q)).slice(0, 60)
      .map((n) => ({ id: n.id, label: n.label, sub: n.kind === "entity" ? n.type : n.category, kind: n.kind }));
    heading = `Matching “${state.text}”`;
  }
  view.setHighlight(hits.length ? new Set(hits.map((h) => h.id)) : null);
  if (!heading) { results.innerHTML = ""; return; }
  results.innerHTML = `<h3>${esc(heading)}</h3>` + (hits.length
    ? `<ul>${hits.map((h) => `<li><button class="result" data-node="${esc(h.id)}">
        <span class="result-label">${esc(h.label)}</span><span class="result-sub">${esc(h.sub || "")}${state.byId.has(h.id) ? "" : " · not in the loaded graph"}</span></button></li>`).join("")}</ul>`
    : `<p class="hint">Nothing matches. ${state.search ? "Try other words, or clear the project filter." : "Press Enter to search by meaning instead."}</p>`)
    + (state.search ? `<button class="quiet" id="clear-search">Clear search</button>` : "");
  $$("[data-node]", results).forEach((b) => b.onclick = () => {
    if (state.byId.has(b.dataset.node)) select(b.dataset.node, { center: true });
    else toast("That memory is outside the loaded graph");
  });
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
    updateHighlight();
    if (state.view === "list") { state.list = state.search.results; renderList(); }
  } catch (e) { showError(e); }
}

function clearSearch() {
  state.search = null; state.text = ""; $("#q").value = "";
  updateHighlight();
  if (state.view === "list") loadList();
}

// ---- selection --------------------------------------------------------

function select(id, { center = false } = {}) {
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
    const data = await api(`/api/memories?limit=100&project=${encodeURIComponent(project)}&q=${encodeURIComponent(q)}`);
    state.list = data.memories || [];
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
  $("#list-scope").textContent = state.search ? `Search results in ${scope}` : `Newest current memories in ${scope}`;
  const projectIds = [state.filters.project || "default", ...state.projects.map((p) => p.project_id)];
  if (!state.list.length && !state.creating) {
    // Empty state teaches the interface rather than saying "no results".
    list.innerHTML = state.search ? `<div class="empty"><h2>No memory matches that</h2><p>Search compares meaning, not words. Try describing the idea, or widen the project.</p></div>`
      : `<div class="empty"><h2>Nothing stored yet</h2>
      <p>Memories arrive when your agent's session ends, or you can write one now.
      From a terminal: <code>collivind add "…"</code></p></div>`;
    return;
  }
  list.innerHTML = (state.creating ? `<article class="memory">${editorMarkup(null, [...new Set(projectIds)])}</article>` : "")
    + state.list.map((m) => m.id === state.editingId ? `<article class="memory">${editorMarkup(m)}</article>` : card(m)).join("");
  wireList();
}

function wireList() {
  const list = $("#list");
  $$("[data-edit]", list).forEach((b) => b.onclick = () => { state.editingId = b.dataset.edit; state.creating = false; renderList(); $("#f-summary")?.focus(); });
  $$("[data-forget]", list).forEach((b) => b.onclick = () => confirmForget(b.dataset.forget));
  $$("[data-show], [data-open]", list).forEach((b) => b.onclick = () => {
    const id = b.dataset.show || b.dataset.open;
    if (!state.byId.has(id)) return toast("That memory is outside the loaded graph");
    setView("graph"); select(id, { center: true });
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
  $("#f-project").onchange = (e) => { state.search = null; setProject(e.target.value); if (state.view === "list") loadList(); };
  $("#f-entities").onchange = (e) => { state.filters.showEntities = e.target.checked; $("#f-entity-types").hidden = !e.target.checked; applyFilters(); };
  $("#f-superseded").onchange = (e) => { state.filters.hideSuperseded = e.target.checked; applyFilters(); };
  $("#f-from").onchange = (e) => { state.filters.from = e.target.value; applyFilters(); };
  $("#f-to").onchange = (e) => { state.filters.to = e.target.value; applyFilters(); };
  $("#f-depth").onchange = (e) => { state.depth = Number(e.target.value) || 1; if (state.local) applyFilters(); };
  $("#f-reset").onclick = () => {
    state.filters = { ...state.filters, project: "", categories: new Set(CATEGORIES), from: "", to: "", hideSuperseded: false, showEntities: true,
      entityTypes: new Set(state.graph.nodes.filter((n) => n.kind === "entity").map((n) => n.type)) };
    $("#f-project").value = ""; $("#f-from").value = ""; $("#f-to").value = ""; $("#f-superseded").checked = false; $("#f-entities").checked = true;
    $("#f-entity-types").hidden = false;
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
