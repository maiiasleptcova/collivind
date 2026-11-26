// The wiki: the store read as pages, one per topic (web/wiki.py builds them).
// Layout and navigation follow OpenWiki's visualizer (langchain-ai/openwiki,
// MIT): the reader is the main pane, a sidebar indexes every topic by type,
// and a small map is navigation, labelled only around what you point at.
// Opening a page never moves the map; clicking empty map does nothing.

import { $, $$, api, esc, plural, showError, clearError } from "./util.js";
import { GraphView } from "./graph.js";

const TYPE_ORDER = ["project", "summary", "named-project", "service", "tool", "library", "concept", "file", "person"];
const TYPE_HEAD = { project: "Projects", "named-project": "Projects named in memories", summary: "Summaries", service: "Services", tool: "Tools", library: "Libraries",
  concept: "Concepts", file: "Files", person: "People" };
const BACKLINK_CHIPS = 24;

const w = { index: null, byId: new Map(), id: null, within: null, page: null, filter: "", seq: 0 };
let ctx, map;

export function initWiki(context) {
  ctx = context;
  map = new GraphView($("#w-canvas"), {
    colorOf: ctx.colorOf,
    contextLabels: true,
    edgeLabels: false,
    onSelect: (id) => { if (id) go(id); }, // empty canvas: nothing, the page stays
    onConnect: () => {},
  });
  $("#w-filter").addEventListener("input", (e) => { w.filter = e.target.value.trim().toLowerCase(); renderIndex(); });
  $("#w-fit").onclick = () => { map.userMoved = false; map.fit(); };
  $("#w-find").onclick = () => { if (w.id) map.centerOn(w.id, false); };
  window.addEventListener("hashchange", fromHash);
}

// "#wiki/<id>" opens a page; "~<id>" narrows it to the memories also under another topic.
const hashOf = (id, within) => `#wiki/${encodeURIComponent(id)}${within ? `~${encodeURIComponent(within)}` : ""}`;
export const wikiHref = (id) => hashOf(id, null);

export function fromHash() {
  const m = location.hash.match(/^#wiki\/([^~]*)(?:~(.*))?$/);
  if (!m) return false;
  ctx.showWiki();
  const id = decodeURIComponent(m[1]), within = m[2] ? decodeURIComponent(m[2]) : null;
  if (id && (id !== w.id || within !== w.within)) openPage(id, within);
  return true;
}

function go(id, within = null) {
  const target = hashOf(id, within);
  if (location.hash === target) openPage(id, within); else location.hash = target;
}

export async function loadWiki() {
  try {
    const project = ctx.project();
    w.index = await api(`/api/wiki${project ? `?project=${encodeURIComponent(project)}` : ""}`);
    w.byId = new Map(w.index.topics.map((t) => [t.id, t]));
    renderIndex();
    renderMap();
    if (!fromHash()) {
      const first = w.id && w.byId.has(w.id) ? w.id : startTopic();
      if (first) { history.replaceState(null, "", hashOf(first)); openPage(first); }
      else renderEmpty();
    }
  } catch (e) { showError(e); }
}

// The project page when there is one project, else the busiest topic.
function startTopic() {
  const projects = w.index.topics.filter((t) => t.kind === "project");
  if (projects.length === 1) return projects[0].id;
  return [...w.index.topics].sort((a, b) => b.count - a.count)[0]?.id;
}

function renderEmpty() {
  $("#w-page").innerHTML = `<div class="empty"><h2>No topics yet</h2><p>Pages appear once memories are about something:
    a service, a library, a file, an idea. Agents name these when they save; see the Graph view for what is stored.</p></div>`;
}

// ---- sidebar index ---------------------------------------------------------

function renderIndex() {
  const groups = new Map();
  for (const t of w.index?.topics || []) {
    if (w.filter && !`${t.title} ${t.description}`.toLowerCase().includes(w.filter)) continue;
    // a project-type entity is a topic memories name, not one of the store's projects
    const key = t.kind === "entity" && t.type === "project" ? "named-project" : TYPE_HEAD[t.type] ? t.type : "other";
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(t);
  }
  const order = [...TYPE_ORDER, "other"].filter((k) => groups.has(k));
  $("#w-index").innerHTML = order.length ? order.map((k) => {
    const rows = groups.get(k).sort((a, b) => b.count - a.count || a.title.localeCompare(b.title));
    return `<section><h2>${esc(TYPE_HEAD[k] || "Other")} <span class="count">${rows.length}</span></h2><ul>${rows.map((t) =>
      `<li><a href="${hashOf(t.id)}" data-topic="${esc(t.id)}"${t.id === w.id ? ' aria-current="page"' : ""} title="${esc(t.description)}">
        <span>${esc(t.title)}</span><span class="count">${t.count.toLocaleString()}</span></a></li>`).join("")}</ul></section>`;
  }).join("") : `<p class="hint">No topic matches “${esc(w.filter)}”.</p>`;
}

// ---- topic map --------------------------------------------------------------

function renderMap() {
  const nodes = w.index.topics.map((t) => t.kind === "summary"
    ? { id: t.id, kind: "memory", category: "summary", label: t.title, members: t.count, size: t.count, project_id: "" }
    : { id: t.id, kind: "entity", type: t.type, label: t.title, count: t.count, size: t.count });
  const edges = w.index.links.map((l) => ({ source: l.source, target: l.target, type: "TOPIC", note: "" }));
  map.setGraph(nodes, edges, { reheat: 0.6 });
  if (w.id) map.select(w.id);
}

// ---- the page --------------------------------------------------------------

async function openPage(id, within = null) {
  const seq = ++w.seq;
  w.id = id; w.within = within;
  const params = new URLSearchParams({ id });
  if (within) params.set("within", within);
  if (ctx.project()) params.set("project", ctx.project());
  try {
    clearError();
    const page = await api(`/api/wiki/page?${params}`);
    if (seq !== w.seq) return; // a newer page was asked for meanwhile
    w.page = page;
    renderPage();
    $$("#w-index [data-topic]").forEach((a) => a.toggleAttribute("aria-current", a.dataset.topic === id));
    $(`#w-index [data-topic="${CSS.escape(id)}"]`)?.scrollIntoView({ block: "nearest" });
    map.select(id); // highlighted where it is: the camera stays put
    $("#w-page").scrollTop = 0;
  } catch (e) { if (seq === w.seq) showError(e); }
}

const topicLink = (t) => `<a href="${hashOf(t.id)}" class="topic-link">${esc(t.title)}</a>`;

function memoryRow(m, { compact = false } = {}) {
  const att = [...(m.about || []), ...(m.summaries || [])];
  const links = (m.links || []).filter((l) => !(compact && (l.type === "RESOLVES" || l.type === "CAUSED_BY")));
  return `<div class="w-mem cat-${esc(m.category)}" data-mem="${esc(m.id)}">
    <div class="w-mem-head">
      <span class="w-kind kind-${esc(m.category)}">${esc(m.kind)}</span>
      <button class="w-label" aria-expanded="false" data-expand="${esc(m.id)}">${esc(m.label)}</button>
      <time datetime="${esc(m.date)}">${esc(m.date)}</time>
    </div>
    ${m.gist ? `<p class="w-gist">${esc(m.gist)}</p>` : ""}
    ${att.length ? `<p class="w-att">Also under ${att.map(topicLink).join(", ")}${m.more_about ? ` and ${m.more_about} more` : ""}</p>` : ""}
    ${links.length ? `<ul class="w-links">${links.map((l) => `<li><span class="rel">${esc(l.phrase)}</span>
      <button class="quiet" data-expand-link="${esc(l.memory.id)}">${esc(l.memory.label)}</button>${l.memory.live ? "" : ` <span class="hint">(older version)</span>`}${l.note ? ` <span class="w-note">— ${esc(l.note)}</span>` : ""}</li>`).join("")}</ul>` : ""}
    <div class="w-full" hidden><p>${esc(m.content)}</p>
      <p class="w-full-meta">${esc(m.project)} · <button class="quiet" data-graph="${esc(m.id)}">Show on graph</button></p></div>
  </div>`;
}

function pairRows(m) {
  const fixes = m.fixes.map((f) => `<div class="w-fix"><p class="w-pair-head">Fixed by${f.note ? ` <span class="w-note">— ${esc(f.note)}</span>` : ""}</p>${memoryRow(f, { compact: true })}</div>`).join("");
  const causes = m.causes.map((c) => `<div class="w-cause"><p class="w-pair-head">Caused by${c.note ? ` <span class="w-note">— ${esc(c.note)}</span>` : ""}</p>${memoryRow(c, { compact: true })}</div>`).join("");
  return `<li class="w-pair">${memoryRow(m, { compact: true })}${causes}${fixes ||
    `<p class="w-nofix">No fix recorded yet. If one exists, link it from the fix with “resolves”.</p>`}</li>`;
}

function sectionHtml(s) {
  const rows = s.items.map((m) => (s.key === "problems" ? pairRows(m) : `<li>${memoryRow(m)}</li>`)).join("");
  const more = s.more ? `<button class="quiet w-more" data-more="${esc(s.key)}" data-offset="${s.offset + s.items.length}">
    Show ${Math.min(s.more, 20)} more <span class="count">${s.more.toLocaleString()} not shown</span></button>` : "";
  return `<section class="w-section" id="w-sec-${esc(s.key)}" data-section="${esc(s.key)}">
    <h2>${esc(s.title)} <span class="count">${s.total.toLocaleString()}</span></h2>
    <ol class="w-list">${rows}</ol>${more}</section>`;
}

function renderPage() {
  const p = w.page;
  const kindWord = p.kind === "entity" ? `${p.type} · topic` : p.kind;
  const toc = p.sections.length > 1 ? `<nav class="w-toc" aria-label="On this page">${p.sections.map((s) =>
    `<a href="#w-sec-${esc(s.key)}" data-jump="w-sec-${esc(s.key)}">${esc(s.title)} <span class="count">${s.total}</span></a>`).join("")}</nav>` : "";
  const within = p.within ? `<p class="w-within" role="status">Only the memories also under ${topicLink(p.within)}.
    <a href="${hashOf(p.id)}">Show all of ${esc(p.title)}</a></p>` : "";
  const facts = p.facts?.length ? `<section class="w-section"><h2>What it says</h2><ul class="w-facts">${p.facts.map((f) => `<li>${esc(f)}</li>`).join("")}</ul></section>` : "";
  const projectTopics = p.kind === "project" ? (p.related || []).filter((r) => r.kind === "entity") : [];
  const related = (p.related || []).filter((r) => !projectTopics.includes(r));
  const relatedIds = new Set((p.related || []).map((r) => r.id));
  const backlinks = (p.backlinks || []).filter((b) => !relatedIds.has(b.id));
  $("#w-page").innerHTML = `
    <header class="w-head">
      <p class="w-type">${esc(kindWord)}</p>
      <h1>${esc(p.title)}</h1>
      <p class="w-desc">${esc(p.description)}</p>
      <p class="w-meta">${esc(p.stats)}<span class="w-src">${p.described_by === "agent" ? "Description written by an agent" : "Description built from its links and counts"}</span>
        <button class="quiet" id="w-find-here" title="Center the map on this topic">Find on map</button></p>
      ${p.hub && !p.within ? `<p class="hint">Many memories are about this. Each section shows the newest first, in pages; narrow it to one related topic below.</p>` : ""}
    </header>
    ${within}${facts}
    ${projectTopics.length ? `<section class="w-section"><h2>Topics in this project <span class="count">${projectTopics.length}</span></h2>
      <ul class="w-topics">${projectTopics.map((r) => `<li>${topicLink(r)} <span class="hint">${esc(r.type)} · ${esc(r.reason)}</span></li>`).join("")}</ul>
      ${p.unfiled ? `<p class="hint">Below: the ${plural(p.unfiled, "memory")} about no topic.</p>` : ""}</section>` : ""}
    ${toc}
    ${p.sections.map(sectionHtml).join("") || (p.kind === "project" ? "" : `<p class="hint">No memories are about this yet; it is known through its links.</p>`)}
    ${related.length ? `<section class="w-section"><h2>Related topics</h2><ul class="w-related">${related.map((r) => `<li>
      ${topicLink(r)} <span class="w-type-chip">${esc(r.type)}</span> <span class="w-reason">${esc(r.reason)}</span>
      ${r.count > 1 && p.kind !== "summary" && r.kind === "entity" ? `<a class="w-narrow" href="${hashOf(p.id, r.id)}">read these ${r.count}</a>` : ""}</li>`).join("")}</ul></section>` : ""}
    ${backlinks.length ? `<section class="w-section"><h2>Linked from <span class="count">${backlinks.length}</span></h2><p class="w-chips">${backlinks.slice(0, BACKLINK_CHIPS).map((b) =>
      `<a class="chip" href="${hashOf(b.id)}">${esc(b.title)}</a>`).join("")}${backlinks.length > BACKLINK_CHIPS ? `<span class="hint">and ${backlinks.length - BACKLINK_CHIPS} more</span>` : ""}</p></section>` : ""}`;
  wirePage();
}

function wirePage() {
  const page = $("#w-page");
  $("#w-find-here").onclick = () => map.centerOn(w.id, false);
  $$("[data-jump]", page).forEach((a) => a.onclick = (e) => { e.preventDefault(); $(`#${a.dataset.jump}`)?.scrollIntoView({ block: "start" }); });
  page.onclick = (e) => {
    const t = e.target.closest("button");
    if (!t || !page.contains(t)) return;
    if (t.dataset.expand) return toggle(t);
    if (t.dataset.expandLink) {
      const row = t.closest(".w-mem");
      const shown = page.querySelector(`.w-mem[data-mem="${CSS.escape(t.dataset.expandLink)}"] .w-label`);
      if (shown) { shown.scrollIntoView({ block: "center" }); if (shown.getAttribute("aria-expanded") !== "true") toggle(shown); shown.focus(); return; }
      return ctx.showMemory(t.dataset.expandLink, row?.dataset.mem);
    }
    if (t.dataset.graph) return ctx.showMemory(t.dataset.graph);
    if (t.dataset.more) return more(t);
  };
}

function toggle(button) {
  const full = button.closest(".w-mem").querySelector(".w-full");
  const open = button.getAttribute("aria-expanded") !== "true";
  button.setAttribute("aria-expanded", String(open));
  full.hidden = !open;
}

async function more(button) {
  const key = button.dataset.more, offset = Number(button.dataset.offset);
  const params = new URLSearchParams({ id: w.id, section: key, offset: String(offset), limit: "20" });
  if (w.within) params.set("within", w.within);
  if (ctx.project()) params.set("project", ctx.project());
  try {
    button.disabled = true;
    const page = await api(`/api/wiki/page?${params}`);
    const s = page.sections[0];
    if (!s) { button.remove(); return; }
    const list = $(`#w-sec-${key} .w-list`);
    const first = list.children.length;
    list.insertAdjacentHTML("beforeend", s.items.map((m) => (key === "problems" ? pairRows(m) : `<li>${memoryRow(m)}</li>`)).join(""));
    list.children[first]?.querySelector(".w-label")?.focus();
    if (s.more) { button.dataset.offset = String(s.offset + s.items.length); button.disabled = false;
      button.innerHTML = `Show ${Math.min(s.more, 20)} more <span class="count">${s.more.toLocaleString()} not shown</span>`; }
    else button.remove();
  } catch (e) { button.disabled = false; showError(e); }
}

export function focusFilter() { $("#w-filter").focus(); }
export function focusMap() { $("#w-canvas").focus(); }
export function resizeMap() { map.resize(); map.readTheme(); }
export function themeChanged() { map.readTheme(); }
