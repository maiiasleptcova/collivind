// The side panel: one memory or entity, read first, editable on demand.

import { $, $$, api, esc, fmtDate, relPhrase, toast, showError, clearError, projectName } from "./util.js";
import { detailsMarkup, editorMarkup, readEditor } from "./memory.js";

const panel = () => $("#panel");
let ctx = null;          // { graph, linkable, projects, select, reload, confirm, local }
let current = null;      // { kind, id, data }
let pendingTarget = null;

export function initPanel(context) {
  ctx = context;
  $("#panel-close").onclick = () => ctx.select(null);
}

export function closePanel() {
  current = null;
  panel().hidden = true;
  document.body.classList.remove("has-panel");
}

const nodeById = (id) => ctx.graph().byId.get(id);
const labelOf = (id) => nodeById(id)?.label || id;
const kindOf = (id) => nodeById(id)?.kind;

export async function openNode(id, { linkTo = null } = {}) {
  const node = nodeById(id);
  if (!node) return closePanel();
  pendingTarget = linkTo;
  panel().hidden = false;
  document.body.classList.add("has-panel");
  $("#panel-body").innerHTML = `<div class="skeleton"></div>`;
  try {
    if (node.kind === "memory") {
      const m = await api(`/api/memories/${encodeURIComponent(id)}`);
      current = { kind: "memory", id, data: m };
    } else {
      current = { kind: "entity", id, data: node };
    }
    render();
    if (linkTo) $("#link-type")?.focus();
  } catch (e) {
    $("#panel-body").innerHTML = `<p class="hint">${esc(e.message)}</p>`;
  }
}

function edgesOf(id) {
  return ctx.graph().edges
    .filter((e) => e.source === id || e.target === id)
    .map((e) => ({ ...e, out: e.source === id, other: e.source === id ? e.target : e.source }));
}

// Newest first, dated: an entity's connections read as its history.
const createdOf = (nodeId) => ctx.graph().byId.get(nodeId)?.created_at || "";

function connectionsMarkup(id) {
  const edges = edgesOf(id).sort((a, b) => createdOf(b.other).localeCompare(createdOf(a.other)));
  if (!edges.length) return `<p class="hint">No connections yet. Link this to another memory or entity below, or shift-drag between two nodes on the graph.</p>`;
  return `<ul class="connections">${edges.map((e) => {
    const removable = e.type !== "SUPERSEDES";
    const meaning = ctx.meanings()[e.type];
    return `<li class="${e.origin === "auto_detection" ? "is-auto" : ""}">
      <span class="rel" title="${esc(meaning ? meaning.description : e.type)}">${esc(relPhrase(ctx.meanings(), e.type, e.out))}</span>
      <button class="link-like" data-goto="${esc(e.other)}">${esc(labelOf(e.other))}</button>
      ${createdOf(e.other) ? `<time class="when">${esc(fmtDate(createdOf(e.other)))}</time>` : ""}
      ${removable ? `<button class="icon" data-unlink="${esc(JSON.stringify({ source_id: e.source, target_id: e.target, type: e.type }))}" aria-label="Remove ${esc(e.type)} link to ${esc(labelOf(e.other))}" title="Remove link">×</button>` : ""}
      ${e.note ? `<span class="note">${esc(e.note)}</span>` : ""}
    </li>`;
  }).join("")}</ul>`;
}

function linkFormMarkup() {
  return `<form class="link-form" id="link-form" autocomplete="off">
    <label>Link to
      <input id="link-target" placeholder="Type to find a memory or entity" value="${esc(pendingTarget ? labelOf(pendingTarget) : "")}">
    </label>
    <div class="suggestions" id="link-suggestions" role="listbox" aria-label="Matching nodes"></div>
    <div class="editor-row">
      <label>This
        <select id="link-type"></select>
      </label>
      <label class="grow">Why (optional)
        <input id="link-note" maxlength="300" placeholder="e.g. stores every vector in docker mode">
      </label>
      <div class="form-actions"><button type="submit" class="primary" id="link-submit">Add link</button></div>
    </div>
  </form>`;
}

function render() {
  const body = $("#panel-body");
  if (current.kind === "memory") {
    const m = current.data;
    const meta = ctx.projects();
    body.innerHTML = `
      <div class="panel-meta">
        <span class="tag">${esc(m.category)}</span>
        <span class="project-dot" data-color="${esc(m.project_id)}"></span>
        <span>${esc(projectName(m.project_id, meta))}</span>
        ${m.created_at ? `<span>${esc(fmtDate(m.created_at))}</span>` : ""}
        ${m.superseded_by || m.valid_to ? `<span class="muted-chip">superseded</span>` : ""}
      </div>
      <h2 class="panel-title">${esc(m.summary || m.content.slice(0, 120))}</h2>
      <div class="prose">${esc(m.content)}</div>
      <div class="panel-actions">
        <button class="quiet" id="p-edit">Edit</button>
        <button class="quiet" id="p-local" aria-pressed="${ctx.local() ? "true" : "false"}">Local graph</button>
        ${m.version > 1 || m.superseded_by ? `<button class="quiet" id="p-versions">Versions</button>` : ""}
        <button class="danger" id="p-forget">Forget</button>
      </div>
      <div id="p-version-list"></div>
      ${detailsMarkup(m)}
      <section class="panel-section"><h3>Connections</h3>${connectionsMarkup(m.id)}${linkFormMarkup()}</section>`;
  } else {
    const n = current.data;
    const memories = edgesOf(n.id).filter((e) => kindOf(e.other) === "memory").length;
    body.innerHTML = `
      <div class="panel-meta"><span class="tag tag-entity">${esc(n.type)}</span><span>entity</span><span>${memories} linked memories</span></div>
      <h2 class="panel-title">${esc(n.label)}</h2>
      <div class="panel-actions">
        <button class="quiet" id="p-local" aria-pressed="${ctx.local() ? "true" : "false"}">Local graph</button>
      </div>
      <section class="panel-section"><h3>Connections</h3>${connectionsMarkup(n.id)}${linkFormMarkup()}</section>`;
  }
  $$(".project-dot", body).forEach((el) => { el.style.background = ctx.colorOf(el.dataset.color); });
  wire();
}

function wire() {
  const body = $("#panel-body");
  $$("[data-goto]", body).forEach((b) => b.onclick = () => ctx.select(b.dataset.goto, { center: true }));
  $$("[data-unlink]", body).forEach((b) => b.onclick = () => unlink(JSON.parse(b.dataset.unlink)));
  $("#p-local").onclick = () => ctx.toggleLocal();
  $("#p-edit")?.addEventListener("click", startEdit);
  $("#p-forget")?.addEventListener("click", () => ctx.confirmForget(current.id));
  $("#p-versions")?.addEventListener("click", showVersions);
  wireLinkForm();
}

function startEdit() {
  const m = current.data;
  const body = $("#panel-body");
  body.innerHTML = `<h2 class="panel-title">Edit memory</h2>${editorMarkup(m)}${detailsMarkup(m, true)}`;
  const form = $("form.editor", body);
  $("[data-cancel]", body).onclick = render;
  form.onsubmit = async (e) => {
    e.preventDefault();
    try {
      clearError();
      const updated = await api(`/api/memories/${encodeURIComponent(m.id)}`, { method: "PATCH", body: readEditor(false) });
      current = { ...current, data: updated };
      toast("Saved");
      await ctx.reload();
      render();
    } catch (err) { showError(err); }
  };
  $("#f-summary").focus();
}

async function showVersions() {
  const box = $("#p-version-list");
  try {
    const { versions } = await api(`/api/memories/${encodeURIComponent(current.id)}/versions`);
    box.innerHTML = `<ol class="versions">${versions.map((v) => `<li>
      <button class="link-like" data-goto="${esc(v.id)}"${v.id === current.id ? ' aria-current="true"' : ""}>v${esc(v.version)} · ${esc(v.summary || v.content.slice(0, 80))}</button>
      <span class="hint">${esc(fmtDate(v.created_at))}</span></li>`).join("")}</ol>`;
    $$("[data-goto]", box).forEach((b) => b.onclick = () => ctx.select(b.dataset.goto, { center: true }));
  } catch (e) { showError(e); }
}

// The link types the server accepts for this pair of node kinds. Memory to
// entity is the only mixed direction, so an entity panel linking to a memory
// sends the edge the other way round.
function linkPlan(targetId) {
  const a = current.kind, b = kindOf(targetId);
  if (!b) return null;
  if (a === "entity" && b === "memory") return { source: targetId, target: current.id, types: ctx.linkable()["memory:entity"] || [] };
  return { source: current.id, target: targetId, types: ctx.linkable()[`${a}:${b}`] || [] };
}

function wireLinkForm() {
  const input = $("#link-target"), list = $("#link-suggestions"), select = $("#link-type"), submit = $("#link-submit");
  const refreshTypes = () => {
    const plan = pendingTarget ? linkPlan(pendingTarget) : null;
    const types = plan ? plan.types : [];
    // read from the open node: "Neo4j  is part of  …", or "… is used by" when the edge points here
    const outgoing = !plan || plan.source === current.id;
    select.innerHTML = types.map((t) => `<option value="${esc(t)}" title="${esc(ctx.meanings()[t]?.description || "")}">${esc(relPhrase(ctx.meanings(), t, outgoing))}</option>`).join("");
    select.disabled = submit.disabled = !types.length;
    select.closest(".editor-row").hidden = !types.length; // nothing to choose until a target is picked
  };
  refreshTypes();
  input.oninput = () => {
    pendingTarget = null; refreshTypes();
    const q = input.value.trim().toLowerCase();
    if (q.length < 2) { list.innerHTML = ""; return; }
    // every loaded node, not just what the current view draws; names that start
    // with the text, then entities, come first ("collivind" before a memory about it)
    const rank = (n) => (n.label.toLowerCase().startsWith(q) ? 0 : 2) + (n.kind === "entity" ? 0 : 1);
    const hits = [...ctx.graph().byId.values()]
      .filter((n) => n.id !== current.id && n.kind !== "cluster" && (n.label || "").toLowerCase().includes(q))
      .sort((a, b) => rank(a) - rank(b) || a.label.length - b.label.length)
      .slice(0, 7);
    list.innerHTML = hits.map((n) => `<button type="button" role="option" data-pick="${esc(n.id)}">
      <span class="kind">${n.kind === "entity" ? esc(n.type) : esc(n.category)}</span>${esc(n.label)}</button>`).join("")
      || `<p class="hint">Nothing loaded matches. Clear the filters to link across projects.</p>`;
    $$("[data-pick]", list).forEach((b) => b.onclick = () => {
      pendingTarget = b.dataset.pick; input.value = labelOf(pendingTarget); list.innerHTML = ""; refreshTypes(); select.focus();
    });
  };
  $("#link-form").onsubmit = async (e) => {
    e.preventDefault();
    const plan = pendingTarget && linkPlan(pendingTarget);
    if (!plan || !select.value) return;
    try {
      clearError();
      await api("/api/links", { method: "POST", body: { source_id: plan.source, target_id: plan.target, type: select.value, note: $("#link-note").value.trim() } });
      pendingTarget = null;
      toast("Linked");
      await ctx.reload();
      render();
    } catch (err) { showError(err); }
  };
}

async function unlink(body) {
  try {
    clearError();
    await api("/api/links", { method: "DELETE", body });
    toast("Link removed");
    await ctx.reload();
    render();
  } catch (e) { showError(e); }
}

export const currentId = () => current?.id || null;
export function refreshPanel() { if (current) render(); }
