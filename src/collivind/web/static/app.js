"use strict";

const $ = (sel) => document.querySelector(sel);
const list = $("#list");
const errorBox = $("#error");
const CATEGORIES = ["fact", "decision", "pattern", "error", "architecture", "preference", "snippet"];

let memories = [];
let editingId = null;
let creating = false;

const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

// An ISO timestamp is data, not reading material — show it the way the reader
// keeps time, and fall back to the raw string rather than printing "Invalid Date".
const fmtTime = (v) => {
  if (v === null || v === undefined || v === "") return null;
  const d = new Date(v);
  return isNaN(d.getTime()) ? String(v) : d.toLocaleString();
};

// Every field the store carries beyond summary/content/category. Rows whose
// value is absent are dropped, so `valid to` showing up means invalidated.
const META_ROWS = [
  ["Project", (m) => m.project_id],
  ["Source", (m) => m.source],
  ["Confidence", (m) => (m.confidence === null || m.confidence === undefined ? null : Number(m.confidence).toFixed(2))],
  ["Version", (m) => m.version],
  ["Previous version", (m) => m.previous_version_id],
  ["Superseded by", (m) => m.superseded_by],
  ["Valid from", (m) => fmtTime(m.valid_from)],
  ["Valid to", (m) => fmtTime(m.valid_to)],
  ["Created", (m) => fmtTime(m.created_at)],
  ["Updated", (m) => fmtTime(m.updated_at)],
  ["Session", (m) => m.session_id],
  ["User", (m) => m.user_id],
  ["ID", (m) => m.id],
];

async function api(path, options = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
    body: options.body ? JSON.stringify(options.body) : undefined,
  });
  const payload = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(payload.error || `${res.status} ${res.statusText}`);
  return payload;
}

function showError(e) {
  errorBox.textContent = e.message;
  errorBox.hidden = false;
}
function clearError() { errorBox.hidden = true; }

function render() {
  list.setAttribute("aria-busy", "false");

  if (creating) {
    list.innerHTML = editorMarkup(null) + memories.map(card).join("");
  } else if (!memories.length) {
    // Empty state teaches the interface rather than saying "no results".
    list.innerHTML = `<div class="empty">
      <h2>Nothing stored yet</h2>
      <p>Memories arrive automatically when your agent's session ends, or you can
      write one now. From a terminal: <code>collivind add "…"</code></p>
    </div>`;
    return;
  } else {
    list.innerHTML = memories.map(card).join("");
  }
  wire();
}

function card(m) {
  if (m.id === editingId) return editorMarkup(m);
  const entities = (m.entities || []).map(
    (e) => `<button class="entity" data-entity="${esc(e.name || e)}">${esc(e.name || e)}</button>`).join("");
  return `<article class="memory" data-id="${esc(m.id)}">
    <p class="summary">${esc(m.summary || m.content)}</p>
    <div class="content">${esc(m.content)}</div>
    <div class="meta">
      <span class="tag">${esc(m.category)}</span>
      ${m.score !== undefined ? `<span>match ${(m.score).toFixed(2)}</span>` : ""}
      ${m.created_at ? `<span>${esc(String(m.created_at).slice(0, 10))}</span>` : ""}
      ${entities}
      <span class="row-actions">
        <button class="quiet" data-edit="${esc(m.id)}">Edit</button>
        <button class="danger" data-forget="${esc(m.id)}">Forget</button>
      </span>
    </div>
    ${detailsMarkup(m)}
  </article>`;
}

// Density on demand (PRODUCT.md): the full record is one disclosure away,
// never in the way of reading. <details> is keyboard-reachable for free.
function detailsMarkup(m) {
  const tags = (m.tags || []).map(
    (t) => `<button class="entity" data-entity="${esc(t)}">${esc(t)}</button>`).join("");
  const rows = META_ROWS
    .map(([label, read]) => [label, read(m)])
    .filter(([, v]) => v !== null && v !== undefined && v !== "")
    .map(([label, v]) => `<dt>${esc(label)}</dt><dd>${esc(v)}</dd>`)
    .join("");
  if (!tags && !rows) return "";
  return `<details class="details">
    <summary>Details</summary>
    ${tags ? `<div class="meta tags">${tags}</div>` : ""}
    ${rows ? `<dl class="fields">${rows}</dl>` : ""}
  </details>`;
}

function editorMarkup(m) {
  const isNew = m === null;
  return `<article class="memory">
    <div class="editor">
      <label>Summary
        <input id="f-summary" value="${esc(m?.summary || "")}" placeholder="One line that stands alone">
      </label>
      <label>Content
        <textarea id="f-content" rows="5" placeholder="What is worth remembering?">${esc(m?.content || "")}</textarea>
      </label>
      ${isNew ? `<label>Category
        <select id="f-category">${CATEGORIES.map((c) => `<option value="${c}">${c}</option>`).join("")}</select>
      </label>` : ""}
      <div class="meta">
        <span class="row-actions">
          <button class="quiet" data-cancel="1">Cancel</button>
          <button class="primary" data-save="${esc(m?.id || "")}">${isNew ? "Create" : "Save"}</button>
        </span>
      </div>
    </div>
  </article>`;
}

function wire() {
  list.querySelectorAll("[data-edit]").forEach((b) =>
    b.onclick = () => { editingId = b.dataset.edit; creating = false; render(); $("#f-summary")?.focus(); });
  list.querySelectorAll("[data-cancel]").forEach((b) =>
    b.onclick = () => { editingId = null; creating = false; render(); });
  list.querySelectorAll("[data-save]").forEach((b) => b.onclick = () => save(b.dataset.save));
  list.querySelectorAll("[data-forget]").forEach((b) => b.onclick = () => confirmForget(b.dataset.forget));
  list.querySelectorAll("[data-entity]").forEach((b) =>
    b.onclick = () => { $("#q").value = b.dataset.entity; load(); });
}

async function save(id) {
  const body = { summary: $("#f-summary").value.trim(), content: $("#f-content").value.trim() };
  if (!body.content) return showError(new Error("Content cannot be empty"));
  try {
    clearError();
    if (id) await api(`/api/memories/${encodeURIComponent(id)}`, { method: "PATCH", body });
    else await api("/api/memories", { method: "POST", body: { ...body, category: $("#f-category").value } });
    editingId = null; creating = false;
    await load();
  } catch (e) { showError(e); }
}

// Destructive actions are never one click (PRODUCT.md).
function confirmForget(id) {
  const dialog = $("#confirm");
  dialog.showModal();
  $("#confirm-cancel").onclick = () => dialog.close();
  $("#confirm-ok").onclick = async () => {
    dialog.close();
    try {
      clearError();
      await api(`/api/memories/${encodeURIComponent(id)}`, { method: "DELETE" });
      await load();
    } catch (e) { showError(e); }
  };
}

let timer;
async function load() {
  clearError();
  list.setAttribute("aria-busy", "true");
  try {
    const q = encodeURIComponent($("#q").value.trim());
    const data = await api(`/api/memories?limit=100&q=${q}`);
    memories = data.memories || [];
    render();
  } catch (e) {
    list.innerHTML = "";
    list.setAttribute("aria-busy", "false");
    showError(e);
  }
}

$("#q").addEventListener("input", () => { clearTimeout(timer); timer = setTimeout(load, 220); });
$("#new").onclick = () => { creating = true; editingId = null; render(); $("#f-summary")?.focus(); };
$("#compact").onclick = (e) => {
  const on = document.body.classList.toggle("compact");
  e.target.setAttribute("aria-pressed", String(on));
};
// Keyboard-first, borrowed from the Linear/Raycast reference.
document.addEventListener("keydown", (e) => {
  if (e.key === "/" && document.activeElement !== $("#q")) { e.preventDefault(); $("#q").focus(); }
});

api("/api/status")
  .then((s) => { $("#store").textContent = `${s.mode} store`; })
  .catch(() => { $("#store").textContent = "store unknown"; });
load();
