// How one memory is drawn and edited, shared by the list and the side panel.

import { $, esc, fmtNum, fmtTime, fmtDate } from "./util.js";

export const CATEGORIES = ["fact", "decision", "pattern", "error", "architecture", "preference", "snippet"];

// Every field the store carries beyond summary/content/category. Rows whose
// value is absent are dropped, so `valid to` showing up means invalidated.
// The provenance rows (session, user, source) are read-only: they record who
// wrote the memory, and rewriting them would falsify that.
export const META_ROWS = [
  ["Project", (m) => m.project_id],
  ["Source", (m) => m.source],
  ["Confidence", (m) => fmtNum(m.confidence)],
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

export function card(m) {
  return `<article class="memory${m.superseded_by || m.valid_to ? " is-superseded" : ""}" data-id="${esc(m.id)}">
    <button class="summary-link" data-open="${esc(m.id)}">${esc(m.summary || m.content)}</button>
    <div class="content">${esc(m.content)}</div>
    <div class="meta">
      <span class="tag">${esc(m.category)}</span>
      ${m.score !== undefined ? `<span>match ${Number(m.score).toFixed(2)}</span>` : ""}
      ${m.created_at ? `<span>${esc(fmtDate(m.created_at))}</span>` : ""}
      ${m.superseded_by || m.valid_to ? `<span>superseded</span>` : ""}
      <span class="row-actions">
        <button class="quiet" data-show="${esc(m.id)}">Show on graph</button>
        <button class="quiet" data-edit="${esc(m.id)}">Edit</button>
        <button class="danger" data-forget="${esc(m.id)}">Forget</button>
      </span>
    </div>
    ${detailsMarkup(m)}
  </article>`;
}

// Density on demand (PRODUCT.md): the full record is one disclosure away,
// never in the way of reading. <details> is keyboard-reachable for free.
export function detailsMarkup(m, open = false) {
  const tags = (m.tags || []).map((t) => `<span class="chip">${esc(t)}</span>`).join("");
  const rows = META_ROWS
    .map(([label, read]) => [label, read(m)])
    .filter(([, v]) => v !== null && v !== undefined && v !== "")
    .map(([label, v]) => `<dt>${esc(label)}</dt><dd>${esc(v)}</dd>`)
    .join("");
  // Every card carries the same visible word, so the accessible name says
  // which memory; otherwise a screen reader reads "Details" a hundred times.
  const label = `Details for ${(m.summary || m.content || "this memory").slice(0, 60)}`;
  return `<details class="details"${open ? " open" : ""}>
    <summary aria-label="${esc(label)}">Details</summary>
    ${tags ? `<div class="chips">${tags}</div>` : ""}
    ${rows ? `<dl class="fields">${rows}</dl>` : ""}
  </details>`;
}

export function editorMarkup(m, projects = []) {
  const isNew = m === null;
  const chosen = m?.category || "fact";
  // A stored category the client does not know would otherwise leave no
  // option selected, and Save would silently reclassify the memory as "fact".
  const choices = CATEGORIES.includes(chosen) ? CATEGORIES : [...CATEGORIES, chosen];
  const options = choices.map(
    (c) => `<option value="${esc(c)}"${c === chosen ? " selected" : ""}>${esc(c)}</option>`).join("");
  const projectField = isNew ? `<label>Project
      <input id="f-project" list="project-ids" value="${esc(projects[0] || "default")}" maxlength="200">
      <datalist id="project-ids">${projects.map((p) => `<option value="${esc(p)}">`).join("")}</datalist>
    </label>` : "";
  return `<form class="editor" data-editor="${esc(m?.id || "")}">
      <label>Summary
        <input id="f-summary" value="${esc(m?.summary || "")}" placeholder="One line that stands alone">
      </label>
      <label>Content
        <textarea id="f-content" rows="6" placeholder="What is worth remembering?">${esc(m?.content || "")}</textarea>
      </label>
      <div class="editor-row">
        <label>Category
          <select id="f-category">${options}</select>
        </label>
        <label>Confidence
          <input id="f-confidence" type="number" min="0" max="1" step="0.05" value="${esc(m?.confidence ?? 1)}">
        </label>
        ${projectField}
      </div>
      <label>Tags
        <input id="f-tags" value="${esc((m?.tags || []).join(", "))}" placeholder="Comma separated; replaces the existing set">
      </label>
      ${isNew ? "" : `<p class="hint">Saving keeps the memory's id and history; content edits re-index it for search.</p>`}
      <div class="form-actions">
        <button type="button" class="quiet" data-cancel="1">Cancel</button>
        <button type="submit" class="primary">${isNew ? "Create memory" : "Save changes"}</button>
      </div>
  </form>`;
}

// Reads the editor; throws a user-facing Error on a bad field.
export function readEditor(isNew) {
  const body = {
    summary: $("#f-summary").value.trim(),
    content: $("#f-content").value.trim(),
    category: $("#f-category").value,
    // Tags are a set, not prose: an empty box means "no tags", a real edit.
    tags: $("#f-tags").value.split(",").map((t) => t.trim()).filter(Boolean),
  };
  if (!body.content) throw new Error("Content cannot be empty");
  // An empty number input reads as "" and Number("") is 0, so sending it
  // unconditionally would silently rewrite a 1.00 memory to 0.00.
  const raw = $("#f-confidence").value.trim();
  if (raw !== "") {
    const confidence = Number(raw);
    if (!Number.isFinite(confidence) || confidence < 0 || confidence > 1) {
      throw new Error("Confidence must be a number between 0 and 1");
    }
    body.confidence = confidence;
  }
  if (isNew) body.project_id = $("#f-project").value.trim() || "default";
  return body;
}
