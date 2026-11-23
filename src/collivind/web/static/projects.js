// Projects: what the store holds, per project, and the metadata that names
// and colours each one. A list to read, not a grid of tiles.

import { $, $$, api, esc, safeHref, toast, showError, clearError, projectColor } from "./util.js";

let ctx = null;   // { projects(): rows, reload(), openInGraph(id) }
let editing = null;

export function initProjects(context) { ctx = context; }

export function renderProjects() {
  const root = $("#projects-list");
  const rows = ctx.rows();
  if (!rows.length) {
    root.innerHTML = `<div class="empty"><h2>No projects yet</h2>
      <p>A project appears here as soon as a memory is saved under it. Agents pick the project from the
      repository they are working in; from a terminal: <code>collivind add "…" --project my-app</code></p></div>`;
    return;
  }
  const meta = Object.fromEntries(rows.map((r) => [r.project_id, r]));
  root.innerHTML = rows.map((p) => p.project_id === editing ? editor(p, meta) : row(p, meta)).join("");
  $$(".swatch", root).forEach((el) => { el.style.background = projectColor(el.dataset.project, meta); });
  $$("[data-edit-project]", root).forEach((b) => b.onclick = () => { editing = b.dataset.editProject; renderProjects(); $("#pf-name")?.focus(); });
  $$("[data-graph-project]", root).forEach((b) => b.onclick = () => ctx.openInGraph(b.dataset.graphProject));
  $$("[data-list-project]", root).forEach((b) => b.onclick = () => ctx.openInList(b.dataset.listProject));
  const form = $("form.project-editor", root);
  if (form) {
    $("[data-cancel]", form).onclick = () => { editing = null; renderProjects(); };
    $("#pf-nocolor").onclick = () => { $("#pf-color").dataset.cleared = "1"; $("#pf-color-note").textContent = "Automatic colour"; };
    $("#pf-color").oninput = () => {
      const input = $("#pf-color");
      delete input.dataset.cleared; input.dataset.touched = "1"; $("#pf-color-note").textContent = "";
    };
    form.onsubmit = (e) => { e.preventDefault(); save(form.dataset.project); };
  }
}

function row(p, meta) {
  const name = p.display_name || p.project_id;
  const superseded = p.memories - p.live;
  const href = safeHref(p.repo);
  const repo = p.repo ? (href ? `<a href="${esc(href)}" target="_blank" rel="noopener noreferrer">${esc(p.repo)}</a>` : `<code>${esc(p.repo)}</code>`) : "";
  return `<article class="project">
    <div class="project-head">
      <span class="swatch" data-project="${esc(p.project_id)}" aria-hidden="true"></span>
      <h2>${esc(name)}</h2>
      ${p.display_name ? `<code class="project-id">${esc(p.project_id)}</code>` : ""}
    </div>
    <p class="project-counts">${p.live === 1 ? "1 current memory" : `${p.live} current memories`}${superseded ? `, ${superseded} superseded` : ""}</p>
    ${p.description ? `<p class="project-desc">${esc(p.description)}</p>` : ""}
    ${repo || p.tags.length ? `<div class="meta">${repo}${p.tags.map((t) => `<span class="chip">${esc(t)}</span>`).join("")}</div>` : ""}
    <div class="row-actions">
      <button class="quiet" data-graph-project="${esc(p.project_id)}">Open graph</button>
      <button class="quiet" data-list-project="${esc(p.project_id)}">Read memories</button>
      <button class="quiet" data-edit-project="${esc(p.project_id)}">Edit details</button>
    </div>
  </article>`;
}

function editor(p, meta) {
  const color = p.color || projectColor(p.project_id, meta);
  return `<article class="project">
    <form class="editor project-editor" data-project="${esc(p.project_id)}">
      <div class="project-head"><h2>${esc(p.project_id)}</h2></div>
      <div class="editor-row">
        <label>Display name <input id="pf-name" maxlength="200" value="${esc(p.display_name)}" placeholder="${esc(p.project_id)}"></label>
        <label>Colour on the graph
          <span class="color-field"><input id="pf-color" type="color" value="${esc(color)}">
          <button type="button" class="quiet" id="pf-nocolor">Use automatic</button>
          <span class="hint" id="pf-color-note">${p.color ? "" : "Automatic colour"}</span></span>
        </label>
      </div>
      <label>Description <textarea id="pf-desc" rows="3" maxlength="4000">${esc(p.description)}</textarea></label>
      <label>Repository <input id="pf-repo" maxlength="1000" value="${esc(p.repo)}" placeholder="https://github.com/you/repo or /path/to/checkout"></label>
      <label>Tags <input id="pf-tags" value="${esc(p.tags.join(", "))}" placeholder="Comma separated"></label>
      <div class="form-actions">
        <button type="button" class="quiet" data-cancel="1">Cancel</button>
        <button type="submit" class="primary">Save project</button>
      </div>
    </form>
  </article>`;
}

const hadColor = (projectId) => Boolean(ctx.rows().find((r) => r.project_id === projectId)?.color);

async function save(projectId) {
  const colorInput = $("#pf-color");
  const body = {
    display_name: $("#pf-name").value.trim(),
    description: $("#pf-desc").value.trim(),
    repo: $("#pf-repo").value.trim(),
    tags: $("#pf-tags").value.split(",").map((t) => t.trim()).filter(Boolean),
    // Untouched and never chosen stays automatic, so a palette change later still applies.
    color: colorInput.dataset.cleared ? "" : colorInput.dataset.touched || hadColor(projectId) ? colorInput.value : "",
  };
  try {
    clearError();
    await api(`/api/project?id=${encodeURIComponent(projectId)}`, { method: "PATCH", body });
    editing = null;
    toast("Project saved");
    await ctx.reload();
  } catch (e) { showError(e); }
}
