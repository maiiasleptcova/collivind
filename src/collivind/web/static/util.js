// Small shared helpers. Every string that reaches innerHTML goes through esc().

export const $ = (sel, root = document) => root.querySelector(sel);
export const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

export const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

// An ISO timestamp is data, not reading material: show it the way the reader
// keeps time, and fall back to the raw string rather than "Invalid Date".
export const fmtTime = (v) => {
  if (v === null || v === undefined || v === "") return null;
  const d = new Date(v);
  return isNaN(d.getTime()) ? String(v) : d.toLocaleString();
};

export const fmtDate = (v) => {
  if (!v) return "";
  const d = new Date(v);
  return isNaN(d.getTime()) ? String(v).slice(0, 10) : d.toLocaleDateString(undefined, { day: "numeric", month: "short", year: "numeric" });
};

// Number("") and Number(null) are both 0, so an absent confidence would
// otherwise render as a confident 0.00. Reject empties before coercing.
export const fmtNum = (v) => {
  if (v === null || v === undefined || v === "") return null;
  const n = Number(v);
  return Number.isFinite(n) ? n.toFixed(2) : null;
};

export const plural = (n, word) => `${n} ${word}${n === 1 ? "" : "s"}`;

// Writes carry Content-Type: application/json; the server refuses any write
// without it (and without a same-origin Origin, which the browser adds).
export async function api(path, options = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
    body: options.body !== undefined ? JSON.stringify(options.body) : undefined,
  });
  const payload = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(payload.error || `${res.status} ${res.statusText}`);
  return payload;
}

const errorBox = () => $("#error");
export function showError(e) {
  const box = errorBox();
  $("#error-text").textContent = e?.message || String(e);
  box.hidden = false;
}
export function clearError() { errorBox().hidden = true; }

let toastTimer;
export function toast(text) {
  const el = $("#toast");
  el.textContent = text;
  el.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { el.hidden = true; }, 2600);
}

// Projects without a chosen colour get one of these, keyed by a stable hash
// of the id so a project keeps its colour across reloads. Muted mid-tones
// that read on both the light and the dark canvas.
const PALETTE = ["#4f7cac", "#b0624a", "#5e8c61", "#8a6bb0", "#b08a3e", "#3f8f8f", "#a35d84", "#6d7686"];
export function projectColor(projectId, meta) {
  const chosen = meta?.[projectId]?.color;
  if (chosen) return chosen;
  let h = 0;
  for (const ch of String(projectId)) h = (h * 31 + ch.codePointAt(0)) >>> 0;
  return PALETTE[h % PALETTE.length];
}

export const projectName = (projectId, meta) => meta?.[projectId]?.display_name || projectId;

// Only http(s) URLs become links; the server already refuses other schemes,
// this keeps a hand-edited store from turning into a javascript: href.
export const safeHref = (url) => /^https?:\/\//i.test(url || "") ? url : null;

// "uses" read from the source, "is used by" from the target (models/relations.py).
export function relPhrase(meanings, type, outgoing = true) {
  const m = meanings?.[type];
  if (!m) return type.toLowerCase().replace(/_/g, " ");
  return outgoing ? m.forward : m.inverse;
}
