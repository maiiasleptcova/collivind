// Turn search results into one topic: tick the memories that belong to a
// subject, name it, and every ticked memory is linked ABOUT one entity. A
// subject the extractor never named (a client site, an incident) then opens
// as a single node with its whole history.

import { $, $$, api, esc, toast, showError, clearError } from "./util.js";

const TYPES = ["concept", "project", "service", "person", "tool", "library", "file"];

export function topicBar(total, picked) {
  const types = TYPES.map((t) => `<option value="${t}">${t}</option>`).join("");
  return `<form class="topic-bar" id="topic-bar" aria-label="Link search results to a topic">
    <span class="topic-count"><strong id="topic-picked">${picked}</strong> of ${total} ticked</span>
    <button type="button" class="quiet" data-pick-all="1">All</button>
    <button type="button" class="quiet" data-pick-none="1">None</button>
    <input id="topic-name" placeholder="Topic, e.g. Harley Street Skin" maxlength="200" required>
    <select id="topic-type" aria-label="Topic type">${types}</select>
    <button type="submit" class="primary">Link to topic</button>
  </form>`;
}

export function pickBox(id, checked) {
  return `<label class="pick"><input type="checkbox" data-pick="${esc(id)}"${checked ? " checked" : ""}>
    <span class="sr-only">Include in topic</span></label>`;
}

// picked: a Set of memory ids, owned by the caller; onLinked(result) after a save.
export function wireTopic(root, picked, onLinked) {
  const form = $("#topic-bar", root);
  if (!form) return;
  const count = () => { $("#topic-picked", form).textContent = String(picked.size); };
  const boxes = $$("[data-pick]", root);
  boxes.forEach((b) => b.onchange = () => { b.checked ? picked.add(b.dataset.pick) : picked.delete(b.dataset.pick); count(); });
  const setAll = (on) => boxes.forEach((b) => { b.checked = on; on ? picked.add(b.dataset.pick) : picked.delete(b.dataset.pick); count(); });
  $("[data-pick-all]", form).onclick = () => setAll(true);
  $("[data-pick-none]", form).onclick = () => setAll(false);
  form.onsubmit = async (e) => {
    e.preventDefault();
    if (!picked.size) { showError(new Error("Tick at least one memory")); return; }
    try {
      clearError();
      const body = { name: $("#topic-name", form).value.trim(), type: $("#topic-type", form).value, memory_ids: [...picked] };
      const result = await api("/api/topics", { method: "POST", body });
      toast(`Linked ${result.linked} memories to ${result.name}`);
      onLinked(result);
    } catch (err) { showError(err); }
  };
}
