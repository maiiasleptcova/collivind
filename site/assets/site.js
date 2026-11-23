// Copy buttons on code blocks. Hidden until JS runs and the Clipboard API exists.
for (const btn of document.querySelectorAll("[data-copy]")) {
  if (!navigator.clipboard) continue;
  btn.hidden = false;
  btn.addEventListener("click", async () => {
    const lines = document.getElementById(btn.dataset.copy).innerText.split("\n");
    try {
      await navigator.clipboard.writeText(lines.filter((l) => !l.startsWith("#")).join("\n"));
      btn.textContent = "Copied";
    } catch {
      btn.textContent = "Copy failed";
    }
    setTimeout(() => { btn.textContent = "Copy"; }, 2000);
  });
}
