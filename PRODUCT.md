# Collivind — Product context

## Register

**Product.** Design serves the task. This is a local tool for reading and
tending your own memory graph, not a surface that sells anything. The bar is
earned familiarity: it should disappear into the task.

## What it is

A local web UI over the same collivind store the agents write to. It reads
`~/.collivind/config.toml`, so whatever mode is configured — docker (Qdrant +
Neo4j) or embedded (SQLite) — the browser shows exactly what the hooks and MCP
tools see. No separate database, no sync.

## Who uses it

One person: the developer whose agents have been writing memories all day. They
arrive in one of four moods, and the UI has to serve all four without becoming
four different apps:

1. **Search and read** — "what did we decide about the storage lock?" Find it,
   read it, leave.
2. **Browse** — scan what got captured recently, with no specific target.
3. **Curate** — the store accumulates noise. Fix a wrong category, sharpen a
   summary, delete what should never have been saved.
4. **Inspect the graph** — follow entities between memories, walk a version
   chain to see what superseded what.

## The central tension

Curation and graph traversal pull toward density (more rows, tighter spacing,
table chrome). The chosen feel pulls the other way. **Reading wins by default;
density appears on demand.** The list is comfortable to read at rest, and
compacts only when the user asks for it. Editing is inline and quiet, never a
modal-first workflow.

## Personality

**Warm · curious · unhurried.**

- *Warm* — soft surfaces, generous spacing. Reading your own memory should feel
  like opening a notebook, not filing a ticket.
- *Curious* — entities and relationships are visible and clickable. The graph
  invites a detour without demanding one.
- *Unhurried* — no badges, no counters competing for attention, no urgency
  language. Nothing here is a notification.

## References

**Bear, Obsidian** — for the reading surface: comfortable measure, generous
line-height, content as the loudest element.

**Linear, Raycast** — for the interaction bar only: keyboard reachability,
instant feedback, no gratuitous motion. Borrowed for behaviour, not for looks.

## Anti-references

- **Dashboard chrome.** No stat tiles, no sparklines, no "12 memories this
  week" cards. This is not an analytics surface.
- **Data-table density as the default.** Not a spreadsheet.
- **Hard borders everywhere.** Separation through space and subtle surface
  shifts, not 1px grids.
- **Modal-first editing.** Inline and progressive; modals only for destructive
  confirmation.

## Accessibility

WCAG **AA** floor, verified rather than assumed:

- Body text ≥4.5:1, large text ≥3:1, placeholders held to the body bar
- Every action keyboard reachable, focus always visible
- `prefers-reduced-motion` honoured — transitions convey state, so they degrade
  to instant rather than disappearing
- Never colour alone for meaning; category and confidence carry text too

## Design principles

1. **The memory is the interface.** Content outranks chrome everywhere.
2. **Destructive actions are never one click.** Forgetting is permanent;
   invalidation is preferred and preserves the chain.
3. **Show what mode you're in.** The store being read is stated, not assumed —
   this UI is useless if you are unknowingly reading a different backend, which
   is a failure this project has already shipped once.
4. **Empty states teach.** An empty store explains how memories arrive, not
   "no results".
