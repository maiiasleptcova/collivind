"""The memory wiki as Open Knowledge Format (OKF v0.2) Markdown.

Spec: github.com/GoogleCloudPlatform/knowledge-catalog, okf/SPEC.md. A bundle
is a directory of Markdown files with YAML frontmatter (only `type` is
required); links between files are relative Markdown links; each directory
may carry an `index.md` with no frontmatter (the root one may carry
`okf_version`). OpenWiki (langchain-ai/openwiki) reads such a bundle with
`openwiki visualize`, which is how this export was checked.

One file per topic (web/wiki.py: entities, summaries, projects). A memory is
not a page: on a 3,000-memory store that would be ~3,000 files beside ~200
topics, and a graph of them is the hairball the topic pages replace. Each
memory's full text appears once, on its home page (its most specific topic,
else its summary, else its project), under an anchor; every other page that
lists it links there. Output is deterministic: same store, same bytes. No
clock is read; `generated.at` is the newest memory on the page.
"""

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from collivind.version import __version__
from collivind.web.wiki import MAX_PAGE, PROJECT_PREFIX, Topic, Wiki

OKF_VERSION = "0.2"
MARKER = ".collivind-okf"  # an export directory collivind may rewrite
MANAGED = ("entities", "summaries", "projects")
RESERVED = {"index", "log", "instructions"}
DIRS = {"entity": "entities", "summary": "summaries", "project": "projects"}
TYPE_NAMES = {"summary": "Summary", "project": "Project"}

# Memory text is untrusted Markdown: a "# heading" or "](x.md)" in it would
# break the page or make a link. Escape what Markdown (and OpenWiki's link
# scanner, which also reads code fences) would act on.
_INLINE = re.compile(r"([\\`*_\[\]()<>|~#!])")
_LEADING = re.compile(r"^(\s*)([-+=>]|\d+[.)])")


def esc(text: str) -> str:
    lines = (text or "").splitlines() or [""]
    return "  \n".join(_LEADING.sub(lambda m: m.group(1) + "\\" + m.group(2), _INLINE.sub(r"\\\1", ln)) for ln in lines)


def _one_line(text: str) -> str:
    return esc(" ".join((text or "").split()))


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")[:60].strip("-") or "topic"


def type_name(topic: Topic) -> str:
    return TYPE_NAMES.get(topic.type) or topic.type.replace("_", " ").title()


def anchor(memory_id: str) -> str:
    return "m-" + re.sub(r"[^a-z0-9]", "", memory_id.lower())[:12]


def paths(wiki: Wiki) -> Dict[str, str]:
    """Topic id -> bundle path, deterministic and collision-free."""
    out: Dict[str, str] = {}
    taken: set = set()
    for t in sorted(wiki.topics.values(), key=lambda t: t.id):
        if t.kind == "summary":
            first = wiki.memories[t.id]
            stem = f"{first.created_at.date().isoformat() if first.created_at else 'undated'}-{_slug(t.title)[:48]}"
        elif t.kind == "project":
            stem = _slug(t.id[len(PROJECT_PREFIX) :])
        else:
            stem = _slug(t.title)
        if stem in RESERVED:
            stem += "-topic"
        path = f"{DIRS[t.kind]}/{stem}.md"
        if path in taken:
            path = f"{DIRS[t.kind]}/{stem}-{hashlib.sha1(t.id.encode()).hexdigest()[:6]}.md"
        taken.add(path)
        out[t.id] = path
    return out


def homes(wiki: Wiki) -> Dict[str, str]:
    """Memory id -> the one page that carries its full text."""
    out: Dict[str, str] = {}
    for i in sorted(wiki.live):
        if str(getattr(wiki.memories[i].category, "value", wiki.memories[i].category)) == "summary":
            continue  # a summary's text is its own page
        topic = next((e for e in wiki.attached.get(i, ()) if e in wiki.topics), None)
        topic = topic or next((s for s in sorted(wiki.covered_by.get(i, ())) if s in wiki.topics), None)
        pid = PROJECT_PREFIX + wiki.memories[i].project_id
        out[i] = topic or (pid if pid in wiki.topics else "")
    return out


def _rel(src: str, dst: str) -> str:
    """Relative link from one bundle file to another (OpenWiki resolves only relative links)."""
    a, b = src.split("/")[:-1], dst.split("/")
    common = 0
    while common < len(a) and common < len(b) - 1 and a[common] == b[common]:
        common += 1
    return "/".join([".."] * (len(a) - common) + b[common:])


_PLAIN = re.compile(r"^[A-Za-z0-9/][A-Za-z0-9 ./_()+,-]*$")
_YAML_WORDS = {"true", "false", "yes", "no", "on", "off", "null", "~"}


def _scalar(value: str) -> str:
    """A YAML scalar: plain when nothing in it means anything to YAML, else quoted."""
    if _PLAIN.match(value) and value.lower() not in _YAML_WORDS and not re.fullmatch(r"[\d.+-]+", value):
        return value
    # single quotes have no escapes but '' in YAML, and OpenWiki's reader only strips the outer quotes,
    # so a value without ' reads the same in both; JSON (valid YAML) for the rest
    if "'" not in value and "\\" not in value:
        return f"'{value}'"
    return json.dumps(value, ensure_ascii=False)


def _frontmatter(fields: Dict[str, Any]) -> str:
    """The small YAML subset OKF pages need: strings, lists of strings, mappings, lists of mappings.

    Written by hand because PyYAML is not a runtime dependency; the tests parse it with PyYAML.
    """
    out = ["---"]
    for key, value in fields.items():
        if isinstance(value, str):
            out.append(f"{key}: {_scalar(value)}")
        elif isinstance(value, dict):
            out.append(f"{key}:")
            out += [f"  {k}: {_scalar(v)}" for k, v in value.items()]
        elif value and all(isinstance(v, dict) for v in value):
            out.append(f"{key}:")
            for item in value:
                pairs = [f"{k}: {_scalar(v)}" for k, v in item.items()]
                out += [f"  - {pairs[0]}"] + [f"    {p}" for p in pairs[1:]]
        else:
            out.append(f"{key}:")
            out += [f"  - {_scalar(v)}" for v in value]
    return "\n".join(out) + "\n---\n"


class _Page:
    def __init__(self, wiki: Wiki, where: Dict[str, str], home: Dict[str, str], topic_id: str):
        self.wiki, self.where, self.home, self.id = wiki, where, home, topic_id
        self.path = where[topic_id]

    def link(self, topic_id: str, text: str) -> str:
        return (
            f"[{_one_line(text)}]({_rel(self.path, self.where[topic_id])})"
            if topic_id in self.where
            else _one_line(text)
        )

    def memory_link(self, row: Dict[str, Any]) -> str:
        home = self.home.get(row["id"], "")
        target = f"#{anchor(row['id'])}" if home == self.id else (
            f"{_rel(self.path, self.where[home])}#{anchor(row['id'])}" if home in self.where else ""
        )  # fmt: skip
        return f"[{_one_line(row['label'])}]({target})" if target else f"**{_one_line(row['label'])}**"

    def row(self, row: Dict[str, Any], lead: str = "- ") -> List[str]:
        bits = [f"{lead}{self.memory_link(row)} · {row['kind']} · {row['date']}"]
        if row["gist"]:
            bits[0] += f" — {_one_line(row['gist'])}"
        about = [self.link(t["id"], t["title"]) for t in row["about"] + row["summaries"]]
        if about:
            bits.append(f"  - also under {', '.join(about)}")
        for ln in row["links"]:
            if ln["type"] in ("RESOLVES", "CAUSED_BY"):
                continue  # shown as the pair itself
            note = f": {_one_line(ln['note'])}" if ln["note"] else ""
            bits.append(f"  - {ln['phrase']} {self.memory_link(ln['memory'])}{note}")
        return bits

    def render(self) -> Tuple[str, List[str]]:
        """The page text and the memory ids whose full text it carries."""
        wiki = self.wiki
        topic = wiki.topics[self.id]
        page = wiki.page(self.id, limit=MAX_PAGE * 1000) or {}
        own = [i for i in topic.members if self.home.get(i) == self.id]
        stamps = sorted(
            wiki.memories[i].updated_at or wiki.memories[i].created_at
            for i in topic.members
            if wiki.memories[i].created_at
        )
        fields: Dict[str, Any] = {
            "type": type_name(topic),
            "title": topic.title,
            "description": " ".join(topic.description.split()) or f"{topic.title}.",
            "tags": sorted(
                {
                    "collivind",
                    topic.type,
                    *(r["kind"].lower().replace(" ", "-") for s in page["sections"] for r in s["items"]),
                }
            ),
            "generated": {
                "by": f"collivind/{__version__}",
                **({"at": stamps[-1].strftime("%Y-%m-%dT%H:%M:%SZ")} if stamps else {}),
            },
        }
        if topic.described_by == "agent" and topic.kind != "summary":
            fields["description_by"] = "agent"
        if own:
            fields["sources"] = [
                {"id": anchor(i), "resource": f"collivind://memory/{i}", "title": wiki.memories[i].summary or anchor(i)}
                for i in own
            ]
        out = [
            f"# {_one_line(topic.title)}",
            "",
            f"*{type_name(topic)} · {_one_line(page['stats'])}*",
            "",
            _one_line(topic.description),
            "",
        ]
        if topic.kind == "summary":
            out += ["## What it says", ""] + [f"- {_one_line(f)}" for f in page.get("facts", [])] + [""]
        if page["related"]:
            out += ["## Related topics", ""]
            out += [f"- {self.link(r['id'], r['title'])} — {_one_line(r['reason'])}" for r in page["related"]] + [""]
        if topic.kind == "project" and page.get("unfiled"):
            out += [
                f"The memories below are about no topic ({page['unfiled']} of {len(topic.members)}); "
                "the rest are on the topic pages above.",
                "",
            ]
        for section in page["sections"]:
            out += [f"## {section['title']}", ""]
            for item in section["items"]:
                out += self.row(item)
                for fix in item.get("fixes", []):
                    out += self.row(fix, "  - **Fixed by** ")[:1] + (
                        [f"    - {_one_line(fix['note'])}"] if fix["note"] else []
                    )
                for cause in item.get("causes", []):
                    out += self.row(cause, "  - **Caused by** ")[:1] + (
                        [f"    - {_one_line(cause['note'])}"] if cause["note"] else []
                    )
                if section["key"] == "problems" and not item.get("fixes"):
                    out.append("  - No fix recorded yet.")
            out.append("")
        if own:
            out += ["## Full text", ""]
            for i in own:
                m = wiki.memories[i]
                day = m.created_at.date().isoformat() if m.created_at else ""
                where = f"{wiki.memory_row(i)['kind']} · {day} · project {_one_line(m.project_id)} · `{i}`"
                out += [f'<a id="{anchor(i)}"></a>', f"### {_one_line(m.summary or anchor(i))}", "", f"*{where}*", ""]
                out += [esc(m.content), ""]
        if page["backlinks"]:
            out += ["## Linked from", "", ", ".join(self.link(b["id"], b["title"]) for b in page["backlinks"]), ""]
        return _frontmatter(fields) + "\n" + "\n".join(out).rstrip() + "\n", own


def _index(entries: List[Tuple[str, str, str, str]], heading: str) -> str:
    """entries: (section, title, link, description); an index has no frontmatter (OKF §8)."""
    out: List[str] = []
    last = None
    for section, title, link, description in entries:
        if section != last:
            out += ([""] if out else []) + [f"# {section}", ""]
            last = section
        out.append(f"* [{_one_line(title)}]({link}) - {_one_line(description)}")
    return "\n".join(out or [f"# {heading}", "", "Nothing here yet."]) + "\n"


def render_bundle(wiki: Wiki) -> Dict[str, str]:
    """Bundle path -> file text, for every page and index."""
    where = paths(wiki)
    home = homes(wiki)
    files: Dict[str, str] = {}
    for topic_id in sorted(wiki.topics):
        files[where[topic_id]], _ = _Page(wiki, where, home, topic_id).render()
    for kind, folder in DIRS.items():
        topics = sorted(
            (t for t in wiki.topics.values() if t.kind == kind), key=lambda t: (type_name(t), t.title.lower(), t.id)
        )
        entries = [
            (
                type_name(t) if kind == "entity" else folder.title(),
                f"{t.title} ({len(t.members)})",
                where[t.id][len(folder) + 1 :],
                " ".join(t.description.split()),
            )
            for t in topics
        ]
        files[f"{folder}/index.md"] = _index(entries, folder.title())
    counts = {kind: sum(1 for t in wiki.topics.values() if t.kind == kind) for kind in DIRS}
    scope = f"project {wiki.project}" if wiki.project else "every project"
    root = [
        ("Collivind memory", "Projects", "projects/index.md", f"{counts['project']} projects in {scope}"),
        (
            "Collivind memory",
            "Topics",
            "entities/index.md",
            f"{counts['entity']} things memories are about: services, tools, files, ideas",
        ),
        (
            "Collivind memory",
            "Summaries",
            "summaries/index.md",
            f"{counts['summary']} agent-written summaries of sessions and topics",
        ),
    ]
    files["index.md"] = _frontmatter({"okf_version": OKF_VERSION}) + "\n" + _index(root, "Collivind memory")
    return files


def write_bundle(wiki: Wiki, out_dir: Path) -> Dict[str, int]:
    """Write the bundle into `out_dir`, replacing a previous collivind export there; refuses any other non-empty dir."""
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()) and not (out_dir / MARKER).exists():
        raise FileExistsError(f"{out_dir} is not empty and is not a collivind export; pick an empty directory")
    files = render_bundle(wiki)
    out_dir.mkdir(parents=True, exist_ok=True)
    for folder in MANAGED:  # a topic that went away must not leave its old page behind
        for old in sorted((out_dir / folder).glob("*.md")) if (out_dir / folder).is_dir() else []:
            old.unlink()
    for rel, text in files.items():
        target = out_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(text.encode("utf-8"))
    (out_dir / MARKER).write_text(
        f"Written by collivind export --okf. Re-exporting here replaces {', '.join(MANAGED)} and index.md.\n"
    )
    return {kind: sum(1 for t in wiki.topics.values() if t.kind == kind) for kind in DIRS} | {"files": len(files)}


def export_okf(graph, out_dir: Path, project: Optional[str] = None) -> Dict[str, int]:
    return write_bundle(Wiki(graph, project), out_dir)
