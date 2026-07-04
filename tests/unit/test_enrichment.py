from collivind.engine.enrichment import CATEGORY_KEYWORDS, build_enriched_text
from collivind.models.memory import MemoryCategory, MemoryCreate


def _make_memory(**overrides):
    defaults = {
        "content": "We use PostgreSQL",
        "summary": "DB choice",
        "category": MemoryCategory.DECISION,
        "project_id": "proj",
    }
    defaults.update(overrides)
    return MemoryCreate(**defaults)


def test_content_always_first():
    mem = _make_memory()
    result = build_enriched_text(mem)
    assert result.startswith("We use PostgreSQL")


def test_summary_appended_when_different():
    mem = _make_memory(content="We use PostgreSQL", summary="DB choice")
    result = build_enriched_text(mem)
    assert "DB choice" in result


def test_summary_not_duplicated_when_same_as_content():
    mem = _make_memory(content="We use PostgreSQL", summary="We use PostgreSQL")
    parts = build_enriched_text(mem)
    assert parts.count("We use PostgreSQL") == 1


def test_summary_skipped_when_empty():
    mem = _make_memory(summary="")
    result = build_enriched_text(mem)
    assert result.count("|") == 1  # content | category_keywords


def test_category_keywords_included():
    for cat, keywords in CATEGORY_KEYWORDS.items():
        mem = _make_memory(category=cat, summary="")
        result = build_enriched_text(mem)
        for kw in keywords:
            assert kw in result, f"{kw} missing for {cat}"


def test_entity_names_appended():
    mem = _make_memory(summary="")
    result = build_enriched_text(mem, entity_names=["PostgreSQL", "Redis"])
    assert "PostgreSQL" in result
    assert "Redis" in result


def test_tags_appended():
    mem = _make_memory(summary="", tags=["backend", "infra"])
    result = build_enriched_text(mem)
    assert "backend" in result
    assert "infra" in result


def test_full_enrichment_order():
    mem = _make_memory(
        content="content here",
        summary="summary here",
        category=MemoryCategory.ERROR,
        tags=["debugging"],
    )
    result = build_enriched_text(mem, entity_names=["SomeService"])
    parts = result.split(" | ")
    assert parts[0] == "content here"
    assert parts[1] == "summary here"
    assert "bug" in parts[2]  # ERROR category keywords
    assert parts[3] == "SomeService"
    assert parts[4] == "debugging"


def test_no_entities_no_tags():
    mem = _make_memory(summary="", tags=[])
    result = build_enriched_text(mem)
    # Only content + category keywords
    parts = result.split(" | ")
    assert len(parts) == 2


def test_none_entity_names_ignored():
    mem = _make_memory(summary="")
    result = build_enriched_text(mem, entity_names=None)
    assert result == build_enriched_text(mem)
