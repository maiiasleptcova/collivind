from collivind.engine.enrichment import QUERY_EXPANSIONS, build_enriched_query


def test_basic_query_unchanged():
    result = build_enriched_query("what framework do we use")
    assert result.startswith("what framework do we use")


def test_trigger_word_expands():
    result = build_enriched_query("db migration")
    assert "database" in result
    assert "storage" in result


def test_multiple_triggers():
    result = build_enriched_query("api auth setup")
    assert "endpoint" in result
    assert "authentication" in result


def test_category_keywords_added():
    result = build_enriched_query("how to fix", category="error")
    assert "bug" in result
    assert "debugging" in result


def test_invalid_category_ignored():
    result = build_enriched_query("test query", category="nonexistent")
    assert result.startswith("test query")


def test_tags_appended():
    result = build_enriched_query("query", tags=["backend", "api"])
    assert "backend" in result
    assert "api" in result


def test_entity_names_appended():
    result = build_enriched_query("query", entity_names=["PostgreSQL"])
    assert "PostgreSQL" in result


def test_no_expansion_for_plain_query():
    result = build_enriched_query("what color is the sky")
    parts = result.split(" | ")
    assert len(parts) == 1


def test_all_triggers_have_expansions():
    for trigger, expansions in QUERY_EXPANSIONS.items():
        assert len(expansions) >= 2, f"trigger '{trigger}' needs at least 2 expansions"
