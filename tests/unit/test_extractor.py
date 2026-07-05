import json

from collivind.engine.extractor import (
    build_extraction_prompt,
    extraction_results_to_add_args,
    parse_extraction_response,
)


def test_build_prompt_includes_text():
    prompt = build_extraction_prompt("We use FastAPI", project_id="myproj")
    assert "We use FastAPI" in prompt
    assert "myproj" in prompt


def test_build_prompt_includes_categories():
    prompt = build_extraction_prompt("test")
    assert "architecture" in prompt
    assert "decision" in prompt
    assert "error" in prompt


def test_build_prompt_includes_entity_types():
    prompt = build_extraction_prompt("test")
    assert "library" in prompt
    assert "service" in prompt


def test_parse_valid_response():
    response = json.dumps(
        [
            {
                "content": "FastAPI is our web framework",
                "summary": "FastAPI choice",
                "category": "architecture",
                "confidence": 0.95,
                "tags": ["backend"],
                "entities": [{"name": "FastAPI", "type": "library"}],
            }
        ]
    )
    results = parse_extraction_response(response)
    assert len(results) == 1
    assert results[0].content == "FastAPI is our web framework"
    assert results[0].category == "architecture"
    assert results[0].confidence == 0.95
    assert results[0].entities[0]["name"] == "FastAPI"


def test_parse_response_with_markdown_fence():
    response = "```json\n" + json.dumps([{"content": "test fact", "summary": "test", "category": "fact"}]) + "\n```"
    results = parse_extraction_response(response)
    assert len(results) == 1
    assert results[0].content == "test fact"


def test_parse_response_invalid_json():
    results = parse_extraction_response("this is not json")
    assert results == []


def test_parse_response_empty_array():
    results = parse_extraction_response("[]")
    assert results == []


def test_parse_response_fixes_invalid_category():
    response = json.dumps([{"content": "test", "summary": "s", "category": "invalid_cat"}])
    results = parse_extraction_response(response)
    assert results[0].category == "fact"


def test_parse_response_fixes_invalid_entity_type():
    response = json.dumps(
        [
            {
                "content": "test",
                "summary": "s",
                "category": "fact",
                "entities": [{"name": "X", "type": "invalid_type"}],
            }
        ]
    )
    results = parse_extraction_response(response)
    assert results[0].entities[0]["type"] == "concept"


def test_parse_response_clamps_confidence():
    response = json.dumps([{"content": "test", "summary": "s", "category": "fact", "confidence": 5.0}])
    results = parse_extraction_response(response)
    assert results[0].confidence == 1.0


def test_parse_response_skips_empty_content():
    response = json.dumps(
        [
            {"content": "", "summary": "s", "category": "fact"},
            {"content": "valid", "summary": "s", "category": "fact"},
        ]
    )
    results = parse_extraction_response(response)
    assert len(results) == 1


def test_extraction_results_to_add_args():
    response = json.dumps(
        [
            {
                "content": "We use Redis for caching",
                "summary": "Redis caching",
                "category": "architecture",
                "tags": ["cache"],
                "entities": [{"name": "Redis", "type": "service"}],
            }
        ]
    )
    results = parse_extraction_response(response)
    args = extraction_results_to_add_args(results, project_id="proj", session_id="sess-1")
    assert len(args) == 1
    assert args[0]["project_id"] == "proj"
    assert args[0]["session_id"] == "sess-1"
    assert args[0]["entities"][0]["name"] == "Redis"


def test_parse_single_object_not_array():
    response = json.dumps({"content": "single", "summary": "s", "category": "fact"})
    results = parse_extraction_response(response)
    assert len(results) == 1
