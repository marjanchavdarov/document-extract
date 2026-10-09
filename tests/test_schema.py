"""Schema building and semantic validation."""

from __future__ import annotations

from document_extract.schema import FieldSpec, build_json_schema, validate

FIELDS = [
    FieldSpec("authority", "Issuing body."),
    FieldSpec("value", "Contract value.", kind="float", minimum=1.0),
    FieldSpec("deadline", "Submission deadline."),
    FieldSpec("cancelled", "Cancelled flag.", kind="bool"),
]


def test_schema_is_strict():
    schema = build_json_schema(FIELDS)
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {"authority", "value", "deadline", "cancelled"}
    assert schema["properties"]["value"]["type"] == "number"
    assert schema["properties"]["cancelled"]["type"] == "boolean"


def test_optional_field_allows_null():
    schema = build_json_schema(
        [FieldSpec("note", "Optional note.", required=False)]
    )
    assert schema["properties"]["note"]["type"] == ["string", "null"]


def test_valid_payload_has_no_problems():
    problems = validate(
        {"authority": "Ministry", "value": 1200.0, "deadline": "2026-11-14", "cancelled": False},
        FIELDS,
    )
    assert problems == []


def test_null_sentinel_is_treated_as_missing():
    """The core failure mode: a schema-valid 'N/A' is not a value."""
    problems = validate(
        {"authority": "Ministry", "value": 1200.0, "deadline": "N/A", "cancelled": False},
        FIELDS,
    )
    assert len(problems) == 1
    assert "deadline" in problems[0]


def test_various_null_sentinels():
    for sentinel in ["N/A", "n/a", "", "  ", "unknown", "not specified", "-", None, "TBD"]:
        problems = validate(
            {"authority": sentinel, "value": 10.0, "deadline": "x", "cancelled": False},
            FIELDS,
        )
        assert problems, f"{sentinel!r} should not pass as a value"


def test_zero_value_below_minimum_is_caught():
    """A model that cannot find a number frequently returns 0, which passes a
    naive schema check and is almost never the real answer."""
    problems = validate(
        {"authority": "M", "value": 0, "deadline": "x", "cancelled": False},
        FIELDS,
    )
    assert any("below the minimum" in p for p in problems)


def test_above_maximum_is_caught():
    fields = [FieldSpec("score", "A score.", kind="int", maximum=10)]
    problems = validate({"score": 11}, fields)
    assert any("above the maximum" in p for p in problems)


def test_bool_is_not_accepted_as_a_number():
    """True is an int in Python — without an explicit guard it validates."""
    problems = validate(
        {"authority": "M", "value": True, "deadline": "x", "cancelled": False},
        FIELDS,
    )
    assert any("value" in p for p in problems)


def test_string_is_not_accepted_as_a_number():
    problems = validate(
        {"authority": "M", "value": "1200", "deadline": "x", "cancelled": False},
        FIELDS,
    )
    assert any("expected float" in p for p in problems)


def test_missing_key_is_reported():
    problems = validate({"authority": "M", "value": 5.0, "cancelled": False}, FIELDS)
    assert any("deadline" in p and "missing" in p for p in problems)


def test_optional_field_may_be_absent():
    fields = [
        FieldSpec("required_one", "Needed."),
        FieldSpec("optional_one", "Nice to have.", required=False),
    ]
    assert validate({"required_one": "yes"}, fields) == []


def test_non_object_response_is_rejected():
    problems = validate(["not", "an", "object"], FIELDS)
    assert problems and "expected an object" in problems[0]
