"""The repair loop, end to end, with a scripted model — no API key needed."""

from __future__ import annotations

import json

from document_extract.extractor import Extractor, build_prompt, build_repair_prompt
from document_extract.llm import ScriptedLLM
from document_extract.schema import FieldSpec

FIELDS = [
    FieldSpec("authority", "Issuing body."),
    FieldSpec("value", "Contract value in EUR.", kind="float", minimum=1.0),
]

GOOD = json.dumps({"authority": "Ministry", "value": 1250000.0})
EMPTY_VALUE = json.dumps({"authority": "Ministry", "value": 0})
NULL_DEADLINE = json.dumps({"authority": "N/A", "value": 1250000.0})
NOT_JSON = "Sure! Here is the data you asked for:"


def test_single_attempt_when_valid():
    llm = ScriptedLLM([GOOD])
    result = Extractor(llm, FIELDS).extract("doc")

    assert result.ok
    assert result.data == {"authority": "Ministry", "value": 1250000.0}
    assert len(result.attempts) == 1
    assert result.repairs == 0
    assert len(llm.prompts) == 1


def test_repairs_on_semantic_failure():
    """The headline case: valid JSON, correct types, useless content."""
    llm = ScriptedLLM([EMPTY_VALUE, GOOD])
    result = Extractor(llm, FIELDS).extract("doc")

    assert result.ok
    assert result.data["value"] == 1250000.0
    assert len(result.attempts) == 2
    assert result.repairs == 1
    assert result.attempts[0].problems          # first attempt recorded why
    assert result.attempts[1].problems == []


def test_repair_prompt_carries_the_specific_problem():
    """A generic 'try again' wastes an attempt; the model needs the detail."""
    llm = ScriptedLLM([EMPTY_VALUE, GOOD])
    Extractor(llm, FIELDS).extract("doc")

    repair_prompt = llm.prompts[1]
    assert "below the minimum" in repair_prompt
    assert "value" in repair_prompt
    assert EMPTY_VALUE in repair_prompt


def test_repairs_on_invalid_json():
    llm = ScriptedLLM([NOT_JSON, GOOD])
    result = Extractor(llm, FIELDS).extract("doc")

    assert result.ok
    assert result.repairs == 1
    assert any("not valid JSON" in p for p in result.attempts[0].problems)


def test_gives_up_after_max_repairs():
    llm = ScriptedLLM([NULL_DEADLINE])  # always the same bad response
    result = Extractor(llm, FIELDS, max_repairs=2).extract("doc")

    assert not result.ok
    assert result.data is None
    assert len(result.attempts) == 3            # initial + 2 repairs
    assert len(llm.prompts) == 3
    assert result.unresolved                    # and it says why


def test_zero_repairs_means_one_attempt():
    llm = ScriptedLLM([EMPTY_VALUE, GOOD])
    result = Extractor(llm, FIELDS, max_repairs=0).extract("doc")

    assert not result.ok
    assert len(llm.prompts) == 1


def test_result_reports_attempts_and_repairs():
    llm = ScriptedLLM([EMPTY_VALUE, NOT_JSON, GOOD])
    result = Extractor(llm, FIELDS, max_repairs=3).extract("doc")

    assert result.ok
    assert len(result.attempts) == 3
    assert result.repairs == 2
    assert "[1" not in result.summary()          # summary is readable
    assert "ok in 3 attempt(s), 2 repair(s)" == result.summary()


def test_schema_is_passed_to_every_call():
    llm = ScriptedLLM([EMPTY_VALUE, GOOD])
    Extractor(llm, FIELDS).extract("doc")

    assert len(llm.schemas) == 2
    for schema in llm.schemas:
        assert schema["additionalProperties"] is False
        assert set(schema["required"]) == {"authority", "value"}


def test_empty_field_list_is_rejected():
    import pytest

    with pytest.raises(ValueError):
        Extractor(ScriptedLLM([GOOD]), [])


def test_prompt_includes_field_descriptions():
    prompt = build_prompt("some document", FIELDS)
    assert "authority" in prompt
    assert "Contract value in EUR." in prompt
    assert "some document" in prompt


def test_repair_prompt_includes_previous_response_and_problems():
    prompt = build_repair_prompt("base", "prev", ["authority: missing"])
    assert "base" in prompt
    assert "prev" in prompt
    assert "authority: missing" in prompt
