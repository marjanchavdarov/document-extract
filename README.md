# document-extract

Turn documents into typed records — and prove the extraction actually worked.

The claim: **structured outputs guarantee the shape of a response, not its
content.** A model constrained to a JSON schema will always return the right
keys and the right types. It will still, sometimes, return `0` for a number it
couldn't find and `"N/A"` for a date it didn't read. Both validate perfectly.
Both are wrong.

That gap is the entire job, and this repo is built around it.

---

## See it in one command

```bash
python -m document_extract.cli demo
```

No API key, no network — the demo uses a scripted model so the failure is
reproducible.

```
Attempt 1 — schema-valid, semantically empty
The model returns:
  {"contracting_authority": "Ministry of Infrastructure and Energy",
   "estimated_value_eur": 0, "submission_deadline": "N/A", "is_cancelled": false}

Validation rejects it:
  - estimated_value_eur: 0 is below the minimum 1.0
  - submission_deadline: required, but the model returned 'N/A' — extract the
    real value or report it as absent

Attempt 2 — the repair prompt carries the specifics
  Your previous response was rejected by validation.
  ...
  Problems:
  - estimated_value_eur: 0 is below the minimum 1.0
  - submission_deadline: required, but the model returned 'N/A' ...

Result
  ok in 2 attempt(s), 1 repair(s)

  parsed record:
    contracting_authority    'Ministry of Infrastructure and Energy'
    estimated_value_eur      1250000.0
    submission_deadline      '14 November 2026, 12:00 local time'
    is_cancelled             False

  requests sent to the model: 2
```

That first response is the dangerous one. Every key present, every type correct,
valid JSON — a naive pipeline writes it to the database as a contract worth
**EUR 0 with no deadline**, and nothing raises an error. The row is simply wrong,
and it stays wrong silently.

## What's in here

| Module | Responsibility |
|---|---|
| `schema.py` | `FieldSpec` → JSON Schema, plus the semantic `validate()` |
| `llm.py` | `LLM` protocol, a `ScriptedLLM` for tests, a `ClaudeLLM` using structured output |
| `extractor.py` | The repair loop: call, validate, retry with the specific problems |

Four decisions worth calling out:

**Null sentinels are treated as missing.** `"N/A"`, `"unknown"`, `"not specified"`,
`"-"`, `""` — these pass schema validation and carry no information. A pipeline
that accepts them looks like it worked while producing nothing. See
`test_various_null_sentinels`.

**`0` is not a value.** A model that can't find a number very often returns zero.
`minimum=1.0` on a monetary field turns that into a validation failure instead of
a wrong row.

**`bool` is not an `int`.** In Python `isinstance(True, int)` is `True`, so a
boolean slips through a numeric check unless you explicitly exclude it. It did,
in the first version of this code.

**Repair prompts name the problem.** "Try again" wastes an attempt. The retry
sends back the previous response *and* the specific failures, which is why
`test_repair_prompt_carries_the_specific_problem` asserts on the prompt text.

## Install and run

```bash
pip install -r requirements.txt

python -m document_extract.cli demo          # offline
python -m pytest -q                          # 23 tests, no API key

# real extraction (needs ANTHROPIC_API_KEY)
python -m document_extract.cli extract fixtures/sample_tender.txt
```

## Using it

```python
from document_extract import Extractor, ClaudeLLM, FieldSpec

fields = [
    FieldSpec("contracting_authority", "The body issuing the notice."),
    FieldSpec("estimated_value_eur", "Estimated value in EUR.",
              kind="float", minimum=1.0),
    FieldSpec("is_cancelled", "True only if the notice says so.", kind="bool"),
    FieldSpec("notes", "Optional context.", required=False),
]

result = Extractor(ClaudeLLM(), fields, max_repairs=2).extract(document_text)

if result.ok:
    store(result.data)
else:
    queue_for_review(result.unresolved, result.attempts)
```

Every result carries its `attempts` and, on failure, the `unresolved` problems —
so the caller can route bad extractions somewhere visible instead of writing them
down and hoping.

## Notes and limits

The API call uses `output_config.format` with a JSON Schema, so the response is
constrained to the right shape at generation time. That makes the JSON-parse
failure path rare — it is still handled, because "rare" is not "never", and a
pipeline that assumes well-formed JSON is a pipeline that stops on a bad night.

Retries for 429 and 5xx are the SDK's own (`max_retries`), not hand-rolled —
it already does exponential backoff, and a second implementation on top just
doubles the delay.

The repair loop targets *semantic* failures: missing required values, out-of-range
numbers, the right key holding an empty string. It cannot tell you a number is
factually wrong — only that it isn't obviously absent. Extraction quality above
that line is a prompting and eval problem, not a validation one.
