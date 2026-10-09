"""The repair loop.

One attempt is not an extraction pipeline. This runs the model, validates the
result semantically, and on failure retries with the specific problems appended
to the prompt — capped, recorded, and reported.

Two things it deliberately does *not* do:

  * It does not retry forever. `max_repairs` bounds the cost, and a result that
    never validates is reported as a failure rather than passed downstream.
  * It does not silently accept a bad field. Every result carries its attempts
    and the problems found, so a caller can decide what to do about them.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Sequence

from .llm import LLM
from .schema import FieldSpec, build_json_schema, validate

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Attempt:
    """One call to the model, and what was wrong with what it returned."""

    index: int
    is_repair: bool
    output: str
    problems: list[str] = field(default_factory=list)


@dataclass
class ExtractionResult:
    data: dict[str, Any] | None
    ok: bool
    attempts: list[Attempt] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)

    @property
    def repairs(self) -> int:
        return sum(1 for attempt in self.attempts if attempt.is_repair)

    def summary(self) -> str:
        if self.ok:
            return f"ok in {len(self.attempts)} attempt(s), {self.repairs} repair(s)"
        return f"failed after {len(self.attempts)} attempt(s): {'; '.join(self.unresolved)}"


def build_prompt(document: str, fields: Sequence[FieldSpec]) -> str:
    wanted = "\n".join(
        f"- {f.name} ({f.kind}{'' if f.required else ', optional'}): {f.description}"
        for f in fields
    )
    return (
        "Extract the following fields from the document below.\n\n"
        f"Fields:\n{wanted}\n\n"
        "If a required field is genuinely not present in the document, say so "
        "rather than inventing a plausible value.\n\n"
        "Document:\n"
        '"""\n'
        f"{document}\n"
        '"""'
    )


def build_repair_prompt(original: str, previous: str, problems: Sequence[str]) -> str:
    listed = "\n".join(f"- {problem}" for problem in problems)
    return (
        f"{original}\n\n"
        "---\n"
        "Your previous response was rejected by validation.\n\n"
        f"Previous response:\n{previous}\n\n"
        f"Problems:\n{listed}\n\n"
        "Return a corrected response. Fix only these problems."
    )


class Extractor:
    """Runs a model against a document, validates the output, repairs if needed."""

    def __init__(
        self,
        llm: LLM,
        fields: Sequence[FieldSpec],
        *,
        max_repairs: int = 2,
    ) -> None:
        if not fields:
            raise ValueError("need at least one field")
        self.llm = llm
        self.fields = list(fields)
        self.max_repairs = max_repairs
        self.schema = build_json_schema(self.fields)

    def extract(self, document: str) -> ExtractionResult:
        base_prompt = build_prompt(document, self.fields)
        prompt = base_prompt
        attempts: list[Attempt] = []
        problems: list[str] = []

        for index in range(self.max_repairs + 1):
            raw = self.llm.complete(prompt, schema=self.schema)

            try:
                data = json.loads(raw)
            except json.JSONDecodeError as exc:
                problems = [f"response was not valid JSON: {exc.msg} at position {exc.pos}"]
                attempts.append(Attempt(index, index > 0, raw, problems))
            else:
                problems = validate(data, self.fields)
                attempts.append(Attempt(index, index > 0, raw, problems))
                if not problems:
                    return ExtractionResult(data=data, ok=True, attempts=attempts)

            log.warning("attempt %d rejected: %s", index + 1, "; ".join(problems))

            if index < self.max_repairs:
                prompt = build_repair_prompt(base_prompt, raw, problems)

        return ExtractionResult(
            data=None,
            ok=False,
            attempts=attempts,
            unresolved=problems,
        )
