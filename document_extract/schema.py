"""Field specifications, the JSON Schema derived from them, and semantic validation.

The important distinction this module exists to draw:

    Structured outputs guarantee the *shape* of the response. They guarantee
    nothing about its *content*.

A model constrained to a schema will always return an object with the right
keys and the right types. It will happily return `"min_turnover": 0` for a field
it could not find, or `"licence_number": "N/A"`. Both validate perfectly and both
are wrong. That gap is what `validate()` is for.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Sequence

Kind = Literal["str", "int", "float", "bool"]

JSON_TYPES: dict[str, str] = {
    "str": "string",
    "int": "integer",
    "float": "number",
    "bool": "boolean",
}

# What a model writes when it means "I could not find this". These pass schema
# validation and carry no information — the single most common way an
# extraction pipeline looks like it worked while producing nothing.
NULL_SENTINELS: frozenset[str] = frozenset(
    {
        "", "-", "—", "n/a", "n.a.", "na", "none", "null", "nil", "unknown",
        "not specified", "not stated", "not available", "not found",
        "no information", "tbd", "unspecified",
    }
)


@dataclass(frozen=True)
class FieldSpec:
    """One thing to pull out of a document."""

    name: str
    description: str
    kind: Kind = "str"
    required: bool = True
    minimum: float | None = None
    maximum: float | None = None


def build_json_schema(fields: Sequence[FieldSpec]) -> dict[str, Any]:
    """JSON Schema for structured output.

    Every field is listed in `required` because the API needs it there, but
    optional fields allow `null` so the model can decline explicitly rather than
    inventing a value or writing "N/A".
    """
    properties: dict[str, Any] = {}
    for field in fields:
        spec: dict[str, Any] = {
            "type": JSON_TYPES[field.kind],
            "description": field.description,
        }
        if field.minimum is not None:
            spec["minimum"] = field.minimum
        if field.maximum is not None:
            spec["maximum"] = field.maximum
        if not field.required:
            # Let the model say "not present" without lying.
            spec["type"] = [JSON_TYPES[field.kind], "null"]
        properties[field.name] = spec

    return {
        "type": "object",
        "properties": properties,
        "required": [f.name for f in fields],
        "additionalProperties": False,
    }


def _is_nullish(value: Any) -> bool:
    if value is None:
        return True
    return isinstance(value, str) and value.strip().lower() in NULL_SENTINELS


def _type_ok(value: Any, kind: Kind) -> bool:
    # bool is a subclass of int in Python, so check it first and exclude it
    # from the numeric kinds — otherwise True passes as a valid turnover.
    if kind == "bool":
        return isinstance(value, bool)
    if isinstance(value, bool):
        return False
    if kind == "int":
        return isinstance(value, int)
    if kind == "float":
        return isinstance(value, (int, float))
    return isinstance(value, str)


def validate(data: Any, fields: Sequence[FieldSpec]) -> list[str]:
    """Return a list of human-readable problems. Empty list means acceptable.

    The messages are written to be fed straight back to the model on a repair
    attempt, so they name the field and say what was wrong with it.
    """
    problems: list[str] = []

    if not isinstance(data, dict):
        return [f"response was {type(data).__name__}, expected an object"]

    for field in fields:
        if field.name not in data:
            # Only a problem if the field was actually required. An optional
            # field may be absent as well as null.
            if field.required:
                problems.append(f"{field.name}: missing from the response")
            continue

        value = data[field.name]

        if _is_nullish(value):
            if field.required:
                shown = repr(value) if value is not None else "null"
                problems.append(
                    f"{field.name}: required, but the model returned {shown} "
                    f"— extract the real value or report it as absent"
                )
            continue

        if not _type_ok(value, field.kind):
            problems.append(
                f"{field.name}: expected {field.kind}, got {type(value).__name__} ({value!r})"
            )
            continue

        if field.kind in ("int", "float"):
            if field.minimum is not None and value < field.minimum:
                problems.append(f"{field.name}: {value} is below the minimum {field.minimum}")
            if field.maximum is not None and value > field.maximum:
                problems.append(f"{field.name}: {value} is above the maximum {field.maximum}")

        if field.kind == "str" and not value.strip():
            if field.required:
                problems.append(f"{field.name}: required but blank")

    return problems
