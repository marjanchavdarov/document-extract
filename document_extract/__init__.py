"""document-extract — turn documents into typed records, and prove it worked.

The claim: an LLM extraction pipeline is only as good as its validation. A model
constrained to a schema returns the right *shape* every time and the right
*content* some of the time; the gap between those two is the whole job.
"""

from .extractor import Attempt, ExtractionResult, Extractor, build_prompt, build_repair_prompt
from .llm import ClaudeLLM, LLM, ScriptedLLM
from .schema import FieldSpec, build_json_schema, validate

__version__ = "1.0.0"

__all__ = [
    "Extractor",
    "ExtractionResult",
    "Attempt",
    "build_prompt",
    "build_repair_prompt",
    "FieldSpec",
    "build_json_schema",
    "validate",
    "LLM",
    "ScriptedLLM",
    "ClaudeLLM",
]
