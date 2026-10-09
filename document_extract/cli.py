"""Command line entry point.

`demo` is the one to run first: it exercises the repair loop offline with a
scripted model, so you can see exactly what the pipeline does when the model
returns something schema-valid but wrong.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .extractor import Extractor
from .llm import ClaudeLLM, ScriptedLLM
from .schema import FieldSpec

FIXTURE = Path(__file__).resolve().parent.parent / "fixtures" / "sample_tender.txt"

DEMO_FIELDS = [
    FieldSpec(
        "contracting_authority",
        "The public body issuing the contract notice.",
    ),
    FieldSpec(
        "estimated_value_eur",
        "The estimated contract value in euros, as a number without separators.",
        kind="float",
        minimum=1.0,
    ),
    FieldSpec(
        "submission_deadline",
        "The deadline for submitting a tender, as written in the notice.",
    ),
    FieldSpec(
        "is_cancelled",
        "True only if the notice states the procedure is cancelled or annulled.",
        kind="bool",
    ),
]

# The failure this whole repo exists to catch. Every key is present, every type
# is correct, the JSON parses — and two fields are worthless. A naive pipeline
# would write this row and move on.
FIRST_RESPONSE = json.dumps(
    {
        "contracting_authority": "Ministry of Infrastructure and Energy",
        "estimated_value_eur": 0,
        "submission_deadline": "N/A",
        "is_cancelled": False,
    }
)

SECOND_RESPONSE = json.dumps(
    {
        "contracting_authority": "Ministry of Infrastructure and Energy",
        "estimated_value_eur": 1250000.0,
        "submission_deadline": "14 November 2026, 12:00 local time",
        "is_cancelled": False,
    }
)


def _rule(title: str) -> None:
    print()
    print(title)
    print("-" * len(title))


def cmd_demo(_args: argparse.Namespace) -> int:
    document = FIXTURE.read_text(encoding="utf-8")

    _rule("Attempt 1 — schema-valid, semantically empty")
    print("The model returns:")
    print("  " + FIRST_RESPONSE)

    llm = ScriptedLLM([FIRST_RESPONSE, SECOND_RESPONSE])
    extractor = Extractor(llm, DEMO_FIELDS, max_repairs=2)
    result = extractor.extract(document)

    print()
    print("Validation rejects it:")
    for problem in result.attempts[0].problems:
        print(f"  - {problem}")

    _rule("Attempt 2 — the repair prompt carries the specifics")
    print("The retry prompt appends the validation errors:")
    print("  ...")
    repair_prompt = llm.prompts[1]
    tail = repair_prompt.split("---")[-1].strip()
    for line in tail.splitlines()[:8]:
        print("  " + line)

    _rule("Result")
    print(f"  {result.summary()}")
    print()
    print("  parsed record:")
    for key, value in (result.data or {}).items():
        print(f"    {key:24} {value!r}")
    print()
    print(f"  requests sent to the model: {len(llm.prompts)}")
    print("  The first response would have been written to the database as a")
    print("  contract worth EUR 0 with no deadline. Nothing about it would have")
    print("  raised an error.")
    return 0


def cmd_extract(args: argparse.Namespace) -> int:
    document = Path(args.input).read_text(encoding="utf-8")

    fields = [
        FieldSpec("summary", "A one-sentence summary of the document.", kind="str"),
        FieldSpec("parties", "Named organisations or people, comma separated.", kind="str"),
        FieldSpec(
            "total_value",
            "The principal monetary amount stated, as a number.",
            kind="float",
            required=False,
            minimum=0.0,
        ),
    ]

    extractor = Extractor(ClaudeLLM(model=args.model), fields, max_repairs=args.repairs)
    result = extractor.extract(document)

    print(result.summary())
    if result.data:
        print(json.dumps(result.data, indent=2, ensure_ascii=False))
    for attempt in result.attempts:
        if attempt.problems:
            print(f"\nattempt {attempt.index + 1} rejected:", file=sys.stderr)
            for problem in attempt.problems:
                print(f"  - {problem}", file=sys.stderr)

    return 0 if result.ok else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="document-extract",
        description="Extract typed fields from a document, with validation and repair.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="offline demonstration of the repair loop")
    demo.set_defaults(func=cmd_demo)

    extract = sub.add_parser("extract", help="extract from a file (needs an API key)")
    extract.add_argument("input", help="path to a text document")
    extract.add_argument("--model", default="claude-opus-5-5")
    extract.add_argument("--repairs", type=int, default=2)
    extract.set_defaults(func=cmd_extract)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
