#!/usr/bin/env python3
"""Validate one nuScenes transform/projection evidence trace."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pydantic import ValidationError

from bevcalib.verification.transform_trace import TransformVerificationTrace


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        document = json.loads(arguments.trace.read_text(encoding="utf-8"))
    except OSError as exc:
        print(f"cannot read trace: {exc}", file=sys.stderr)
        return 2
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        print(f"trace is not valid JSON: {exc}", file=sys.stderr)
        return 2
    try:
        trace = TransformVerificationTrace.model_validate(document)
    except ValidationError as exc:
        print(f"transform trace is invalid:\n{exc}", file=sys.stderr)
        return 1
    print(f"transform trace is valid with {len(trace.parity.cases)} parity cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
