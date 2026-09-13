"""Explicit-input validation; success JSON appears only after all checks finish."""

import json
from pathlib import Path
from typing import Annotated

import typer

from bevcalib.artifacts.fault_study import validate_fault_study
from bevcalib.cli.runtime import checked


def validate_study(
    expectations: Annotated[
        Path, typer.Option(help="Independently frozen study and validator identity JSON.")
    ],
    artifacts_dir: Annotated[Path, typer.Option(help="Complete formal five-document directory.")],
    claims: Annotated[Path, typer.Option(help="Exact scalar registry for all publication fields.")],
    raw_runs_dir: Annotated[
        Path, typer.Option(help="Evaluation manifest and all five completed raw runs.")
    ],
    checkpoint_17: Annotated[Path, typer.Option(help="Selected seed 17 checkpoint bytes.")],
    checkpoint_42: Annotated[Path, typer.Option(help="Selected seed 42 checkpoint bytes.")],
    checkpoint_73: Annotated[Path, typer.Option(help="Selected seed 73 checkpoint bytes.")],
    repository_root: Annotated[
        Path, typer.Option(help="Boundary for formal evidence and claims paths.")
    ] = Path("."),
) -> None:
    """Validate input provenance and emit a receipt; does not run or tune an experiment."""
    with checked():
        receipt = validate_fault_study(
            expectations,
            artifacts_dir,
            claims,
            raw_runs_dir,
            {"learned-17": checkpoint_17, "learned-42": checkpoint_42, "learned-73": checkpoint_73},
            repository_root=repository_root,
        )
        typer.echo(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False))
