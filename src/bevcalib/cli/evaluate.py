"""Evaluate native measurements; data roots stay in the invoking shell."""

from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer

from bevcalib.cli.runtime import checked, dataroot, runtime


class Method(StrEnum):
    identity = "identity"
    classical = "classical"
    learned = "learned"


def evaluate(
    context: typer.Context,
    protocol: Annotated[Path, typer.Option()],
    manifest: Annotated[Path, typer.Option()],
    method: Annotated[Method, typer.Option()],
    output_dir: Annotated[Path, typer.Option()],
    checkpoint: Annotated[Path | None, typer.Option()] = None,
) -> None:
    from bevcalib.evaluation import evaluate_calibration

    options = runtime(context)
    with checked():
        result = evaluate_calibration(
            protocol,
            manifest,
            method.value,
            output_dir,
            dataroot=dataroot(),
            checkpoint=checkpoint,
            synthetic_fixture=options.synthetic_fixture,
            model_factory=options.model_factory,
        )
    typer.echo(str(result.directory))
