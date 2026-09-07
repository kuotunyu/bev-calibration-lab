"""Read-only native installation preflight."""

import json
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer

from bevcalib.cli.runtime import checked


class Version(StrEnum):
    mini = "v1.0-mini"
    trainval = "v1.0-trainval"


app = typer.Typer(no_args_is_help=True)


@app.command()
def preflight(
    dataroot: Annotated[Path, typer.Option()],
    version: Annotated[Version, typer.Option()],
    output: Annotated[Path, typer.Option()],
) -> None:
    from bevcalib.nuscenes_adapter.installation import resolve_installation

    with checked():
        document = resolve_installation(dataroot, version.value).preflight()
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(document, sort_keys=True, indent=2) + "\n")
    typer.echo(str(output))
