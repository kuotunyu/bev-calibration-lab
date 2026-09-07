"""Run the real Torch backend with explicitly identified local weights."""

import os
from pathlib import Path
from typing import Annotated

import typer

from bevcalib.cli.runtime import checked, dataroot, runtime


def train(
    context: typer.Context,
    config: Annotated[Path, typer.Option()],
    development_manifest: Annotated[Path, typer.Option()],
    calibration_manifest: Annotated[Path, typer.Option()],
    output_dir: Annotated[Path, typer.Option()],
    seed: Annotated[int, typer.Option()],
) -> None:
    from bevcalib.training.engine import train_learned_corrector
    from bevcalib.training.torch_backend import PretrainedWeights, TorchBackend

    options = runtime(context)
    with checked():
        root = dataroot()
        weights = None
        if not options.synthetic_fixture:
            names = (
                "BEVCALIB_PRETRAINED_WEIGHTS",
                "BEVCALIB_PRETRAINED_SOURCE",
                "BEVCALIB_PRETRAINED_SHA256",
            )
            values = [os.environ.get(name) for name in names]
            if not all(values):
                raise ValueError("formal training requires " + ", ".join(names))
            weights = PretrainedWeights(Path(str(values[0])), str(values[1]), str(values[2]))
        backend = TorchBackend(
            root,
            weights=weights,
            synthetic_fixture=options.synthetic_fixture,
            model_factory=options.model_factory,
            device=os.environ.get("BEVCALIB_DEVICE", "cpu"),
        )
        result = train_learned_corrector(
            config,
            development_manifest,
            calibration_manifest,
            output_dir,
            seed,
            backend=backend,
            synthetic_fixture=options.synthetic_fixture,
        )
    typer.echo(str(result.checkpoint_path))
