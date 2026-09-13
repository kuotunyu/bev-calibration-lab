"""Root command-line application."""

from __future__ import annotations

import typer

from bevcalib.cli import cohort, data
from bevcalib.cli.evaluate import evaluate
from bevcalib.cli.report import audit_claims, generate_claims, report
from bevcalib.cli.train import train
from bevcalib.cli.validate_study import validate_study

app = typer.Typer(
    name="bev-calib",
    help="Calibration fault sensitivity and recovery for LiDAR-camera BEV perception.",
    no_args_is_help=True,
)


@app.callback()
def main() -> None:
    """Group the calibration lab commands under one entry point."""


app.add_typer(data.app, name="data")
app.add_typer(cohort.app, name="cohort")
app.command()(evaluate)
app.command()(train)
app.command()(report)
app.command("audit-claims")(audit_claims)
app.command("generate-claims")(generate_claims)
app.command("validate-study")(validate_study)
