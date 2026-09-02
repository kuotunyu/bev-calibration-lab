"""Root command-line application."""

from __future__ import annotations

import typer

app = typer.Typer(
    name="bev-calib",
    help="Calibration fault sensitivity and recovery for LiDAR-camera BEV perception.",
    no_args_is_help=True,
)


@app.callback()
def main() -> None:
    """Group the calibration lab commands under one entry point."""
