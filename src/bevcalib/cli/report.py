"""Portable safe-artifact report and public claim audit commands."""

from pathlib import Path
from typing import Annotated

import typer

from bevcalib.cli.runtime import checked, runtime


def report(
    context: typer.Context,
    claims: Annotated[Path, typer.Option()],
    artifacts_dir: Annotated[Path, typer.Option()],
    output_dir: Annotated[Path, typer.Option()],
) -> None:
    from bevcalib.report.builder import build_report

    with checked():
        result = build_report(
            claims, artifacts_dir, output_dir, repository_root=runtime(context).repository_root
        )
    typer.echo(str(result))


def audit_claims(context: typer.Context, claims: Annotated[Path, typer.Option()]) -> None:
    from bevcalib.analysis.claims import audit_claims as audit

    with checked():
        violations = audit(claims, runtime(context).repository_root or Path.cwd())
        if violations:
            raise ValueError("\n".join(violations))
    typer.echo("All claims resolve to their declared evidence.")
