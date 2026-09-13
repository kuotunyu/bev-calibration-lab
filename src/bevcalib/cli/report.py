"""Portable safe-artifact report and public claim audit commands."""

from functools import partial
from pathlib import Path
from typing import Annotated

import typer

from bevcalib.cli.runtime import checked, runtime


def report(
    context: typer.Context,
    claims: Annotated[Path, typer.Option()],
    artifacts_dir: Annotated[Path, typer.Option()],
    output_dir: Annotated[Path, typer.Option()],
    formal: Annotated[
        bool, typer.Option("--formal", help="Use the validated formal five-document report.")
    ] = False,
    figures: Annotated[
        bool,
        typer.Option("--figures", help="Include audited recovery and BEV SVGs; requires --formal."),
    ] = False,
) -> None:
    from bevcalib.report.builder import build_report
    from bevcalib.report.formal import build_formal_report

    with checked():
        if figures and not formal:
            raise ValueError("figures require --formal")
        builder = build_formal_report if formal else build_report
        if figures:
            builder = partial(build_formal_report, include_figures=True)
        result = builder(
            claims,
            artifacts_dir,
            output_dir,
            repository_root=runtime(context).repository_root or Path.cwd(),
        )
    typer.echo(str(result))


def audit_claims(context: typer.Context, claims: Annotated[Path, typer.Option()]) -> None:
    from bevcalib.analysis.claims import audit_claims as audit

    with checked():
        violations = audit(claims, runtime(context).repository_root or Path.cwd())
        if violations:
            raise ValueError("\n".join(violations))
    typer.echo("All claims resolve to their declared evidence.")


def generate_claims(
    context: typer.Context,
    artifacts_dir: Annotated[Path, typer.Option()],
    output: Annotated[Path, typer.Option()],
) -> None:
    from bevcalib.analysis.formal_claims import generate_formal_claims

    with checked():
        result = generate_formal_claims(
            artifacts_dir, output, repository_root=runtime(context).repository_root or Path.cwd()
        )
    typer.echo(str(result))
