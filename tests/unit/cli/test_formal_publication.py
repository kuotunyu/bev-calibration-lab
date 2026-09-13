"""CLI formal mode is explicit and preserves legacy report dispatch."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from bevcalib.cli.app import app
from bevcalib.cli.runtime import Runtime


def test_formal_report_figures_option_dispatches_to_same_builder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import bevcalib.report.formal as publication

    calls = []

    def builder(*args, **kwargs):  # type: ignore[no-untyped-def]
        calls.append(kwargs)
        return tmp_path / "index.html"

    monkeypatch.setattr(publication, "build_formal_report", builder)
    result = CliRunner().invoke(
        app,
        [
            "report",
            "--formal",
            "--figures",
            "--claims",
            "claims.yaml",
            "--artifacts-dir",
            "evidence",
            "--output-dir",
            "site",
        ],
        obj=Runtime(repository_root=tmp_path),
    )
    assert result.exit_code == 0, result.output
    assert calls == [{"repository_root": tmp_path, "include_figures": True}]


def test_figures_option_refuses_legacy_mode(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app,
        [
            "report",
            "--figures",
            "--claims",
            "claims.yaml",
            "--artifacts-dir",
            "evidence",
            "--output-dir",
            "site",
        ],
        obj=Runtime(repository_root=tmp_path),
    )
    assert result.exit_code != 0
    assert "figures require --formal" in result.output
    assert not (tmp_path / "site").exists()


@pytest.mark.parametrize("formal", [False, True])
def test_report_selects_explicit_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, formal: bool
) -> None:
    import bevcalib.report.builder as legacy
    import bevcalib.report.formal as publication

    calls = []

    def legacy_report(*args, **kwargs):  # type: ignore[no-untyped-def]
        calls.append(("legacy", args, kwargs))
        return tmp_path / "legacy.html"

    def formal_report(*args, **kwargs):  # type: ignore[no-untyped-def]
        calls.append(("formal", args, kwargs))
        return tmp_path / "formal.html"

    monkeypatch.setattr(legacy, "build_report", legacy_report)
    monkeypatch.setattr(publication, "build_formal_report", formal_report)
    command = [
        "report",
        "--claims",
        "claims.yaml",
        "--artifacts-dir",
        "evidence",
        "--output-dir",
        "site",
    ]
    if formal:
        command.append("--formal")
    result = CliRunner().invoke(app, command, obj=Runtime(repository_root=tmp_path))
    assert result.exit_code == 0, result.output
    assert calls == [
        (
            "formal" if formal else "legacy",
            (Path("claims.yaml"), Path("evidence"), Path("site")),
            {"repository_root": tmp_path},
        )
    ]


def test_generate_claims_dispatches_with_repository_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import bevcalib.analysis.formal_claims as publication

    calls = []

    def generate(*args, **kwargs):  # type: ignore[no-untyped-def]
        calls.append((args, kwargs))
        return tmp_path / "claims.yaml"

    monkeypatch.setattr(publication, "generate_formal_claims", generate)
    result = CliRunner().invoke(
        app,
        ["generate-claims", "--artifacts-dir", "evidence", "--output", "claims.yaml"],
        obj=Runtime(repository_root=tmp_path),
    )
    assert result.exit_code == 0, result.output
    assert calls == [((Path("evidence"), Path("claims.yaml")), {"repository_root": tmp_path})]
