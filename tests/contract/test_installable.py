"""Contracts for how this repository is assembled: packaging, entry point, verifier.

These are the promises a person relies on before any calibration code matters. The
console script has to resolve to something callable, the interpreter range has to be
the one the lock was resolved against, and the verifier has to stop at the first failing
gate rather than reporting the last one.
"""

from __future__ import annotations

import tomllib
from collections.abc import Sequence
from importlib import import_module
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = REPO_ROOT / "pyproject.toml"


def pyproject() -> dict[str, Any]:
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))


def test_the_declared_console_entry_point_resolves_to_a_callable() -> None:
    """A console script naming a module path that does not import is a broken install."""

    target = pyproject()["project"]["scripts"]["bev-calib"]
    module_path, _, attribute = target.partition(":")

    resolved = getattr(import_module(module_path), attribute)

    assert callable(resolved)


def test_the_help_screen_runs_through_the_declared_application() -> None:
    """`bev-calib --help` is the first thing anyone runs; it must exit 0."""

    from typer.testing import CliRunner

    from bevcalib.cli.app import app

    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0, result.output
    assert "bev-calib" in result.output or "Usage" in result.output


def test_the_interpreter_range_is_pinned_to_one_minor_version() -> None:
    """The lock is resolved for one interpreter; a widened range would silently unpin it."""

    assert pyproject()["project"]["requires-python"] == ">=3.12,<3.13"


def test_the_nuscenes_adapter_dependency_is_core_rather_than_optional() -> None:
    """This package exists to read nuScenes, so its devkit cannot be an optional extra."""

    project = pyproject()["project"]
    core = {name.split(">")[0].split("=")[0].strip() for name in project["dependencies"]}

    assert "nuscenes-devkit" in core


def test_the_verifier_stops_at_the_first_failing_stage(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reporting a later gate would hide the earliest cause and waste a full run."""

    from bevcalib import dev

    attempted: list[str] = []

    def runner(stage: str, command: Sequence[str], cwd: Path) -> int:
        attempted.append(stage)
        return 3 if stage == "lint" else 0

    exit_code = dev.verify_repository(REPO_ROOT, runner=runner)

    assert exit_code == 3
    assert attempted == list(dev.VERIFY_STAGES[: dev.VERIFY_STAGES.index("lint") + 1])


def test_the_verifier_returns_zero_only_after_every_stage_ran() -> None:
    """A verifier that can pass while skipping a gate is worse than no verifier."""

    from bevcalib import dev

    attempted: list[str] = []

    def runner(stage: str, command: Sequence[str], cwd: Path) -> int:
        attempted.append(stage)
        return 0

    exit_code = dev.verify_repository(REPO_ROOT, runner=runner)

    assert exit_code == 0
    assert attempted == list(dev.VERIFY_STAGES)


def test_malformed_schema_files_are_reported_by_path(tmp_path: Path) -> None:
    """A schema that does not parse cannot gate anything, and it fails quietly."""

    from bevcalib import dev

    (tmp_path / "schemas").mkdir()
    (tmp_path / "schemas" / "good.json").write_text('{"a": 1}', encoding="utf-8")
    (tmp_path / "schemas" / "broken.json").write_text("{not json", encoding="utf-8")

    assert dev.verify_schema_contracts(tmp_path) == 1


def test_well_formed_schema_files_pass(tmp_path: Path) -> None:
    """The stage has to be satisfiable, including when no schema exists yet."""

    from bevcalib import dev

    (tmp_path / "schemas").mkdir()
    (tmp_path / "schemas" / "envelope.json").write_text('{"type": "object"}', encoding="utf-8")

    assert dev.verify_schema_contracts(tmp_path) == 0


def test_a_markdown_link_to_a_missing_local_file_is_reported(tmp_path: Path) -> None:
    """Documentation that points at nothing is how a reader loses trust in all of it."""

    from bevcalib import dev

    (tmp_path / "README.md").write_text("see [the card](docs/dataset-card.md)\n", encoding="utf-8")

    assert dev.verify_docs_links(tmp_path) == 1


def test_external_anchor_and_resolvable_links_pass(tmp_path: Path) -> None:
    """Only local files are checkable here; URLs and anchors must not be flagged."""

    from bevcalib import dev

    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "card.md").write_text("# card\n", encoding="utf-8")
    (tmp_path / "README.md").write_text(
        "[card](docs/card.md) [web](https://example.org) [top](#heading)\n", encoding="utf-8"
    )

    assert dev.verify_docs_links(tmp_path) == 0


def test_the_developer_commands_dispatch_by_name(tmp_path: Path) -> None:
    """These names are typed by hand and wired into CI; each must reach its own code."""

    from bevcalib import dev

    (tmp_path / "schemas").mkdir()
    (tmp_path / "README.md").write_text("# lab\n", encoding="utf-8")
    calls: list[str] = []

    def runner(stage: str, command: Sequence[str], cwd: Path) -> int:
        calls.append(stage)
        return 0

    assert dev.main(["verify", "--repo-root", str(tmp_path)], runner=runner) == 0
    assert calls == list(dev.VERIFY_STAGES)
    assert dev.main(["schema-contracts", "--repo-root", str(tmp_path)]) == 0
    assert dev.main(["docs-links", "--repo-root", str(tmp_path)]) == 0


def test_the_developer_module_entry_point_dispatches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The verify stages shell out to `python -m bevcalib.dev`, so that path must work."""

    import runpy
    import sys

    from bevcalib import dev

    (tmp_path / "schemas").mkdir()
    assert dev.__file__ is not None
    monkeypatch.setattr(sys, "argv", ["dev.py", "schema-contracts", "--repo-root", str(tmp_path)])

    with pytest.raises(SystemExit) as raised:
        runpy.run_path(dev.__file__, run_name="__main__")

    assert raised.value.code == 0


def test_the_default_runner_reports_a_real_process_exit_code(tmp_path: Path) -> None:
    """Everything above injects a runner; the real one is what CI actually executes."""

    import sys

    from bevcalib import dev

    assert (
        dev.subprocess_runner("probe", [sys.executable, "-c", "raise SystemExit(7)"], tmp_path) == 7
    )


def test_a_root_relative_markdown_link_resolves_from_the_repository_root(tmp_path: Path) -> None:
    """`/docs/x.md` means the repository root, not the filesystem root."""

    from bevcalib import dev

    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "card.md").write_text("# card\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("[card](/docs/card.md)\n", encoding="utf-8")

    assert dev.verify_docs_links(tmp_path) == 0


def test_private_evidence_copies_are_outside_public_documentation(tmp_path: Path) -> None:
    from bevcalib import dev

    private_copy = tmp_path / "artifacts" / "preflight" / "historical.md"
    private_copy.parent.mkdir(parents=True)
    private_copy.write_text("[original relative link](../dataset-card.md)", encoding="utf-8")
    (tmp_path / "README.md").write_text("# Public repository", encoding="utf-8")

    assert dev.verify_docs_links(tmp_path) == 0


@pytest.mark.parametrize("relative", ["README.md", "docs/card.md", "docs/artifacts/card.md"])
def test_public_broken_links_remain_checked(tmp_path: Path, relative: str) -> None:
    from bevcalib import dev

    public_file = tmp_path / relative
    public_file.parent.mkdir(parents=True, exist_ok=True)
    public_file.write_text("[missing](missing-card.md)", encoding="utf-8")

    assert dev.verify_docs_links(tmp_path) == 1
