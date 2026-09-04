"""Contracts for the guard that keeps private state out of the Git index.

The guard reads Git's index, not the working tree, because the index is what a commit
would capture. Everything it judges is a fact about a path or about blob bytes, so the
decision logic is a pure function and only the reading is impure.
"""

from __future__ import annotations

import runpy
import subprocess
import sys
from pathlib import Path

import pytest

MARKER = "PRIVATE HANDOFF" + " - DO NOT COMMIT"


def make_repo(tmp_path: Path, files: dict[str, str]) -> Path:
    """Create a real Git repository whose index holds exactly the given files."""

    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(
        ["git", "init", "--initial-branch=main", "."], cwd=repo, check=True, capture_output=True
    )
    for relative, content in files.items():
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        subprocess.run(["git", "add", "--", relative], cwd=repo, check=True, capture_output=True)
    return repo


def test_a_tracked_private_handoff_marker_is_a_violation(tmp_path: Path) -> None:
    """The single fact this guard exists for: private state reaching the index."""

    from bevcalib.private_guard import check_repository

    repo = make_repo(tmp_path, {"notes.md": f"{MARKER}\nscene tokens\n"})

    violations = check_repository(repo)

    assert violations == ("notes.md: contains private handoff marker",)


def test_a_tracked_handoff_document_is_a_violation_wherever_it_sits(tmp_path: Path) -> None:
    """A handoff can be renamed or emptied; its location alone is disqualifying."""

    from bevcalib.private_guard import check_repository

    repo = make_repo(tmp_path, {"docs/handoff/status.md": "nothing secret here\n"})

    violations = check_repository(repo)

    assert violations == ("docs/handoff/status.md: forbidden handoff file",)


def test_a_tracked_dataset_or_model_artifact_is_a_violation(tmp_path: Path) -> None:
    """nuScenes data and model weights are licensed or large; neither belongs in Git."""

    from bevcalib.private_guard import check_repository

    repo = make_repo(tmp_path, {"corrector.pt": "weights", "scene.zip": "archive"})

    violations = check_repository(repo)

    assert violations == (
        "corrector.pt: forbidden data or model artifact",
        "scene.zip: forbidden data or model artifact",
    )


def test_a_clean_index_reports_nothing(tmp_path: Path) -> None:
    """A guard that cannot be satisfied would be routed around within a day."""

    from bevcalib.private_guard import check_repository

    repo = make_repo(tmp_path, {"README.md": "# lab\n", "src/bevcalib/__init__.py": ""})

    assert check_repository(repo) == ()


def test_a_tracked_environment_file_is_a_violation_but_its_example_is_not(tmp_path: Path) -> None:
    """`.env` holds real secrets; `.env.example` exists precisely to be committed."""

    from bevcalib.private_guard import check_repository

    repo = make_repo(tmp_path, {".env": "NUSCENES_TOKEN=real\n", ".env.example": "TOKEN=\n"})

    assert check_repository(repo) == (".env: forbidden environment file",)


def test_a_tracked_credential_is_a_violation_whatever_the_file_is_called(tmp_path: Path) -> None:
    """A key pasted into a notebook or a config is the same leak as one in `.env`."""

    from bevcalib.private_guard import check_repository

    repo = make_repo(tmp_path, {"notebook.py": 'token = "hf_' + "a" * 34 + '"\n'})

    assert check_repository(repo) == ("notebook.py: possible credential",)


def test_every_violation_is_reported_once_in_a_stable_order(tmp_path: Path) -> None:
    """Fixing one leak per run, in an order that shifts, is how the second one ships."""

    from bevcalib.private_guard import check_repository

    repo = make_repo(
        tmp_path,
        {
            "weights.pth": "binary-ish",
            "handoff/plan.md": f"{MARKER}\n",
            ".env": "SECRET=1\n",
        },
    )

    assert check_repository(repo) == (
        ".env: forbidden environment file",
        "handoff/plan.md: contains private handoff marker",
        "handoff/plan.md: forbidden handoff file",
        "weights.pth: forbidden data or model artifact",
    )


def test_the_guard_judges_the_index_and_not_the_working_tree(tmp_path: Path) -> None:
    """A commit captures the index, so that is the only thing worth judging."""

    from bevcalib.private_guard import check_repository

    repo = make_repo(tmp_path, {"notes.md": "clean when staged\n"})
    (repo / "notes.md").write_text(f"{MARKER}\n", encoding="utf-8")

    assert check_repository(repo) == ()


def test_a_binary_blob_is_skipped_as_text_but_still_judged_by_its_path(tmp_path: Path) -> None:
    """Weights are not decodable text; the suffix rule has to stand on its own."""

    from bevcalib.private_guard import check_repository

    repo = make_repo(tmp_path, {"README.md": "# lab\n"})
    (repo / "corrector.pt").write_bytes(b"\xff\xfe\x00binary")
    subprocess.run(["git", "add", "--", "corrector.pt"], cwd=repo, check=True, capture_output=True)

    assert check_repository(repo) == ("corrector.pt: forbidden data or model artifact",)


def test_an_unreadable_git_index_fails_closed(tmp_path: Path) -> None:
    """Reporting `no violations` because Git broke would be the worst possible answer."""

    from bevcalib.private_guard import GitIndexError, check_repository

    with pytest.raises(GitIndexError):
        check_repository(tmp_path / "not-a-repository")


def test_the_command_line_reports_each_violation_and_exits_one(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """CI reads the exit code; a human reads the lines."""

    from bevcalib.private_guard import main

    repo = make_repo(tmp_path, {"handoff/plan.md": f"{MARKER}\n"})

    exit_code = main([str(repo)])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "handoff/plan.md: contains private handoff marker" in captured.out
    assert "handoff/plan.md: forbidden handoff file" in captured.out


def test_the_command_line_exits_zero_for_a_clean_index(tmp_path: Path) -> None:
    """The guard runs on every verification, so a clean repository must pass silently."""

    from bevcalib.private_guard import main

    assert main([str(make_repo(tmp_path, {"README.md": "# lab\n"}))]) == 0


def test_the_command_line_exits_two_when_the_index_cannot_be_read(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Two separates `this repository is dirty` from `I could not tell`."""

    from bevcalib.private_guard import main

    exit_code = main([str(tmp_path / "not-a-repository")])

    assert exit_code == 2
    assert "unable to read Git index" in capsys.readouterr().err


def test_the_command_line_defaults_to_the_current_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`python -m bevcalib.private_guard` with no argument is how the verifier calls it."""

    from bevcalib.private_guard import main

    repo = make_repo(tmp_path, {"README.md": "# lab\n"})
    monkeypatch.chdir(repo)

    assert main([]) == 0


def test_the_module_entry_point_used_by_the_verifier_exits_cleanly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The verifier calls `python -m bevcalib.private_guard`, not the function."""

    import bevcalib.private_guard as guard

    repo = make_repo(tmp_path, {"README.md": "# lab\n"})
    assert guard.__file__ is not None
    monkeypatch.setattr(sys, "argv", ["private_guard.py", str(repo)])

    with pytest.raises(SystemExit) as raised:
        runpy.run_path(guard.__file__, run_name="__main__")

    assert raised.value.code == 0


def test_a_dot_slash_prefixed_path_is_judged_under_its_plain_name() -> None:
    """Git can hand back `./name`; two spellings of one path would report twice."""

    from bevcalib.private_guard import find_forbidden_tracked_files

    violations = find_forbidden_tracked_files(("./weights.pt",), {"./weights.pt": ""})

    assert violations == ("weights.pt: forbidden data or model artifact",)


def test_a_directory_that_is_not_a_repository_fails_closed(tmp_path: Path) -> None:
    """The directory exists, so only Git can tell us this is not a repository."""

    from bevcalib.private_guard import GitIndexError, check_repository

    with pytest.raises(GitIndexError):
        check_repository(tmp_path)


def test_a_tracked_path_that_is_not_valid_utf8_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An undecodable path cannot be judged, and unjudged is not the same as clean."""

    import bevcalib.private_guard as guard

    repo = make_repo(tmp_path, {"README.md": "# lab\n"})
    monkeypatch.setattr(guard, "_run_git", lambda root, arguments: b"\xff\xfe.md\x00")

    with pytest.raises(guard.GitIndexError, match=r"^tracked path is not valid UTF-8$"):
        guard.check_repository(repo)
