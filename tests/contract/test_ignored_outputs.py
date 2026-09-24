"""Outputs of the documented build commands must never be one `git add` away from a commit."""

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).parents[2]


def test_site_mutation_and_distribution_outputs_are_ignored() -> None:
    # site/ is the Pages build, mutants/ the mutmut sandbox, and dist-a/, dist-b/,
    # dist-from-sdist/ and .ci/ the CI reproducibility builds.
    outputs = [
        "site/index.html",
        "mutants/src/bevcalib/dev.py",
        "dist-a/SHA256SUMS",
        "dist-b/SHA256SUMS",
        "dist-from-sdist/bev_calibration_lab-1.0.0-py3-none-any.whl",
        ".ci/sdist-source/bev_calibration_lab-1.0.0/pyproject.toml",
    ]

    result = subprocess.run(
        ["git", "check-ignore", "--no-index", "--", *outputs],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    assert result.stdout.splitlines() == outputs, result.stderr
