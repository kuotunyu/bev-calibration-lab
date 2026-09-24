"""The committed explorer page is exactly what the explorer builder makes.

The READMEs link the Pages copy, which the site build regenerates, but the
repository also keeps `docs/demo/calibration-explorer.html`. A template change
that is not followed by regenerating that file would leave the repository copy
stale without failing any other check.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
COMMITTED = REPO_ROOT / "docs" / "demo" / "calibration-explorer.html"


def test_committed_explorer_matches_its_builder_byte_for_byte() -> None:
    from bevcalib.report.explorer import build_explorer

    assert build_explorer().encode("utf-8") == COMMITTED.read_bytes()
