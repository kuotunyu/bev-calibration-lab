"""The committed derived analysis is exactly what its producer makes from the evidence.

`docs/analysis/operating_envelope_v1/` is a derived publication with its own source
identity. Rebuilding it from the frozen evidence must give the same bytes, so the
README and the website can cite it without anyone retyping a number.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = REPO_ROOT / "docs" / "evidence" / "nuscenes_calibration_v1"
COMMITTED = REPO_ROOT / "docs" / "analysis" / "operating_envelope_v1"


def test_committed_operating_envelope_is_reproduced_byte_for_byte(tmp_path: Path) -> None:
    from bevcalib.analysis.operating_envelope import build_operating_envelope

    data, figure = build_operating_envelope(
        EVIDENCE, tmp_path / "rebuilt", repository_root=REPO_ROOT
    )

    assert sorted(path.name for path in COMMITTED.iterdir()) == [data.name, figure.name]
    assert (COMMITTED / data.name).read_bytes() == data.read_bytes()
    assert (COMMITTED / figure.name).read_bytes() == figure.read_bytes()
    assert json.loads(data.read_bytes())["evidence_type"] == "derived"
    assert len(figure.read_bytes()) < 100_000
