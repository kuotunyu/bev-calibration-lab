"""Citation and notice files describe the release and scope each licence.

The code is MIT, but the published results derive from nuScenes and the corrector
was initialized from non-commercial weights. These files are where a reader looks
for that split, so they must stay consistent with the package they describe.
"""

from __future__ import annotations

import datetime as dt
import tomllib
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def test_citation_describes_the_released_package_and_cites_nuscenes() -> None:
    citation = yaml.safe_load((ROOT / "CITATION.cff").read_text(encoding="utf-8"))
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]

    assert citation["cff-version"] == "1.2.0"
    assert citation["title"] == project["name"]
    assert str(citation["version"]) == project["version"]
    assert citation["license"] == project["license"]
    released = citation["date-released"]
    assert isinstance(released, str)
    assert dt.date.fromisoformat(released).isoformat() == released
    assert citation["repository-code"] == "https://github.com/kuotunyu/bev-calibration-lab"
    assert [author["alias"] for author in citation["authors"]] == ["kuotunyu"]
    titles = [reference["title"] for reference in citation["references"]]
    assert titles == ["nuScenes: A Multimodal Dataset for Autonomous Driving"]


def test_notice_scopes_code_data_weights_and_embedded_javascript() -> None:
    notice = (ROOT / "NOTICE").read_text(encoding="utf-8")

    for phrase in (
        "MIT License",
        "nuScenes",
        "CC BY-NC-SA 4.0",
        "convnextv2_tiny.fcmae_ft_in1k",
        "CC BY-NC 4.0",
        "Plotly.js",
        "MapLibre GL JS",
        "BSD 3-Clause",
    ):
        assert phrase in notice, phrase
