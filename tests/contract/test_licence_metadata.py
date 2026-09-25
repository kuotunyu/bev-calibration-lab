"""Citation and notice files describe the release and scope each licence.

The code is MIT, but the published results derive from nuScenes and the corrector
was initialized from non-commercial weights. These files are where a reader looks
for that split, so they must stay consistent with the package they describe.
"""

from __future__ import annotations

import datetime as dt
import tomllib
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
RESTATED = "the values restated from them elsewhere in this repository"


def _words(text: str) -> str:
    return " ".join(text.split())


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
    notice = _words((ROOT / "NOTICE").read_text(encoding="utf-8"))

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


def test_notice_covers_values_restated_outside_the_result_files() -> None:
    """READMEs, cards, known issues and test fixtures repeat evidence values."""

    notice = _words((ROOT / "NOTICE").read_text(encoding="utf-8"))
    derived = notice.split("nuScenes-derived results", 1)[1].split("Pretrained", 1)[0]

    assert RESTATED in derived
    assert "fixture and example values in src/ and tests/" in derived
    assert "apart from the nuScenes-derived values they restate" in notice


def test_notice_claims_only_the_licence_marks_the_explorer_bundle_keeps() -> None:
    """The bundle keeps MapLibre GL JS's identifier and a link, not a copyright line."""

    notice = _words((ROOT / "NOTICE").read_text(encoding="utf-8"))
    explorer = (ROOT / "docs" / "demo" / "calibration-explorer.html").read_text(encoding="utf-8")

    assert "the embedded code keeps their licence identifiers and links" in notice
    assert "copyright and licence notices are kept" not in notice
    assert "@license 3-Clause BSD. Full text of license: https://github.com/maplibre/" in explorer


@pytest.mark.parametrize(
    ("readme", "heading", "phrase"),
    [
        ("README.en.md", "## Data, model and third-party licences", RESTATED),
        ("README.md", "## 資料、模型與第三方授權", "以及在本 repository 其他地方轉述的這些數值"),
    ],
)
def test_readme_licence_sections_cover_the_restated_values(
    readme: str, heading: str, phrase: str
) -> None:
    text = (ROOT / readme).read_text(encoding="utf-8")
    section = text.split(heading, 1)[1].split("\n## ", 1)[0]

    assert phrase in _words(section)
