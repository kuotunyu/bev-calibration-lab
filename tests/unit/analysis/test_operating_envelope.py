"""The derived operating envelope counts released intervals without re-estimating them."""

from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
EVIDENCE = REPO_ROOT / "docs" / "evidence" / "nuscenes_calibration_v1"
PATHS = {
    name: f"docs/evidence/nuscenes_calibration_v1/{name}.json"
    for name in ("metrics", "intervals", "recovery")
}


@pytest.fixture(scope="module")
def released():  # type: ignore[no-untyped-def]
    from bevcalib.analysis.formal_claims import load_publication

    return load_publication(EVIDENCE, REPO_ROOT)


@pytest.fixture(scope="module")
def envelope(released) -> dict[str, Any]:  # type: ignore[no-untyped-def]
    from bevcalib.analysis.operating_envelope import analyse

    return analyse(released, PATHS)


def test_the_document_is_derived_and_names_its_exact_sources(envelope, released) -> None:  # type: ignore[no-untyped-def]
    from bevcalib.artifacts.result_documents import digest

    assert envelope["schema_version"] == "bev-calibration-operating-envelope/v1"
    assert envelope["evidence_type"] == "derived"
    assert envelope["source"]["evidence_type"] == "observed"
    assert envelope["source"]["protocol_hash"] == released.metrics.identity.protocol_hash
    for name, path in PATHS.items():
        assert envelope["source"]["documents"][name] == {
            "path": path,
            "document_sha256": getattr(released, name).document_sha256,
        }
    body = {key: value for key, value in envelope.items() if key != "document_sha256"}
    assert envelope["document_sha256"] == digest(body)


def test_learned_mean_beats_classical_everywhere_except_on_the_edge_proxy(envelope) -> None:  # type: ignore[no-untyped-def]
    counts = envelope["interval_counts"]["classical->learned-fixed-three-seed-mean"]
    for metric in (
        "rotation_geodesic_deg",
        "translation_norm_cm",
        "pixel_frame_p50_px",
        "pixel_frame_p90_px",
        "recovery_rate_pct",
    ):
        assert counts[metric]["after_better"] == counts[metric]["total"] == 60
    assert counts["edge_score_px"]["before_better"] == 60


def test_correcting_beats_leaving_alone_only_for_large_faults(envelope) -> None:  # type: ignore[no-untyped-def]
    learned = envelope["interval_counts"]["identity->learned-fixed-three-seed-mean"]
    pixel = learned["pixel_frame_p50_px"]
    assert (pixel["after_better"], pixel["before_better"], pixel["inconclusive"]) == (14, 40, 6)
    assert not any(key.startswith("z:") for key in pixel["after_better_conditions"])
    assert "recovery_rate_pct" not in learned
    classical = envelope["interval_counts"]["identity->classical"]["pixel_frame_p50_px"]
    assert (classical["after_better"], classical["before_better"]) == (4, 53)
    assert {
        axis: value["magnitude"]
        for axis, value in envelope["break_even"]["identity->learned-fixed-three-seed-mean"].items()
    } == {"roll": 1.0, "pitch": 1.0, "yaw": 2.0, "x": 0.2, "y": 0.2, "z": None}
    assert envelope["break_even"]["identity->classical"]["roll"]["magnitude"] == 2.0
    assert envelope["break_even"]["identity->classical"]["pitch"]["magnitude"] is None


def test_residual_floor_pan_residual_and_boundary_are_copied_from_their_sources(
    envelope, released
) -> None:  # type: ignore[no-untyped-def]
    runs = released.metrics.runs
    floor = envelope["residual"]["learned-17"]["rotation_geodesic_deg"]
    assert floor["min"] == runs["learned-17"][floor["min_condition"]]["rotation_geodesic_deg"].value
    assert floor["max"] == runs["learned-17"][floor["max_condition"]]["rotation_geodesic_deg"].value
    assert 0.6 < floor["min"] < floor["max"] < 0.9
    pan = envelope["pan_residual"]["learned-42"]
    assert pan["pitch:2"] == runs["learned-42"]["pitch:2"]["rotation_abs_pitch_deg"].value
    assert sorted(pan) == sorted(key for key in runs["learned-42"] if key.startswith("pitch:"))
    boundary = envelope["recovery_boundary"]
    assert boundary["conditions"] == [
        "roll:-0.25",
        "roll:0.25",
        "pitch:-0.25",
        "pitch:0.25",
        "yaw:-0.25",
        "yaw:0.25",
    ]
    assert set(boundary["identity_recovery_rate_pct"].values()) == {0.0}
    assert all(value > 0.25 for value in boundary["identity_rotation_geodesic_deg"].values())
    assert (boundary["identity_comparisons"], boundary["identity_comparisons_above_zero"]) == (
        30,
        24,
    )
    series = envelope["series"]
    assert len(series["verdict"]) == 60
    assert series["pixel_frame_p50_px"]["identity"]["roll:1"] == (
        runs["identity"]["roll:1"]["pixel_frame_p50_px"].value
    )


def test_evidence_other_than_the_declared_documents_is_refused(released) -> None:  # type: ignore[no-untyped-def]
    from bevcalib.analysis.operating_envelope import V1_SOURCE_SHA256, analyse

    with pytest.raises(ValueError, match="differ from the declared evidence"):
        analyse(released, PATHS, expected_sha256=V1_SOURCE_SHA256 | {"metrics": "0" * 64})


def test_unavailable_intervals_and_missing_values_stay_explicit() -> None:
    from types import SimpleNamespace

    from bevcalib.analysis.operating_envelope import _extreme, _verdict

    assert _verdict(None) == "unavailable"
    assert _verdict(SimpleNamespace(low=-1.0, high=1.0)) == "inconclusive"
    assert _extreme({"a": None}) == {
        "min": None,
        "min_condition": None,
        "max": None,
        "max_condition": None,
    }
    assert _extreme({"a": 2.0, "b": None, "c": 1.0}) == {
        "min": 1.0,
        "min_condition": "c",
        "max": 2.0,
        "max_condition": "a",
    }


def test_break_even_reaches_the_smallest_level_when_every_level_qualifies() -> None:
    from types import SimpleNamespace

    from bevcalib.analysis.operating_envelope import _break_even, _conditions

    better = SimpleNamespace(interval=SimpleNamespace(low=1.0, high=2.0))
    result = _break_even({key: better for _, _, key in _conditions()})

    assert result["roll"] == {"magnitude": 0.1, "unit": "degree"}
    assert result["z"] == {"magnitude": 0.02, "unit": "metre"}


def test_build_writes_new_files_and_refuses_an_existing_directory(tmp_path: Path) -> None:
    from bevcalib.analysis.operating_envelope import build_operating_envelope

    data, figure = build_operating_envelope(EVIDENCE, tmp_path / "out", repository_root=REPO_ROOT)
    assert json.loads(data.read_bytes())["evidence_type"] == "derived"
    assert figure.read_text(encoding="utf-8").startswith("<svg ")
    with pytest.raises(FileExistsError, match="already exists"):
        build_operating_envelope(EVIDENCE, tmp_path / "out", repository_root=REPO_ROOT)


def test_module_entry_point_builds_from_the_command_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import bevcalib.analysis.operating_envelope as module

    output = tmp_path / "cli"
    monkeypatch.chdir(REPO_ROOT)
    monkeypatch.setattr(
        sys,
        "argv",
        ["operating_envelope.py", "--artifacts-dir", str(EVIDENCE), "--output-dir", str(output)],
    )
    with pytest.raises(SystemExit, match="0"):
        runpy.run_path(module.__file__, run_name="__main__")
    printed = capsys.readouterr().out.split()
    assert printed == [
        str(output / "operating-envelope.json"),
        str(output / "operating-envelope.svg"),
    ]
