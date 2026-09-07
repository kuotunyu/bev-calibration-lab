"""Availability is resolved once; required keyframes fail preflight closed."""

from pathlib import Path

import pytest
from tests.unit.nuscenes_adapter.test_installation import installation_root as installation_root
from tests.unit.nuscenes_adapter.test_installation import rewrite


def test_timing_uses_resolved_snapshot_until_explicit_refresh(installation_root: Path) -> None:
    from bevcalib.nuscenes_adapter.installation import resolve_installation

    installation = resolve_installation(installation_root, "v1.0-mini")
    (installation_root / "samples/LIDAR_TOP/selected.bin").unlink()
    assert installation.select_timing("camera", 100).selected_sample_data_token == "selected"
    assert installation.preflight()["missing_payloads"] == []
    refreshed = resolve_installation(installation_root, "v1.0-mini")
    assert refreshed.select_timing("camera", 100).selected_sample_data_token == "lidar"
    assert refreshed.preflight()["missing_payloads"] == ["selected"]


@pytest.mark.parametrize("payload", ["CAM_FRONT/camera.png", "LIDAR_TOP/lidar.bin"])
def test_preflight_refuses_missing_required_keyframe_payload(
    installation_root: Path, payload: str
) -> None:
    from bevcalib.nuscenes_adapter.installation import resolve_installation

    (installation_root / "samples" / payload).unlink()
    installation = resolve_installation(installation_root, "v1.0-mini")
    with pytest.raises(ValueError, match="required paired keyframe payload"):
        installation.preflight()


def test_repeated_timing_selection_does_not_scan_tables_or_stat_files(
    installation_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from bevcalib.nuscenes_adapter.installation import resolve_installation

    installation = resolve_installation(installation_root, "v1.0-mini")

    def unexpected(*args, **kwargs):
        raise AssertionError("snapshot selection performed a filesystem stat")

    monkeypatch.setattr(Path, "is_file", unexpected)
    for _ in range(3):
        for offset in (-200, -100, -50, 50, 100, 200):
            installation.select_timing("camera", offset)


def test_indexed_selector_preserves_token_ties_and_boundary_targets(
    installation_root: Path,
) -> None:
    from bevcalib.nuscenes_adapter.installation import resolve_installation

    # Equal timestamps retain the smallest token, including when target lies beyond both.
    def duplicate(rows):
        rows.append(rows[2] | {"token": "a-selected"})

    rewrite(installation_root, "sample_data", duplicate)
    installation = resolve_installation(installation_root, "v1.0-mini")
    assert installation.select_timing("camera", 100).selected_sample_data_token == "a-selected"
    assert installation.select_timing("camera", 200).selected_sample_data_token == "a-selected"
    assert installation.select_timing("camera", -200).selected_sample_data_token == "lidar"


def test_preflight_binds_actual_table_bytes_and_selected_timing_evidence(
    installation_root: Path,
) -> None:
    import hashlib

    from bevcalib.nuscenes_adapter.installation import TABLES, resolve_installation

    installation = resolve_installation(installation_root, "v1.0-mini")
    expected = {
        name: hashlib.sha256(
            (installation_root / "v1.0-mini" / (name + ".json")).read_bytes()
        ).hexdigest()
        for name in TABLES
    }
    # The resolved metadata/hash snapshot is retained if the source changes later.
    rewrite(installation_root, "sample_annotation", lambda rows: rows.append({"token": "later"}))
    report = installation.preflight()
    assert report["table_sha256"] == expected
    row = report["timing"]["100"]["selections"][0]
    assert row["camera_token"] == "camera"
    assert row["camera_timestamp_us"] == 1_020_000
    assert row["selected_sample_data_token"] == "selected"
    assert row["selected_timestamp_us"] == 1_115_000
    assert row["realized_offset_ms"] == 95
    assert row["absolute_error_ms"] == 5
    assert row["valid"] is True
