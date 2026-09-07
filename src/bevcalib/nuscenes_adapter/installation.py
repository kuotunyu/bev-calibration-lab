"""File-backed native nuScenes tables, fixed observations and timing availability."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bevcalib.artifacts.results import CalibrationFaultModel
from bevcalib.cohort.records import SceneRecord, validate_records
from bevcalib.geometry.projection import validate_intrinsic
from bevcalib.nuscenes_adapter.frames import SensorPacket
from bevcalib.nuscenes_adapter.samples import read_sensor_packet
from bevcalib.nuscenes_adapter.sweeps import TimingSelection
from bevcalib.perturbations.schedule import TIMING_OFFSET_MS
from bevcalib.perturbations.timing import select_lidar_for_timing_fault

TABLES = (
    "scene",
    "log",
    "sample",
    "sample_data",
    "calibrated_sensor",
    "ego_pose",
    "sensor",
    "sample_annotation",
)


@dataclass(frozen=True)
class NuScenesInstallation:
    """An explicitly resolved local dataset; no downloads or timestamp relabelling."""

    dataroot: Path
    version: str
    tables: dict[str, dict[str, dict[str, Any]]]

    def lookup(self, table: str, token: str) -> dict[str, Any]:
        try:
            return self.tables[table][token]
        except KeyError as exc:
            raise ValueError(f"unresolved {table} token: {token}") from exc

    def channel(self, token: str) -> str:
        data = self.lookup("sample_data", token)
        calibrated = self.lookup("calibrated_sensor", data["calibrated_sensor_token"])
        return str(self.lookup("sensor", calibrated["sensor_token"])["channel"])

    def scene_log(self, token: str) -> tuple[str, str]:
        data = self.lookup("sample_data", token)
        sample = self.lookup("sample", data["sample_token"])
        scene = self.lookup("scene", sample["scene_token"])
        return str(scene["token"]), str(scene["log_token"])

    def payload_path(self, token: str) -> Path:
        path = (self.dataroot / self.lookup("sample_data", token)["filename"]).resolve()
        if not path.is_relative_to(self.dataroot):
            raise ValueError("sensor payload path escapes dataroot")
        return path

    def packet(self, token: str, expected_channel: str) -> SensorPacket:
        if self.channel(token) != expected_channel or expected_channel not in (
            "CAM_FRONT",
            "LIDAR_TOP",
        ):
            raise ValueError(f"unexpected sensor channel for {token}: expected {expected_channel}")
        camera = expected_channel == "CAM_FRONT"
        return read_sensor_packet(
            self.lookup,
            token,
            ego_frame="camera_ego" if camera else "lidar_ego",
            sensor_frame="camera_sensor" if camera else "lidar_sensor",
        )

    def select_timing(self, camera_token: str, requested_offset_ms: int) -> TimingSelection:
        camera = self.packet(camera_token, "CAM_FRONT")
        scene_log = self.scene_log(camera_token)
        candidates = tuple(
            self.packet(token, "LIDAR_TOP")
            for token in self.tables["sample_data"]
            if self.channel(token) == "LIDAR_TOP"
            and self.scene_log(token) == scene_log
            and self.payload_path(token).is_file()
        )
        return select_lidar_for_timing_fault(
            candidates,
            camera.timestamp_us,
            CalibrationFaultModel(
                rotation_rpy_deg=(0, 0, 0),
                translation_xyz_m=(0, 0, 0),
                requested_time_offset_ms=requested_offset_ms,
            ),
        )

    def scene_records(self) -> tuple[SceneRecord, ...]:
        """Read official split membership and every paired CAM_FRONT/LIDAR_TOP keyframe."""
        from nuscenes.utils.splits import create_splits_scenes

        splits = create_splits_scenes()
        prefix = "mini_" if self.version == "v1.0-mini" else ""
        membership = {name: role for role in ("train", "val") for name in splits[prefix + role]}
        keyframes: dict[str, dict[str, dict[str, Any]]] = {}
        for token, data in self.tables["sample_data"].items():
            channel = self.channel(token)
            if data["is_key_frame"] and channel in ("CAM_FRONT", "LIDAR_TOP"):
                entry = keyframes.setdefault(data["sample_token"], {})
                if channel in entry:
                    raise ValueError("duplicate keyframe sensor channel")
                entry[channel] = data
        records = []
        for token, scene in sorted(self.tables["scene"].items()):
            if scene["name"] not in membership:
                raise ValueError("scene is absent from the official dataset split")
            samples = sorted(
                (
                    sample
                    for sample in self.tables["sample"].values()
                    if sample["scene_token"] == token
                ),
                key=lambda sample: (sample["timestamp"], sample["token"]),
            )
            if not samples or any(
                set(keyframes.get(sample["token"], {})) != {"CAM_FRONT", "LIDAR_TOP"}
                for sample in samples
            ):
                raise ValueError("scene lacks paired keyframe camera/LiDAR metadata")
            pairs = [keyframes[sample["token"]] for sample in samples]
            records.append(
                SceneRecord(
                    scene_token=token,
                    log_token=scene["log_token"],
                    location=self.lookup("log", scene["log_token"])["location"],
                    official_split=membership[scene["name"]],
                    sample_tokens=tuple(sample["token"] for sample in samples),
                    camera_sample_data_tokens=tuple(pair["CAM_FRONT"]["token"] for pair in pairs),
                    lidar_sample_data_tokens=tuple(pair["LIDAR_TOP"]["token"] for pair in pairs),
                    sample_timestamps=tuple(sample["timestamp"] for sample in samples),
                    camera_timestamps=tuple(pair["CAM_FRONT"]["timestamp"] for pair in pairs),
                    lidar_timestamps=tuple(pair["LIDAR_TOP"]["timestamp"] for pair in pairs),
                )
            )
        return validate_records(records)

    def preflight(self) -> dict[str, Any]:
        records = self.scene_records()
        camera_tokens = [token for scene in records for token in scene.camera_sample_data_tokens]
        timing = {}
        for offset in TIMING_OFFSET_MS:
            selections = [self.select_timing(token, offset) for token in camera_tokens]
            valid = sum(s.valid for s in selections)
            timing[str(offset)] = {
                "total": len(selections),
                "valid": valid,
                "invalid": len(selections) - valid,
                "valid_fraction": valid / len(selections) if selections else None,
                "reasons": dict(
                    sorted(Counter(s.reason for s in selections if not s.valid).items())
                ),
            }
        return {
            "dataset_version": self.version,
            "scenes": len(records),
            "samples": len(camera_tokens),
            "missing_payloads": sorted(
                token
                for token in self.tables["sample_data"]
                if not self.payload_path(token).is_file()
            ),
            "timing": timing,
        }


def resolve_installation(dataroot: Path, version: str) -> NuScenesInstallation:
    """Fail closed on unavailable tables and broken sample-data pose directions."""
    if version not in ("v1.0-mini", "v1.0-trainval"):
        raise ValueError(f"unsupported nuScenes version: {version}")
    root = Path(dataroot).resolve()
    if not root.is_dir() or not (root / version).is_dir():
        raise FileNotFoundError(f"nuScenes root or {version} table directory is missing: {root}")
    tables = {}
    for name in TABLES:
        rows = json.loads((root / version / f"{name}.json").read_text(encoding="utf-8"))
        mapped = {row["token"]: row for row in rows}
        if len(mapped) != len(rows):
            raise ValueError(f"duplicate {name} token")
        tables[name] = mapped
    installation = NuScenesInstallation(root, version, tables)
    for token, data in tables["sample_data"].items():
        calibrated = installation.lookup("calibrated_sensor", data["calibrated_sensor_token"])
        ego = installation.lookup("ego_pose", data["ego_pose_token"])
        if ego["timestamp"] != data["timestamp"]:
            raise ValueError("ego pose timestamp does not match actual sensor timestamp")
        installation.payload_path(token)
        installation.scene_log(token)
        channel = installation.channel(token)
        if channel in ("CAM_FRONT", "LIDAR_TOP"):
            installation.packet(token, channel)
        if channel == "CAM_FRONT":
            validate_intrinsic(calibrated["camera_intrinsic"])
    return installation
