"""Verified fixed observations and the declared learned-input geometry."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import numpy.typing as npt
from PIL import Image

from bevcalib.artifacts.results import CalibrationFaultModel
from bevcalib.cohort.records import SceneRecord
from bevcalib.correctors.learned import build_five_channel_input
from bevcalib.geometry.projection import project_camera
from bevcalib.geometry.se3 import transform_points
from bevcalib.nuscenes_adapter.frames import SensorPacket, lidar_to_camera_chain
from bevcalib.nuscenes_adapter.installation import NuScenesInstallation
from bevcalib.nuscenes_adapter.samples import read_lidar_points
from bevcalib.operators.rasterize import rasterize_min_depth
from bevcalib.perturbations.apply import apply_metadata_fault

PREPROCESSING_ID = "bev-input/v1:pil-bilinear-half-pixel:imagenet-rgb:log1p80-depth:valid-mask"


@dataclass(frozen=True)
class Observation:
    camera: SensorPacket
    lidar: SensorPacket
    rgb: npt.NDArray[np.uint8]
    points: npt.NDArray[np.float64]
    intrinsic: npt.NDArray[np.float64]
    boxes: tuple[dict[str, Any], ...]


def load_observation(
    installation: NuScenesInstallation,
    scene: SceneRecord,
    index: int,
    *,
    lidar_token: str | None = None,
) -> Observation:
    """Recheck frozen tokens/timestamps against tables before reading either payload."""
    camera_token = scene.camera_sample_data_tokens[index]
    nominal_lidar = scene.lidar_sample_data_tokens[index]
    expected = (
        (camera_token, "CAM_FRONT", scene.camera_timestamps[index]),
        (nominal_lidar, "LIDAR_TOP", scene.lidar_timestamps[index]),
    )
    for token, channel, timestamp in expected:
        packet = installation.packet(token, channel)
        if (
            packet.timestamp_us != timestamp
            or packet.sample_token != scene.sample_tokens[index]
            or installation.scene_log(token) != (scene.scene_token, scene.log_token)
        ):
            raise ValueError("sensor metadata differs from frozen manifest")
    camera = installation.packet(camera_token, "CAM_FRONT")
    lidar = installation.packet(lidar_token or nominal_lidar, "LIDAR_TOP")
    if installation.scene_log(lidar.sample_data_token) != (scene.scene_token, scene.log_token):
        raise ValueError("selected LiDAR differs from manifest scene/log")
    data = installation.lookup("sample_data", camera_token)
    with Image.open(installation.payload_path(camera_token)) as image:
        if image.size != (data["width"], data["height"]):
            raise ValueError("camera image dimensions disagree with metadata")
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    calibrated = installation.lookup("calibrated_sensor", data["calibrated_sensor_token"])
    return Observation(
        camera,
        lidar,
        rgb,
        read_lidar_points(installation.payload_path(lidar.sample_data_token)),
        np.asarray(calibrated["camera_intrinsic"], dtype=np.float64),
        tuple(
            box
            for box in installation.tables["sample_annotation"].values()
            if box["sample_token"] == camera.sample_token
        ),
    )


def assumed_camera(observation: Observation, fault: CalibrationFaultModel) -> SensorPacket:
    return replace(
        observation.camera,
        calibrated_sensor=replace(
            observation.camera.calibrated_sensor,
            value=apply_metadata_fault(observation.camera.calibrated_sensor.value, fault),
        ),
    )


def prepare_input(
    observation: Observation, fault: CalibrationFaultModel, height: int, width: int
) -> npt.NDArray[np.float32]:
    """Resize with half-pixel geometry, normalize RGB and rasterize assumed depth."""
    camera = assumed_camera(observation, fault)
    old_height, old_width = observation.rgb.shape[:2]
    sx, sy = width / old_width, height / old_height
    intrinsic = (
        np.array([[sx, 0, (sx - 1) / 2], [0, sy, (sy - 1) / 2], [0, 0, 1]]) @ observation.intrinsic
    )
    resized = (
        np.asarray(
            Image.fromarray(observation.rgb).resize((width, height), Image.Resampling.BILINEAR),
            dtype=np.float32,
        )
        / 255
    )
    rgb = (
        ((resized - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225]))
        .transpose(2, 0, 1)
        .astype(np.float32)
    )
    points = transform_points(
        lidar_to_camera_chain(observation.lidar, camera).value, observation.points[:, :3]
    )
    projected = project_camera(points, intrinsic, (width, height))
    depth, valid = rasterize_min_depth(
        projected.uv, projected.optical_depth, projected.valid, (width, height)
    )
    return build_five_channel_input(rgb, depth, valid)
