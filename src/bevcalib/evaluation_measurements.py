"""Measured fixed-observation evaluation over a verified cohort and complete fault inventory."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

import numpy as np
import yaml

from bevcalib.artifacts.result_documents import (
    fault_for_condition,
)
from bevcalib.artifacts.results import (
    CalibrationFaultModel,
    CalibrationResultV2,
    GroundContactResult,
    PoseErrors,
    SensorIdentity,
    TimingEvidence,
)
from bevcalib.correctors.classical import coarse_to_fine_correct
from bevcalib.correctors.identity import identity_correction
from bevcalib.geometry.projection import project_camera
from bevcalib.geometry.se3 import compose, inverse, transform_points
from bevcalib.metrics.calibration import calibration_errors
from bevcalib.nuscenes_adapter.frames import SensorPacket, lidar_to_camera_chain
from bevcalib.nuscenes_adapter.sweeps import timing_measurements
from bevcalib.operators.ground_contact import ground_plane_z_from_ego, observe_ground_contact
from bevcalib.operators.image_edges import ImageEdgeEvidence
from bevcalib.operators.lidar_edges import trimmed_distance_field_score
from bevcalib.perturbations.apply import apply_metadata_fault, inverse_fault
from bevcalib.preprocessing import Observation, assumed_camera, prepare_input

ZERO = CalibrationFaultModel(
    rotation_rpy_deg=(0, 0, 0), translation_xyz_m=(0, 0, 0), requested_time_offset_ms=0
)


def sensor_identity(packet: SensorPacket, channel: str) -> SensorIdentity:
    return SensorIdentity(
        token=packet.sample_data_token, timestamp_us=packet.timestamp_us, channel=channel
    )


def corrected_camera(
    observation: Observation, fault: CalibrationFaultModel, correction: CalibrationFaultModel
) -> SensorPacket:
    assumed = assumed_camera(observation, fault)
    return replace(
        assumed,
        calibrated_sensor=replace(
            assumed.calibrated_sensor,
            value=apply_metadata_fault(assumed.calibrated_sensor.value, correction),
        ),
    )


def measure_condition(
    observation: Observation,
    *,
    scene_token: str,
    axis: Any,
    level: float,
    method: str,
    edges: ImageEdgeEvidence,
    edge_points: np.ndarray,
    predictor: Any = None,
) -> CalibrationResultV2:
    fault = fault_for_condition(axis, level)
    height, width = observation.rgb.shape[:2]

    def project(camera: SensorPacket, points: np.ndarray) -> Any:
        return project_camera(
            transform_points(lidar_to_camera_chain(observation.lidar, camera).value, points),
            observation.intrinsic,
            (width, height),
        )

    def objective(values: np.ndarray) -> float:
        correction = CalibrationFaultModel(
            rotation_rpy_deg=tuple(values[:3]),
            translation_xyz_m=tuple(values[3:]),
            requested_time_offset_ms=0,
        )
        projected = project(corrected_camera(observation, fault, correction), edge_points)
        if not projected.valid.any():
            return -math.hypot(width, height)
        return trimmed_distance_field_score(projected.uv[projected.valid], edges.distance_field)

    estimate = identity_correction()
    if method == "classical" and edges.distance_field is not None:
        estimate = coarse_to_fine_correct(objective)
    if method == "learned":
        config = yaml.safe_load(predictor.metadata["config_raw"])
        values = predictor.predict(
            prepare_input(observation, fault, config["input_height"], config["input_width"])
        )
        correction = CalibrationFaultModel(
            rotation_rpy_deg=tuple(values[:3]),
            translation_xyz_m=tuple(values[3:]),
            requested_time_offset_ms=0,
        )
    else:
        correction = CalibrationFaultModel(
            rotation_rpy_deg=estimate.rotation_rpy_deg,
            translation_xyz_m=estimate.translation_xyz_m,
            requested_time_offset_ms=0,
        )
    camera = corrected_camera(observation, fault, correction)
    errors = calibration_errors(correction, ZERO if axis == "time" else inverse_fault(fault))
    pose = PoseErrors(
        rotation_rpy_error_deg=tuple(
            errors[f"{name}_error_deg"] for name in ("roll", "pitch", "yaw")
        ),
        translation_xyz_error_m=tuple(errors[f"{name}_error_m"] for name in ("x", "y", "z")),
        rotation_geodesic_error_deg=errors["rotation_geodesic_error_deg"],
        translation_error_m=errors["translation_error_m"],
    )
    true_projection = project(observation.camera, observation.points[:, :3])
    actual_projection = project(camera, observation.points[:, :3])
    paired = true_projection.valid & actual_projection.valid
    pixel_errors = tuple(
        float(value)
        for value in np.linalg.norm(
            true_projection.uv[paired] - actual_projection.uv[paired], axis=1
        )
    )
    projected_edges = project(camera, edge_points)
    score = (
        None
        if edges.distance_field is None or not projected_edges.valid.any()
        else trimmed_distance_field_score(
            projected_edges.uv[projected_edges.valid], edges.distance_field
        )
    )
    true_from_global = inverse(
        compose(observation.camera.ego_pose.value, observation.camera.calibrated_sensor.value)
    )
    actual_from_global = inverse(compose(camera.ego_pose.value, camera.calibrated_sensor.value))
    contacts = []
    for box in observation.boxes:
        contact = observe_ground_contact(
            box_token=box["token"],
            box_center_global=np.asarray(box["translation"], dtype=np.float64),
            size_wlh=tuple(box["size"]),
            orientation_wxyz=tuple(box["rotation"]),
            true_camera_from_global=true_from_global,
            assumed_camera_from_global=actual_from_global,
            intrinsic=observation.intrinsic,
            image_size_wh=(width, height),
            ground_z_global=ground_plane_z_from_ego(observation.camera.ego_pose),
        )
        error = (
            float(
                np.linalg.norm(np.subtract(contact.assumed_ground_xy_m, contact.oracle_ground_xy_m))
            )
            if contact.valid
            else None
        )
        contacts.append(
            GroundContactResult(
                box_token=contact.box_token,
                range_m=contact.range_m,
                error_m=error,
                invalid_reason=None if contact.valid else "ground_contact_geometry_or_range",
            )
        )
    reasons = []
    if not pixel_errors:
        reasons.append("no_paired_projection")
    if edges.distance_field is None:
        reasons.append("no_image_edges")
    elif score is None:
        reasons.append("no_projected_lidar_edges")
    if not any(contact.error_m is not None for contact in contacts):
        reasons.append("no_valid_ground_contact")
    realized, timing_error = timing_measurements(
        observation.camera.timestamp_us,
        observation.lidar.timestamp_us,
        fault.requested_time_offset_ms,
    )
    return CalibrationResultV2(
        schema_version="bev-calibration-result/v2",
        sample_token=observation.camera.sample_token,
        scene_token=scene_token,
        fault_axis=axis,
        fault_level=level,
        fault=fault,
        estimate=correction,
        camera=sensor_identity(observation.camera, "CAM_FRONT"),
        lidar=sensor_identity(observation.lidar, "LIDAR_TOP"),
        timing=TimingEvidence(
            requested_offset_ms=fault.requested_time_offset_ms,
            realized_offset_ms=realized,
            absolute_error_ms=timing_error,
            reason="valid" if axis == "time" else "nominal_pair",
        ),
        pose=pose,
        pixel_errors_px=pixel_errors,
        projection_count=len(observation.points),
        edge_alignment_score=score,
        ground_contacts=tuple(contacts),
        valid=not reasons,
        invalid_reason=";".join(reasons) if reasons else None,
    )


def invalid_timing_result(
    observation: Observation, scene_token: str, selection: Any, selected_lidar: SensorPacket | None
) -> CalibrationResultV2:
    return CalibrationResultV2(
        schema_version="bev-calibration-result/v2",
        sample_token=observation.camera.sample_token,
        scene_token=scene_token,
        fault_axis="time",
        fault_level=selection.requested_offset_ms,
        fault=fault_for_condition("time", selection.requested_offset_ms),
        estimate=None,
        camera=sensor_identity(observation.camera, "CAM_FRONT"),
        lidar=None if selected_lidar is None else sensor_identity(selected_lidar, "LIDAR_TOP"),
        timing=TimingEvidence(
            requested_offset_ms=selection.requested_offset_ms,
            realized_offset_ms=selection.realized_offset_ms,
            absolute_error_ms=selection.absolute_error_ms,
            reason=selection.reason,
        ),
        pose=None,
        pixel_errors_px=(),
        projection_count=0,
        edge_alignment_score=None,
        ground_contacts=(),
        valid=False,
        invalid_reason=selection.reason,
    )
