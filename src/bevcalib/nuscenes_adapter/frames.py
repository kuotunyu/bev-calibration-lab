"""One sensor reading, and the chain that carries LiDAR points into the camera."""

from __future__ import annotations

from dataclasses import dataclass

from bevcalib.geometry.frames import FramedTransform, compose_framed, inverse_framed


@dataclass(frozen=True)
class SensorPacket:
    """Everything one `sample_data` record contributes to the transform chain.

    The timestamp is kept in microseconds, the unit nuScenes uses, so that no
    conversion happens implicitly on the way in. Timing faults are requested in
    milliseconds and the two units are never mixed in one number.
    """

    sample_token: str
    sample_data_token: str
    timestamp_us: int
    calibrated_sensor: FramedTransform
    ego_pose: FramedTransform
    file_relative_path: str


def lidar_to_camera_chain(lidar: SensorPacket, camera: SensorPacket) -> FramedTransform:
    """Return `T_camera_sensor_lidar_sensor` for one pair of readings.

    Two ego poses are used, one per packet, because the sweep and the exposure
    happen at different instants and the vehicle moves in between. At 50 km/h
    that gap is about 0.7 m over 50 ms, the same size as the translation faults
    this study injects, so collapsing the two poses would be indistinguishable
    from the effect being measured.

    The camera half of the chain runs backwards: nuScenes stores the sensor pose
    in ego and the ego pose in global, so reaching the camera means inverting
    both of its links.
    """

    if lidar.calibrated_sensor.source != "lidar_sensor":
        raise ValueError(
            "the first packet must be a LiDAR reading, but its sensor frame is "
            f"{lidar.calibrated_sensor.source!r}, not 'lidar_sensor'"
        )
    if camera.calibrated_sensor.source != "camera_sensor":
        raise ValueError(
            "the second packet must be a camera reading, but its sensor frame is "
            f"{camera.calibrated_sensor.source!r}, not 'camera_sensor'"
        )

    global_from_lidar_sensor = compose_framed(lidar.ego_pose, lidar.calibrated_sensor)
    camera_sensor_from_global = compose_framed(
        inverse_framed(camera.calibrated_sensor), inverse_framed(camera.ego_pose)
    )
    return compose_framed(camera_sensor_from_global, global_from_lidar_sensor)
