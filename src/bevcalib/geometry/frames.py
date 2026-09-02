"""Frame names attached to transforms, so a mismatched composition cannot happen."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .se3 import SE3, compose

FrameName = Literal["lidar_sensor", "lidar_ego", "global", "camera_ego", "camera_sensor"]

# Written in chain order: a LiDAR point travels along this list to reach the image.
# Two ego frames, not one, because the sweep and the exposure happen at different
# instants and the vehicle moves in between.
FRAME_NAMES: tuple[FrameName, ...] = (
    "lidar_sensor",
    "lidar_ego",
    "global",
    "camera_ego",
    "camera_sensor",
)


@dataclass(frozen=True)
class FramedTransform:
    """One rigid transform that knows which frame it starts from and lands in."""

    target: FrameName
    source: FrameName
    value: SE3

    def __post_init__(self) -> None:
        for role, frame in (("target", self.target), ("source", self.source)):
            if frame not in FRAME_NAMES:
                raise ValueError(
                    f"unknown {role} frame {frame!r}; expected one of {list(FRAME_NAMES)}"
                )


def compose_framed(left: FramedTransform, right: FramedTransform) -> FramedTransform:
    """Compose `T_a_b` with `T_b_c` into `T_a_c`, refusing any gap between b and b.

    The arithmetic in `se3.compose` will happily combine two unrelated transforms
    and return three finite numbers per point. Whether the result means anything
    depends on the frames lining up, which is a fact about the data rather than
    about the matrices, so it is checked here.
    """

    if left.source != right.target:
        raise ValueError(
            f"cannot compose {left.target!r}<-{left.source!r} with "
            f"{right.target!r}<-{right.source!r}: "
            f"the left transform starts from {left.source!r} "
            f"but the right one lands in {right.target!r}"
        )
    return FramedTransform(
        target=left.target,
        source=right.source,
        value=compose(left.value, right.value),
    )
