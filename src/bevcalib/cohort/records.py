"""Validated, immutable keyframe provenance; no dataset access is performed here."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise
from typing import Annotated, Literal

from pydantic import ConfigDict, Field
from pydantic.dataclasses import dataclass

Location = Literal[
    "boston-seaport", "singapore-hollandvillage", "singapore-onenorth", "singapore-queenstown"
]
LOCATIONS: tuple[Location, ...] = (
    "boston-seaport",
    "singapore-hollandvillage",
    "singapore-onenorth",
    "singapore-queenstown",
)
Identifier = Annotated[str, Field(min_length=1)]
Timestamp = Annotated[int, Field(ge=0, strict=True)]


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class SceneRecord:
    """All CAM_FRONT/LIDAR_TOP keyframes, with each sensor's own timestamp."""

    scene_token: Identifier
    log_token: Identifier
    location: Location
    official_split: Literal["train", "val"]
    sample_tokens: tuple[Identifier, ...]
    camera_sample_data_tokens: tuple[Identifier, ...]
    lidar_sample_data_tokens: tuple[Identifier, ...]
    sample_timestamps: tuple[Timestamp, ...]
    camera_timestamps: tuple[Timestamp, ...]
    lidar_timestamps: tuple[Timestamp, ...]

    def __post_init__(self) -> None:
        sequences = (
            self.sample_tokens,
            self.camera_sample_data_tokens,
            self.lidar_sample_data_tokens,
            self.sample_timestamps,
            self.camera_timestamps,
            self.lidar_timestamps,
        )
        if not self.sample_tokens or any(len(s) != len(self.sample_tokens) for s in sequences):
            raise ValueError("keyframe sequences must be aligned and nonempty")
        for tokens in sequences[:3]:
            if len(set(tokens)) != len(tokens):
                raise ValueError("keyframe identifiers must be unique")
        all_tokens = (
            self.sample_tokens + self.camera_sample_data_tokens + self.lidar_sample_data_tokens
        )
        if len(set(all_tokens)) != len(all_tokens):
            raise ValueError("sample, camera and LiDAR identifiers must be distinct")
        for timestamps in (self.sample_timestamps, self.camera_timestamps, self.lidar_timestamps):
            if any(a >= b for a, b in pairwise(timestamps)):
                raise ValueError("each timestamp sequence must be strictly increasing")


def validate_records(records: Sequence[SceneRecord]) -> tuple[SceneRecord, ...]:
    """Deduplicate identical scenes, refusing ambiguous log/scene/token ownership.

    A log has one location. Official splits belong to scenes, so a shared
    train/val log is valid input and allocation must assign it exclusively.
    """

    scenes: dict[str, SceneRecord] = {}
    logs: dict[str, str] = {}
    identifiers: set[str] = set()
    for scene in records:
        if scene.scene_token in scenes:
            if scenes[scene.scene_token] != scene:
                raise ValueError("inconsistent duplicate scene metadata")
            continue
        metadata = scene.location
        if scene.log_token in logs and logs[scene.log_token] != metadata:
            raise ValueError("inconsistent log location metadata")
        tokens = set(
            scene.sample_tokens + scene.camera_sample_data_tokens + scene.lidar_sample_data_tokens
        )
        if identifiers & tokens:
            raise ValueError("keyframe identifier belongs to multiple scenes")
        identifiers.update(tokens)
        scenes[scene.scene_token] = scene
        logs[scene.log_token] = metadata
    return tuple(scenes.values())
