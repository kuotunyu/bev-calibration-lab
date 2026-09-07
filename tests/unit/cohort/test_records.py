"""A keyframe is three distinct timed observations, with aligned identifiers."""

from dataclasses import FrozenInstanceError, replace

import pytest


@pytest.mark.parametrize(
    "change",
    [
        {"sample_tokens": ()},
        {"camera_sample_data_tokens": ("a", "b")},
        {
            "sample_tokens": ("same", "same"),
            "camera_sample_data_tokens": ("a", "b"),
            "lidar_sample_data_tokens": ("c", "d"),
            "sample_timestamps": (1, 2),
            "camera_timestamps": (2, 3),
            "lidar_timestamps": (0, 1),
        },
        {"location": "unknown"},
        {"official_split": "test"},
        {"camera_timestamps": (-1,)},
        {"sample_tokens": ("scene-camera",)},
    ],
)
def test_invalid_record_is_refused(change):
    from tests.unit.cohort.test_splits import record

    with pytest.raises(ValueError):
        replace(record("scene", "log"), **change)


def test_record_is_frozen():
    from tests.unit.cohort.test_splits import record

    scene = record("scene", "log")
    with pytest.raises(FrozenInstanceError):
        scene.log_token = "other"


def test_nonmonotonic_sensor_times_are_refused():
    from tests.unit.cohort.test_splits import record

    with pytest.raises(ValueError, match="strictly increasing"):
        replace(
            record("scene", "log"),
            sample_tokens=("a", "b"),
            camera_sample_data_tokens=("c", "d"),
            lidar_sample_data_tokens=("e", "f"),
            sample_timestamps=(1, 2),
            camera_timestamps=(2, 2),
            lidar_timestamps=(0, 1),
        )


def test_shared_sensor_identifier_across_scenes_is_refused():
    from tests.unit.cohort.test_splits import record

    from bevcalib.cohort.records import validate_records

    a = record("a", "log")
    b = replace(record("b", "log"), camera_sample_data_tokens=a.camera_sample_data_tokens)
    with pytest.raises(ValueError, match="multiple scenes"):
        validate_records([a, b])
