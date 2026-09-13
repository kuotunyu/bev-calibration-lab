"""Fixed native-image edge profile and cached scoring preserve exact operator values."""

import numpy as np
import pytest
from PIL import Image
from scipy.ndimage import distance_transform_edt, sobel


def test_native_sobel_profile_matches_explicit_calculation() -> None:
    from bevcalib.operators.image_edges import IMAGE_EDGE_POLICY, image_edge_evidence

    rgb = np.zeros((7, 9, 3), dtype=np.uint8)
    rgb[:, 4:] = [200, 100, 20]
    gray = np.asarray(Image.fromarray(rgb).convert("L"), dtype=np.float64)
    magnitude = np.hypot(sobel(gray, axis=0, mode="reflect"), sobel(gray, axis=1, mode="reflect"))
    expected_threshold = float(np.quantile(magnitude[magnitude > 0], 0.90, method="linear"))
    expected = (magnitude > 0) & (magnitude >= expected_threshold)
    evidence = image_edge_evidence(rgb)
    assert evidence.threshold == expected_threshold
    assert evidence.policy == IMAGE_EDGE_POLICY
    np.testing.assert_array_equal(evidence.mask, expected)
    np.testing.assert_array_equal(evidence.distance_field, distance_transform_edt(~expected))
    assert evidence.mask.shape == (7, 9)


def test_constant_image_is_explicitly_unmeasurable() -> None:
    from bevcalib.operators.image_edges import image_edge_evidence

    evidence = image_edge_evidence(np.full((4, 6, 3), 128, dtype=np.uint8))
    assert not evidence.mask.any()
    assert evidence.mask.shape == (4, 6)
    assert evidence.mask.dtype == np.bool_
    assert evidence.threshold is None
    assert evidence.distance_field is None


def test_edge_threshold_keeps_strong_contrast_and_rejects_weak_contrast() -> None:
    from bevcalib.operators.image_edges import image_edge_evidence

    rgb = np.zeros((6, 12, 3), dtype=np.uint8)
    rgb[:, 4:8] = 10
    rgb[:, 8:] = 255

    evidence = image_edge_evidence(rgb, with_distance_field=False)

    assert not evidence.mask[:, 3:5].any()
    assert evidence.mask[:, 7:9].all()


@pytest.mark.parametrize(
    "rgb", [np.zeros((4, 6)), np.zeros((4, 6, 3)), np.zeros((0, 6, 3), dtype=np.uint8)]
)
def test_invalid_image_input_is_refused(rgb) -> None:
    from bevcalib.operators.image_edges import image_edge_evidence

    with pytest.raises(ValueError, match="RGB uint8"):
        image_edge_evidence(rgb)


@pytest.mark.parametrize("quantile", [0.1, 0.9, 1.0])
def test_cached_score_is_identical_to_existing_mask_operator(quantile: float) -> None:
    from bevcalib.operators.lidar_edges import (
        trimmed_distance_field_score,
        trimmed_distance_transform_score,
    )

    edges = np.zeros((9, 13), dtype=bool)
    edges[:, 5] = True
    points = np.array([[0.2, 0.9], [1.9, 3.1], [5.0, 4.0], [8.8, 2.0], [12.9, 8.9]])
    assert trimmed_distance_field_score(
        points, distance_transform_edt(~edges), quantile
    ) == trimmed_distance_transform_score(points, edges, quantile)


@pytest.mark.parametrize(
    "field", [np.zeros((2,)), np.zeros((0, 2)), np.full((2, 2), -1.0), np.full((2, 2), np.nan)]
)
def test_cached_score_refuses_invalid_distance_field(field) -> None:
    from bevcalib.operators.lidar_edges import trimmed_distance_field_score

    with pytest.raises(ValueError, match="distance field"):
        trimmed_distance_field_score(np.array([[0.0, 0.0]]), field)


def test_threshold_only_pass_does_not_build_a_distance_field() -> None:
    from bevcalib.operators.image_edges import image_edge_evidence

    rgb = np.zeros((4, 6, 3), dtype=np.uint8)
    rgb[:, 3:] = 255
    actual = image_edge_evidence(rgb, with_distance_field=False)
    complete = image_edge_evidence(rgb)
    assert actual.threshold == complete.threshold
    np.testing.assert_array_equal(actual.mask, complete.mask)
    assert actual.distance_field is None
