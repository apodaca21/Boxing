"""Pruebas unitarias para las características biomecánicas."""

from __future__ import annotations

import math
import unittest

from app.geometry import (
    RelativePoint,
    calculate_angle,
    extract_biomechanical_features,
    left_elbow_angle,
    left_shoulder_wrist_distance,
    left_wrist_relative,
    normalized_distance,
    right_elbow_angle,
    right_shoulder_wrist_distance,
    right_wrist_relative,
    wrist_separation,
)
from app.pose_tracker import LandmarkId, LandmarkObservation, PoseObservation


def point(
    x: float,
    y: float,
    *,
    visibility: float = 1.0,
    presence: float = 1.0,
) -> LandmarkObservation:
    return LandmarkObservation(
        x=x,
        y=y,
        z=0.0,
        visibility=visibility,
        presence=presence,
    )


def pose_with(
    replacements: dict[LandmarkId, LandmarkObservation],
    timestamp_ms: int = 123,
) -> PoseObservation:
    landmarks = [point(0.0, 0.0, visibility=0.0, presence=0.0) for _ in range(33)]
    for landmark_id, landmark in replacements.items():
        landmarks[int(landmark_id)] = landmark
    return PoseObservation(timestamp_ms=timestamp_ms, landmarks=tuple(landmarks))


class CalculateAngleTests(unittest.TestCase):
    def test_right_angle(self) -> None:
        result = calculate_angle(point(0.0, 1.0), point(0.0, 0.0), point(1.0, 0.0))
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result, 90.0)

    def test_straight_angle(self) -> None:
        result = calculate_angle(point(0.0, 0.0), point(1.0, 0.0), point(2.0, 0.0))
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result, 180.0)

    def test_zero_angle(self) -> None:
        result = calculate_angle(point(2.0, 0.0), point(1.0, 0.0), point(3.0, 0.0))
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result, 0.0)

    def test_coincident_points_are_invalid(self) -> None:
        self.assertIsNone(
            calculate_angle(point(0.0, 0.0), point(0.0, 0.0), point(1.0, 0.0))
        )

    def test_missing_point_is_invalid(self) -> None:
        self.assertIsNone(calculate_angle(None, point(0.0, 0.0), point(1.0, 0.0)))

    def test_non_finite_point_is_invalid(self) -> None:
        self.assertIsNone(
            calculate_angle(
                point(math.nan, 0.0),
                point(0.0, 0.0),
                point(1.0, math.inf),
            )
        )

    def test_low_confidence_point_is_invalid(self) -> None:
        self.assertIsNone(
            calculate_angle(
                point(0.0, 1.0, visibility=0.2),
                point(0.0, 0.0),
                point(1.0, 0.0),
                min_visibility=0.5,
            )
        )


class NormalizedDistanceTests(unittest.TestCase):
    def test_normalized_distance(self) -> None:
        result = normalized_distance(
            point(0.0, 0.0),
            point(0.0, 2.0),
            point(0.0, 0.0),
            point(1.0, 0.0),
        )
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result, 2.0)

    def test_zero_reference_scale_is_invalid(self) -> None:
        self.assertIsNone(
            normalized_distance(
                point(0.0, 0.0),
                point(0.0, 2.0),
                point(1.0, 1.0),
                point(1.0, 1.0),
            )
        )

    def test_missing_joint_is_invalid(self) -> None:
        self.assertIsNone(
            normalized_distance(
                point(0.0, 0.0),
                None,
                point(0.0, 0.0),
                point(1.0, 0.0),
            )
        )


class BoxingFeatureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pose = pose_with(
            {
                LandmarkId.LEFT_SHOULDER: point(0.4, 0.4),
                LandmarkId.RIGHT_SHOULDER: point(0.6, 0.4),
                LandmarkId.LEFT_ELBOW: point(0.4, 0.5),
                LandmarkId.RIGHT_ELBOW: point(0.7, 0.4),
                LandmarkId.LEFT_WRIST: point(0.55, 0.5),
                LandmarkId.RIGHT_WRIST: point(0.8, 0.4),
            }
        )
        self.image_size = (640, 480)

    def test_left_and_right_elbow_angles(self) -> None:
        left = left_elbow_angle(self.pose, image_size=self.image_size)
        right = right_elbow_angle(self.pose, image_size=self.image_size)
        self.assertIsNotNone(left)
        self.assertIsNotNone(right)
        self.assertAlmostEqual(left, 90.0)
        self.assertAlmostEqual(right, 180.0)

    def test_shoulder_wrist_distances_use_shoulder_scale(self) -> None:
        left = left_shoulder_wrist_distance(self.pose, image_size=self.image_size)
        right = right_shoulder_wrist_distance(self.pose, image_size=self.image_size)
        self.assertIsNotNone(left)
        self.assertIsNotNone(right)
        self.assertAlmostEqual(left, math.hypot(96.0, 48.0) / 128.0)
        self.assertAlmostEqual(right, 1.0)

    def test_missing_pose_returns_no_measurements(self) -> None:
        empty_pose = PoseObservation(timestamp_ms=100)
        features = extract_biomechanical_features(
            empty_pose,
            image_size=self.image_size,
        )
        self.assertEqual(features.timestamp_ms, 100)
        self.assertIsNone(features.left_elbow_angle)
        self.assertIsNone(features.right_elbow_angle)
        self.assertIsNone(features.left_shoulder_wrist_distance)
        self.assertIsNone(features.right_shoulder_wrist_distance)

    def test_low_confidence_wrist_hides_affected_measurements(self) -> None:
        low_confidence_pose = pose_with(
            {
                LandmarkId.LEFT_SHOULDER: point(0.4, 0.4),
                LandmarkId.RIGHT_SHOULDER: point(0.6, 0.4),
                LandmarkId.LEFT_ELBOW: point(0.4, 0.5),
                LandmarkId.LEFT_WRIST: point(0.5, 0.5, presence=0.1),
            }
        )
        self.assertIsNone(
            left_elbow_angle(
                low_confidence_pose,
                min_presence=0.5,
                image_size=self.image_size,
            )
        )
        self.assertIsNone(
            left_shoulder_wrist_distance(
                low_confidence_pose,
                min_presence=0.5,
                image_size=self.image_size,
            )
        )


class RelativePositionTests(unittest.TestCase):
    def test_wrist_position_uses_shoulder_midpoint_and_scale(self) -> None:
        pose = pose_with(
            {
                LandmarkId.LEFT_SHOULDER: point(0.0, 0.0),
                LandmarkId.RIGHT_SHOULDER: point(2.0, 0.0),
                LandmarkId.LEFT_WRIST: point(1.0, 2.0),
                LandmarkId.RIGHT_WRIST: point(3.0, 2.0),
            }
        )

        self.assertEqual(left_wrist_relative(pose), RelativePoint(0.0, 1.0))
        self.assertEqual(right_wrist_relative(pose), RelativePoint(1.0, 1.0))
        self.assertAlmostEqual(wrist_separation(pose) or 0.0, 1.0)

    def test_missing_or_degenerate_scale_returns_none(self) -> None:
        missing = PoseObservation(timestamp_ms=1)
        coincident = pose_with(
            {
                LandmarkId.LEFT_SHOULDER: point(1.0, 1.0),
                LandmarkId.RIGHT_SHOULDER: point(1.0, 1.0),
                LandmarkId.LEFT_WRIST: point(2.0, 2.0),
            }
        )
        low_confidence = pose_with(
            {
                LandmarkId.LEFT_SHOULDER: point(0.0, 0.0),
                LandmarkId.RIGHT_SHOULDER: point(2.0, 0.0),
                LandmarkId.LEFT_WRIST: point(1.0, 2.0, visibility=0.1),
            }
        )

        self.assertIsNone(left_wrist_relative(missing))
        self.assertIsNone(right_wrist_relative(missing))
        self.assertIsNone(wrist_separation(missing))
        self.assertIsNone(left_wrist_relative(coincident))
        self.assertIsNone(left_wrist_relative(low_confidence, min_visibility=0.5))


if __name__ == "__main__":
    unittest.main()
