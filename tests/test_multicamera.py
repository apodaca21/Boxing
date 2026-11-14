"""Pruebas de selección de roles y apoyo de la cámara depth."""

from __future__ import annotations

from unittest.mock import patch
import unittest

import numpy as np

from app.camera import CameraInfo, CameraRig, FramePacket
from app.camera_identity import NamedCamera, suggest_depth_pair
from app.config import AppConfig
from app.multiview import DepthReadiness, MultiViewMonitor, _DepthMotionCue, _focus_measure
from app.pose_tracker import LandmarkId, LandmarkObservation, PoseObservation
from app.trainer.results import ClassificationResult, PunchEvent
from main import parse_args


class MultiCameraSelectionTests(unittest.TestCase):
    def test_single_keeps_only_front(self) -> None:
        rig = CameraRig(front_index=2, setup="single")
        self.assertEqual(rig._resolve_indices(), {"front": 2})

    @patch("app.camera.probe_cameras")
    def test_auto_assigns_extras_to_left_and_right(self, probe) -> None:
        probe.return_value = tuple(
            CameraInfo(index=index, available=True, width=640, height=480)
            for index in (0, 1, 2)
        )
        rig = CameraRig(front_index=0, setup="auto")
        self.assertEqual(
            rig._resolve_indices(),
            {"front": 0, "left": 1, "right": 2},
        )

    @patch("app.camera.probe_cameras")
    def test_depth_keeps_explicit_depth_role(self, probe) -> None:
        probe.return_value = tuple(
            CameraInfo(index=index, available=True, width=640, height=480)
            for index in (0, 1)
        )
        rig = CameraRig(front_index=0, depth_index=1, setup="depth")
        self.assertEqual(rig._resolve_indices(), {"front": 0, "depth": 1})

    def test_config_rejects_duplicate_roles(self) -> None:
        with self.assertRaises(ValueError):
            AppConfig(
                camera_setup="triple",
                camera_left_index=1,
                camera_right_index=1,
            ).validate()

    def test_cli_exposes_roles_and_depth_gate(self) -> None:
        import sys

        original = sys.argv
        try:
            sys.argv = [
                "main.py",
                "--camera-setup",
                "depth",
                "--camera-depth-index",
                "2",
                "--depth-gate",
                "strict",
            ]
            args = parse_args()
        finally:
            sys.argv = original
        self.assertEqual(args.camera_setup, "depth")
        self.assertEqual(args.camera_depth_index, 2)
        self.assertEqual(args.depth_gate, "strict")

    def test_suggest_prefers_macbook_front_and_steren_depth(self) -> None:
        cameras = (
            NamedCamera(0, "Steren COM-126", True, 140.0, 640, 480),
            NamedCamera(1, "MacBook Air Camera", True, 110.0, 640, 480),
            NamedCamera(2, "Alejandro’s iPhone Camera", True, 90.0, 640, 480),
        )
        pair = suggest_depth_pair(cameras)
        self.assertIsNotNone(pair)
        assert pair is not None
        self.assertEqual(pair.front_index, 1)
        self.assertEqual(pair.depth_index, 0)


class DepthAssistTests(unittest.TestCase):
    @staticmethod
    def _good_depth_pose() -> PoseObservation:
        slots = [LandmarkObservation(0.0, 0.0, 0.0, 0.0, 0.0) for _ in range(17)]
        for landmark_id, x, y in (
            (LandmarkId.LEFT_SHOULDER, 0.35, 0.35),
            (LandmarkId.RIGHT_SHOULDER, 0.65, 0.35),
            (LandmarkId.LEFT_WRIST, 0.30, 0.58),
            (LandmarkId.RIGHT_WRIST, 0.70, 0.58),
        ):
            slots[int(landmark_id)] = LandmarkObservation(x, y, 0.0, 1.0, 1.0)
        return PoseObservation(0, tuple(slots), tuple(slots))

    def test_depth_readiness_requires_focus_detection_and_stability(self) -> None:
        monitor = MultiViewMonitor(
            AppConfig(camera_setup="depth", camera_depth_index=1)
        )
        pattern = np.indices((120, 160)).sum(axis=0).astype(np.uint8) * 127
        frame = np.repeat(pattern[:, :, None], 3, axis=2)
        packet = FramePacket(frame, 0, 0, 160, 120)
        pose = self._good_depth_pose()

        readiness = monitor._depth_readiness(packet, pose)
        self.assertFalse(readiness.ready)
        self.assertIn("ESTABILIZANDO", readiness.status)
        for _ in range(8):
            readiness = monitor._depth_readiness(packet, pose)
        self.assertTrue(readiness.ready)
        self.assertIn("DEPTH LISTA", readiness.status)

    def test_focus_measure_drops_when_depth_view_is_blurred(self) -> None:
        sharp = np.indices((120, 160)).sum(axis=0).astype(np.uint8) * 127
        blurred = np.full_like(sharp, 127)
        self.assertGreater(_focus_measure(sharp), _focus_measure(blurred))

    def test_depth_event_is_fused_when_front_view_misses_it(self) -> None:
        monitor = MultiViewMonitor(
            AppConfig(camera_setup="depth", camera_depth_index=1, depth_gate="strict")
        )
        monitor.readiness = DepthReadiness(camera_ok=True, detected=True, ready=True, status="DEPTH LISTA")
        monitor._depth_result = ClassificationResult(
            source="heuristic",
            status="completado",
            label_id=1,
            display_name="Jab izquierdo",
            events=(
                PunchEvent(
                    1,
                    "Jab izquierdo",
                    "left",
                    event_id="left:120",
                    started_ms=80,
                    ended_ms=120,
                ),
            ),
            active_arm="left",
        )
        front = ClassificationResult(
            source="heuristic",
            status="neutral",
            label_id=0,
            display_name="Guardia / Neutral",
        )

        fused = monitor.filter_result(front, 120)
        self.assertEqual(len(fused.events), 1)
        self.assertEqual(fused.events[0].source, "depth")
        self.assertEqual(fused.diagnostics.get("depth_fusion"), "added")

    def test_same_punch_seen_by_both_cameras_is_not_duplicated(self) -> None:
        monitor = MultiViewMonitor(
            AppConfig(camera_setup="depth", camera_depth_index=1, depth_gate="soft")
        )
        monitor._depth_result = ClassificationResult(
            source="heuristic",
            status="completado",
            label_id=3,
            display_name="Gancho izquierdo",
            events=(
                PunchEvent(3, "Gancho izquierdo", "left", event_id="depth:left:220", ended_ms=220),
            ),
            active_arm="left",
        )
        front_event = PunchEvent(
            1,
            "Jab izquierdo",
            "left",
            event_id="front:left:210",
            ended_ms=210,
        )
        front = ClassificationResult(
            source="heuristic",
            status="completado",
            label_id=1,
            display_name="Jab izquierdo",
            events=(front_event,),
            active_arm="left",
        )

        fused = monitor.filter_result(front, 220)
        self.assertEqual(len(fused.events), 1)
        self.assertEqual(fused.events[0].event_id, "front:left:210")
        # Depth ya no pisa jab<->gancho a ciegas; el cue de movimiento lo corrige.
        self.assertEqual(fused.events[0].label_id, 1)
        self.assertEqual(fused.events[0].display_name, "Jab izquierdo")
        self.assertEqual(fused.diagnostics.get("depth_fusion"), "front_class_kept")

    def test_same_label_from_both_cameras_is_not_duplicated(self) -> None:
        monitor = MultiViewMonitor(
            AppConfig(camera_setup="depth", camera_depth_index=1, depth_gate="soft")
        )
        monitor._depth_result = ClassificationResult(
            source="heuristic",
            status="completado",
            label_id=1,
            display_name="Jab izquierdo",
            events=(
                PunchEvent(1, "Jab izquierdo", "left", event_id="depth:left:220", ended_ms=220),
            ),
            active_arm="left",
        )
        front = ClassificationResult(
            source="heuristic",
            status="completado",
            label_id=1,
            display_name="Jab izquierdo",
            events=(
                PunchEvent(1, "Jab izquierdo", "left", event_id="front:left:210", ended_ms=210),
            ),
            active_arm="left",
        )
        fused = monitor.filter_result(front, 220)
        self.assertEqual(len(fused.events), 1)
        self.assertEqual(fused.events[0].event_id, "front:left:210")
        self.assertEqual(fused.diagnostics.get("depth_fusion"), "same_event")

    def test_front_camera_wins_when_auxiliary_inverts_anatomical_side(self) -> None:
        monitor = MultiViewMonitor(
            AppConfig(camera_setup="depth", camera_depth_index=1, depth_gate="soft")
        )
        monitor._depth_result = ClassificationResult(
            source="heuristic",
            status="completado",
            label_id=3,
            display_name="Gancho izquierdo",
            events=(PunchEvent(3, "Gancho izquierdo", "left", event_id="depth:left:220", ended_ms=220),),
            active_arm="left",
        )
        front = ClassificationResult(
            source="heuristic",
            status="completado",
            label_id=4,
            display_name="Gancho derecho",
            events=(PunchEvent(4, "Gancho derecho", "right", event_id="front:right:220", ended_ms=220),),
            active_arm="right",
        )
        fused = monitor.filter_result(front, 220)
        self.assertEqual(fused.events[0].arm, "right")
        self.assertEqual(fused.events[0].label_id, 4)
        self.assertIn("front_side_conflict", str(fused.diagnostics.get("depth_fusion")))

    def test_late_front_event_does_not_double_count_depth_punch(self) -> None:
        monitor = MultiViewMonitor(
            AppConfig(camera_setup="depth", camera_depth_index=1, depth_gate="soft")
        )
        monitor.readiness = DepthReadiness(camera_ok=True, detected=True, ready=True, status="DEPTH LISTA")
        monitor._depth_result = ClassificationResult(
            source="heuristic",
            status="completado",
            label_id=4,
            display_name="Gancho derecho",
            events=(
                PunchEvent(4, "Gancho derecho", "right", event_id="depth:right:100", ended_ms=100),
            ),
            active_arm="right",
        )
        empty_front = ClassificationResult(
            source="heuristic",
            status="neutral",
            label_id=0,
            display_name="Guardia / Neutral",
        )
        first = monitor.filter_result(empty_front, 100)
        self.assertEqual(len(first.events), 1)
        self.assertEqual(first.events[0].source, "depth")

        monitor._depth_result = ClassificationResult(
            source="heuristic",
            status="neutral",
            label_id=0,
            display_name="Guardia / Neutral",
        )
        late_front = ClassificationResult(
            source="heuristic",
            status="completado",
            label_id=2,
            display_name="Cross derecho",
            events=(
                PunchEvent(2, "Cross derecho", "right", event_id="front:right:280", ended_ms=280),
            ),
            active_arm="right",
        )
        second = monitor.filter_result(late_front, 280)
        self.assertEqual(len(second.events), 0)
        self.assertEqual(second.diagnostics.get("depth_fusion"), "suppressed_duplicate")

    def test_depth_stays_ready_while_punching_moves_the_shoulders(self) -> None:
        monitor = MultiViewMonitor(
            AppConfig(camera_setup="depth", camera_depth_index=1)
        )
        pattern = np.indices((120, 160)).sum(axis=0).astype(np.uint8) * 127
        frame = np.repeat(pattern[:, :, None], 3, axis=2)
        packet = FramePacket(frame, 0, 0, 160, 120)
        pose = self._good_depth_pose()
        for _ in range(8):
            readiness = monitor._depth_readiness(packet, pose)
        self.assertTrue(readiness.ready)

        slots = list(pose.landmarks)
        slots[int(LandmarkId.LEFT_SHOULDER)] = LandmarkObservation(0.28, 0.42, 0.0, 1.0, 1.0)
        slots[int(LandmarkId.RIGHT_SHOULDER)] = LandmarkObservation(0.72, 0.30, 0.0, 1.0, 1.0)
        moving = PoseObservation(40, tuple(slots), tuple(slots))
        readiness = monitor._depth_readiness(packet, moving)
        self.assertTrue(readiness.ready)
        self.assertIn("DEPTH LISTA", readiness.status)

    def test_soft_gate_relabels_false_jab_into_hook(self) -> None:
        config = AppConfig(camera_setup="depth", camera_depth_index=1, depth_gate="soft")
        monitor = MultiViewMonitor(config)
        cue = _DepthMotionCue()
        slots = [LandmarkObservation(0.0, 0.0, 0.0, 0.0, 0.0) for _ in range(17)]
        # Guardia
        slots[int(LandmarkId.LEFT_SHOULDER)] = LandmarkObservation(0.4, 0.4, 0.0, 1.0, 1.0)
        slots[int(LandmarkId.LEFT_WRIST)] = LandmarkObservation(0.42, 0.55, 0.0, 1.0, 1.0)
        cue.observe(0, PoseObservation(0, tuple(slots), tuple(slots)))
        # Barrido lateral amplio con poco alcance: tipico de gancho en vista lateral.
        slots[int(LandmarkId.LEFT_WRIST)] = LandmarkObservation(0.18, 0.52, 0.0, 1.0, 1.0)
        cue.observe(40, PoseObservation(40, tuple(slots), tuple(slots)))
        monitor._depth = cue
        result = ClassificationResult(
            source="heuristic",
            status="completado",
            label_id=1,
            display_name="Jab izquierdo",
            events=(PunchEvent(1, "Jab izquierdo", "left", event_id="left:1"),),
            active_arm="left",
        )
        assisted = monitor.filter_result(result, 40)
        self.assertEqual([event.label_id for event in assisted.events], [3])
        self.assertEqual(assisted.diagnostics.get("depth_assist"), "relabeled")


if __name__ == "__main__":
    unittest.main()

