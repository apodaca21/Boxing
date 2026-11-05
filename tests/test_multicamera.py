"""Pruebas de selección de roles sin abrir hardware real."""

from __future__ import annotations

from unittest.mock import patch
import unittest

from app.camera import CameraInfo, CameraRig
from app.config import AppConfig
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


if __name__ == "__main__":
    unittest.main()

