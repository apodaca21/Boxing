"""Pruebas de conversión externa sin descargar ni ejecutar repositorios."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

from app.external_data.conversion import convert_external_label


class ExternalLabelConversionTests(unittest.TestCase):
    def test_explicit_left_right_labels_map(self) -> None:
        self.assertEqual(
            convert_external_label("left hook", dataset="Punch_DL").canonical_label_id,
            3,
        )
        self.assertEqual(
            convert_external_label("right hook", dataset="Punch_DL").canonical_label_id,
            4,
        )
        self.assertEqual(
            convert_external_label("3", dataset="Punch_DL").canonical_label_id,
            3,
        )

    def test_generic_labels_are_not_given_a_side(self) -> None:
        for label in ("HOOK", "straight", "jab", "cross", "none", "uppercut"):
            result = convert_external_label(label, dataset="external")
            self.assertEqual(result.status, "excluded")
            self.assertIsNone(result.canonical_label_id)

    def test_generic_hook_can_only_use_independent_arm_evidence(self) -> None:
        result = convert_external_label(
            "HOOK",
            dataset="external",
            arm="right",
            arm_evidence=True,
        )
        self.assertEqual(result.canonical_label_id, 4)

    def test_lead_rear_requires_declared_orthodox_stance(self) -> None:
        without_stance = convert_external_label(
            "Lead Hook",
            dataset="BoxingVI",
        )
        orthodox = convert_external_label(
            "Lead Hook",
            dataset="BoxingVI",
            stance="orthodox",
        )
        self.assertIsNone(without_stance.canonical_label_id)
        self.assertEqual(orthodox.canonical_label_id, 3)

    def test_external_registry_has_no_unverified_download(self) -> None:
        manifest = json.loads(
            (Path(__file__).parents[1] / "data" / "external" / "manifest.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(manifest["downloaded_files"], [])
        self.assertEqual(manifest["downloaded_sample_count"], 0)


if __name__ == "__main__":
    unittest.main()
