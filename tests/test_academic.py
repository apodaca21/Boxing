"""Pruebas de la auditoría académica. Los fixtures no son resultados universitarios."""

from __future__ import annotations

from argparse import Namespace
import json
from pathlib import Path
import tempfile
import unittest

from app.evolution.academic import PENDING_FILES, build_academic_package
from tests.test_evolution import _valid_dataset


def _args(root: Path, **overrides: object) -> Namespace:
    values = {
        "data_dir": root / "data",
        "results_dir": root / "results",
        "time_steps": 8,
        "seed": 42,
        "experiment": "quick",
        "population": 6,
        "generations": 1,
        "split": "session",
    }
    values.update(overrides)
    return Namespace(**values)


class AcademicReportTests(unittest.TestCase):
    def test_empty_dataset_documents_pending_results(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            code = build_academic_package(_args(root))
            academic = root / "results" / "academic"
            summary = json.loads((academic / "dataset_summary.json").read_text(encoding="utf-8"))
            config = json.loads((academic / "experiment_config.json").read_text(encoding="utf-8"))
            report = (academic / "academic_report.md").read_text(encoding="utf-8")
            self.assertEqual(code, 0)
            self.assertEqual(summary["valid_count"], 0)
            self.assertFalse(summary["contains_synthetic_fill"])
            self.assertFalse(summary["contains_experimental_metrics"])
            self.assertFalse(config["experiment_executed"])
            self.assertFalse(config["test_set_used_for_selection"])
            self.assertFalse(config["test_set_used_for_hyperparameters"])
            for name in PENDING_FILES:
                self.assertFalse((academic / name).exists(), name)
            for heading in (
                "## 1. Definición del problema",
                "## 5. Función objetivo matemática",
                "## 7. Resultados reales",
                "## 10. Trabajo pendiente",
            ):
                self.assertIn(heading, report)
            self.assertIn("pendientes", report)
            self.assertIn("video", report)
            self.assertNotIn("95%", report)

    def test_incomplete_classes_are_named_and_not_filled(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _valid_dataset(root / "data")
            for path in (root / "data").rglob("*.json"):
                payload = json.loads(path.read_text(encoding="utf-8"))
                if payload.get("label_id") not in (None, 0):
                    path.unlink()
            code = build_academic_package(_args(root))
            academic = root / "results" / "academic"
            summary = json.loads((academic / "dataset_summary.json").read_text(encoding="utf-8"))
            report = (academic / "academic_report.md").read_text(encoding="utf-8")
            self.assertEqual(code, 0)
            self.assertFalse(summary["ready_for_training"])
            self.assertIn("Jab izquierdo", report)
            self.assertFalse((academic / "confusion_matrix.png").exists())
            self.assertFalse((academic / "neat_evolution.csv").exists())

    def test_ready_fixture_exports_measured_files_without_ranking(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _valid_dataset(root / "data")
            code = build_academic_package(_args(root))
            academic = root / "results" / "academic"
            metrics = json.loads((academic / "evaluation_metrics.json").read_text(encoding="utf-8"))
            history = (academic / "neat_evolution.csv").read_text(encoding="utf-8")
            report = (academic / "academic_report.md").read_text(encoding="utf-8")
            self.assertEqual(code, 0)
            self.assertIn("best_fitness", history)
            self.assertIn("species_count", history)
            self.assertEqual(set(metrics["splits"]), {"entrenamiento", "validacion", "prueba"})
            self.assertFalse(metrics["test_set_used_for_selection"])
            self.assertIsNone(metrics["heuristic_comparison"]["ranking"])
            self.assertEqual(metrics["heuristic_comparison"]["learned_parameters"]["heuristic"], 0)
            self.assertGreater(metrics["training_seconds"], 0)
            for name in PENDING_FILES:
                self.assertTrue((academic / name).is_file(), name)
            self.assertIn("No se declara un modelo ganador", report)
            self.assertIn("entrenamiento", report)
            self.assertIn("prueba", report)


if __name__ == "__main__":
    unittest.main()
