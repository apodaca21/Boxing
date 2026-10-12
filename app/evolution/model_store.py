"""Persistencia del genoma ganador y de la transformación usada al entrenar."""

from __future__ import annotations

from dataclasses import dataclass
import json
import pickle
from pathlib import Path
import tempfile

import neat

from app.dataset_manager import MOVEMENTS, MovementSample
from app.evolution.features import FeatureScaler, extract_raw_features
from app.evolution.metrics import predict_label
from app.evolution.neat_trainer import load_neat_config


@dataclass(frozen=True, slots=True)
class EvolvedModel:
    genome: neat.DefaultGenome
    config_text: str
    scaler: FeatureScaler
    seed: int
    training: dict[str, object]

    def neat_config(self) -> neat.Config:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "neat.cfg"
            path.write_text(self.config_text, encoding="utf-8")
            return load_neat_config(path)

    def predict_sample(self, sample: MovementSample) -> int:
        vector = self.scaler.transform(
            extract_raw_features(sample, self.scaler.time_steps)
        )
        network = neat.nn.FeedForwardNetwork.create(self.genome, self.neat_config())
        return predict_label(network.activate(vector))

    def predict_vector(self, vector: tuple[float, ...]) -> tuple[float, ...]:
        network = neat.nn.FeedForwardNetwork.create(self.genome, self.neat_config())
        return tuple(float(value) for value in network.activate(vector))


def save_model(path: Path, model: EvolvedModel) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "neat.cfg").write_text(model.config_text, encoding="utf-8")
    (path / "feature_spec.json").write_text(
        json.dumps(model.scaler.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "seed": model.seed,
        "label_map": {
            str(label_id): {
                "name": movement.name,
                "display_name": movement.display_name,
            }
            for label_id, movement in MOVEMENTS.items()
        },
        "selection_rule": "argmax de cinco salidas; no son probabilidades calibradas",
        "winner_selection": "mayor fitness calculado solamente en entrenamiento",
        "fitness_formula": "0.65*R_punch + 0.30*S_neutral + 0.05/(1+C)",
        "fitness_coefficients": "valores iniciales propuestos; todavía deben validarse experimentalmente",
        "training": model.training,
    }
    (path / "training_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (path / "winner.pkl").open("wb") as handle:
        pickle.dump(model.genome, handle)


def load_model(path: Path) -> EvolvedModel:
    scaler = FeatureScaler.from_dict(
        json.loads((path / "feature_spec.json").read_text(encoding="utf-8"))
    )
    manifest = json.loads((path / "training_manifest.json").read_text(encoding="utf-8"))
    with (path / "winner.pkl").open("rb") as handle:
        genome = pickle.load(handle)
    return EvolvedModel(
        genome=genome,
        config_text=(path / "neat.cfg").read_text(encoding="utf-8"),
        scaler=scaler,
        seed=int(manifest["seed"]),
        training=dict(manifest["training"]),
    )
