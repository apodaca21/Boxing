"""Entrenamiento NEAT y línea base de topología fija."""

from __future__ import annotations

from dataclasses import dataclass
import copy
from pathlib import Path
import time

import neat

from app.evolution.metrics import FitnessBreakdown, fitness_from_predictions, predict_label


PRESETS = {
    "quick": {"population": 30, "generations": 5, "max_stagnation": 5},
    "initial": {"population": 100, "generations": 50, "max_stagnation": 15},
}


@dataclass(frozen=True, slots=True)
class GenerationRecord:
    generation: int
    best_fitness: float
    mean_fitness: float
    species_count: int
    best_enabled_connections: int


def enabled_connection_count(genome: neat.DefaultGenome) -> int:
    return sum(1 for gene in genome.connections.values() if gene.enabled)


def neat_parameter_count(genome: neat.DefaultGenome) -> int:
    return enabled_connection_count(genome) + len(genome.nodes)


def render_config(
    template: str,
    *,
    population: int,
    inputs: int,
    max_stagnation: int,
) -> str:
    return (
        template.replace("__POP_SIZE__", str(population))
        .replace("__NUM_INPUTS__", str(inputs))
        .replace("__MAX_STAGNATION__", str(max_stagnation))
    )


def load_neat_config(path: Path) -> neat.Config:
    return neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        str(path),
    )


class _ProgressReporter(neat.reporting.BaseReporter):
    def __init__(self, generations: int) -> None:
        self.generations = generations
        self.history: list[GenerationRecord] = []
        self.winner: neat.DefaultGenome | None = None

    def post_evaluate(self, config, population, species, best_genome) -> None:  # type: ignore[no-untyped-def]
        fitnesses = [genome.fitness for genome in population.values()]
        record = GenerationRecord(
            generation=len(self.history) + 1,
            best_fitness=float(best_genome.fitness),
            mean_fitness=sum(fitnesses) / len(fitnesses),
            species_count=len(species.species),
            best_enabled_connections=enabled_connection_count(best_genome),
        )
        self.history.append(record)
        if self.winner is None or best_genome.fitness > self.winner.fitness:
            self.winner = copy.deepcopy(best_genome)
        print(
            f"Generación {record.generation}/{self.generations} | "
            f"mejor fitness {record.best_fitness:.4f} | "
            f"promedio {record.mean_fitness:.4f} | "
            f"especies {record.species_count} | "
            f"conexiones {record.best_enabled_connections}",
            flush=True,
        )


def train_population(
    vectors: list[tuple[float, ...]],
    labels: list[int],
    config: neat.Config,
    *,
    generations: int,
    seed: int,
) -> tuple[neat.DefaultGenome, tuple[GenerationRecord, ...]]:
    if len(vectors) != len(labels) or not vectors:
        raise ValueError("El entrenamiento necesita vectores y etiquetas alineados.")

    def evaluate(genomes: list[tuple[int, neat.DefaultGenome]], genome_config: neat.Config) -> None:
        for _genome_id, genome in genomes:
            network = neat.nn.FeedForwardNetwork.create(genome, genome_config)
            predicted = [predict_label(network.activate(vector)) for vector in vectors]
            breakdown = fitness_from_predictions(
                labels,
                predicted,
                enabled_connection_count(genome),
            )
            genome.fitness = float(breakdown.value)

    population = neat.Population(config, seed=seed)
    reporter = _ProgressReporter(generations)
    population.add_reporter(reporter)
    population.run(evaluate, generations)
    if reporter.winner is None:
        raise RuntimeError("NEAT terminó sin un genoma evaluado.")
    return reporter.winner, tuple(reporter.history)


def predict_vectors(
    genome: neat.DefaultGenome,
    config: neat.Config,
    vectors: list[tuple[float, ...]],
) -> list[int]:
    network = neat.nn.FeedForwardNetwork.create(genome, config)
    return [predict_label(network.activate(vector)) for vector in vectors]


def activation_vectors(
    genome: neat.DefaultGenome,
    config: neat.Config,
    vectors: list[tuple[float, ...]],
) -> list[tuple[float, ...]]:
    network = neat.nn.FeedForwardNetwork.create(genome, config)
    return [tuple(float(value) for value in network.activate(vector)) for vector in vectors]


def fitness_breakdown_for(
    genome: neat.DefaultGenome,
    config: neat.Config,
    vectors: list[tuple[float, ...]],
    labels: list[int],
) -> FitnessBreakdown:
    return fitness_from_predictions(
        labels,
        predict_vectors(genome, config, vectors),
        enabled_connection_count(genome),
    )
