"""Resultados de partida. Viven fuera del dataset de NEAT."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def save_round(results_dir: Path, payload: dict[str, object]) -> Path:
    game = str(payload.get("game") or "round")
    folder = results_dir / "games" / game
    folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    path = folder / f"{stamp}.json"
    body = dict(payload)
    body["saved_at"] = stamp
    path.write_text(json.dumps(body, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def latest_rounds(results_dir: Path, limit: int = 4) -> list[dict[str, object]]:
    folder = results_dir / "games"
    if not folder.is_dir():
        return []
    files = sorted(folder.glob("*/*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    loaded = []
    for path in files[:limit]:
        loaded.append(json.loads(path.read_text(encoding="utf-8")))
    return loaded
