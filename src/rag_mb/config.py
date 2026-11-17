"""Carga de configuración y rutas del proyecto (fuente única: configs/corpora.yaml)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    """Raíz del repo: variable RMB_ROOT o el directorio que contiene configs/."""
    env = os.environ.get("RMB_ROOT")
    if env:
        return Path(env).resolve()
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "configs" / "corpora.yaml").exists():
            return parent
    return Path.cwd()


@dataclass(frozen=True)
class Paths:
    root: Path

    @property
    def raw(self) -> Path:
        return self.root / "data" / "raw"

    @property
    def processed(self) -> Path:
        return self.root / "data" / "processed"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    @property
    def figures(self) -> Path:
        return self.reports / "figures"

    @property
    def questions(self) -> Path:
        return self.root / "questions"

    def corpus_file(self, saga: str) -> Path:
        return self.processed / f"{saga}_units.jsonl"

    def chunks_file(self, saga: str) -> Path:
        return self.processed / f"{saga}_chunks.jsonl"


def paths() -> Paths:
    return Paths(project_root())


@lru_cache(maxsize=1)
def load_config(path: str | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else project_root() / "configs" / "corpora.yaml"
    with open(cfg_path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def saga_config(saga: str) -> dict[str, Any]:
    cfg = load_config()
    if saga not in cfg["sagas"]:
        raise KeyError(f"Saga '{saga}' no está en configs/corpora.yaml: {list(cfg['sagas'])}")
    return cfg["sagas"][saga]
