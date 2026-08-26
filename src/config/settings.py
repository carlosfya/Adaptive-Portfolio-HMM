"""Carga de la configuración única del sistema (config.yaml)."""

from pathlib import Path

import yaml

# Raíz del repositorio (dos niveles por encima de src/config/)
ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "output"
DATA_DIR = OUTPUT_DIR / "data"


def load_config(path: str | Path | None = None) -> dict:
    """Lee config.yaml y devuelve el diccionario de configuración.

    Parameters
    ----------
    path : str or Path, optional
        Ruta alternativa al YAML; por defecto ``ROOT/config.yaml``.

    Returns
    -------
    dict
        Configuración completa del sistema.
    """
    path = Path(path) if path is not None else ROOT / "config.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dirs() -> None:
    """Crea los directorios de salida si no existen."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
