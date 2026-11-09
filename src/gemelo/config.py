"""Carga de la configuración del proyecto (config/zona.yaml)."""
from pathlib import Path

import yaml

RAIZ = Path(__file__).resolve().parents[2]
RUTA_CONFIG = RAIZ / "config" / "zona.yaml"
DATA_RAW = RAIZ / "data" / "raw"


def cargar_config(ruta: Path = RUTA_CONFIG) -> dict:
    with open(ruta, encoding="utf-8") as f:
        return yaml.safe_load(f)
