"""Carga de la configuración del proyecto (config/zona.yaml)."""
import os
from pathlib import Path

import yaml

# Raíz del proyecto. En desarrollo (instalación editable) se deduce de la ubicación del código;
# en Docker el paquete vive en site-packages, así que se fija con la variable GEMELO_RAIZ.
RAIZ = Path(os.environ.get("GEMELO_RAIZ", Path(__file__).resolve().parents[2]))
RUTA_CONFIG = RAIZ / "config" / "zona.yaml"
DATA_RAW = RAIZ / "data" / "raw"


def cargar_config(ruta: Path = RUTA_CONFIG) -> dict:
    with open(ruta, encoding="utf-8") as f:
        return yaml.safe_load(f)
