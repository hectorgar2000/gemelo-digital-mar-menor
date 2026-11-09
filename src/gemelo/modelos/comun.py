"""Utilidades comunes de modelado: columnas, métricas y configuración de MLflow."""
import mlflow
import numpy as np
import pandas as pd

from gemelo.config import RAIZ

RUTA_MLFLOW_DB = RAIZ / "mlflow.db"
RUTA_ARTEFACTOS = RAIZ / "mlartifacts"
MANIFIESTO = RAIZ / "data" / "processed" / "manifest.json"
FIGURAS = RAIZ / "docs" / "figuras" / "fase3"


def columnas_x(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if not c.startswith("y_") and c not in ("particion", "anio")]


def rmse(y, p) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def mae(y, p) -> float:
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


def skill(error_modelo: float, error_ref: float) -> float:
    """1 = perfecto, 0 = igual que la referencia, <0 = peor que la referencia."""
    return 1 - error_modelo / error_ref


def configurar_mlflow(experimento: str) -> str:
    """Tracking en SQLite local (necesario para el Model Registry de la Fase 4)."""
    mlflow.set_tracking_uri(f"sqlite:///{RUTA_MLFLOW_DB.as_posix()}")
    exp = mlflow.get_experiment_by_name(experimento)
    if exp is None:
        RUTA_ARTEFACTOS.mkdir(exist_ok=True)
        exp_id = mlflow.create_experiment(experimento, artifact_location=RUTA_ARTEFACTOS.as_uri())
    else:
        exp_id = exp.experiment_id
    mlflow.set_experiment(experiment_id=exp_id)
    return exp_id
