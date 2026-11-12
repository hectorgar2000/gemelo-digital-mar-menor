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


def mejora_bootstrap(e_ref, e_mod, bloque: int = 30, n: int = 2000, semilla: int = 0, q=(0.05, 0.95)):
    """IC de la mejora de RMSE (1 - RMSE_mod / RMSE_ref) remuestreando BLOQUES de días consecutivos: los
    errores diarios están autocorrelados y un bootstrap día a día daría intervalos demasiado estrechos."""
    e_ref, e_mod = np.asarray(e_ref, dtype=float), np.asarray(e_mod, dtype=float)
    bloque = min(bloque, len(e_ref))
    rng = np.random.default_rng(semilla)
    inicios = np.arange(0, len(e_ref) - bloque + 1)
    nb = int(np.ceil(len(e_ref) / bloque))
    sk = []
    for _ in range(n):
        idx = (rng.choice(inicios, nb)[:, None] + np.arange(bloque)).ravel()
        sk.append(1 - np.sqrt(np.mean(e_mod[idx] ** 2)) / np.sqrt(np.mean(e_ref[idx] ** 2)))
    return tuple(float(x) for x in np.quantile(sk, q))
