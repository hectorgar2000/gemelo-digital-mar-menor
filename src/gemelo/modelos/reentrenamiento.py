"""Reentrenamiento campeón vs. retador (Fase 7).

  - CAMPEÓN: el modelo en @produccion.
  - RETADOR: mismo tipo de modelo y mismos hiperparámetros, entrenado con TODOS los datos hasta el inicio de
    la ventana de evaluación (purga de frontera: ningún objetivo cae dentro de la ventana).
  - Ambos se evalúan en los últimos `dias` con verdad conocida, con las mismas variables que en servicio
    (previsión REAL si el modelo la usa). El retador solo gana si mejora >= `mejora_minima`.
  - Si gana, se reajusta con todos los datos (incluida la ventana) y ESE es el que se despliega: el objetivo
    del reentrenamiento es aprender el régimen más reciente.

El entrenamiento en sí está en gemelo.modelos.receta (el retador usa la receta de su campeón).
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Ventana:
    ini: pd.Timestamp   # primer día base evaluado
    fin: pd.Timestamp   # último día base evaluado (su objetivo, fin + h, es el último dato conocido)


def ventana_evaluacion(ultimo_dato: pd.Timestamp, h: int, dias: int) -> Ventana:
    fin = pd.Timestamp(ultimo_dato) - pd.Timedelta(days=h)
    return Ventana(ini=fin - pd.Timedelta(days=dias - 1), fin=fin)


def mascara_entrenamiento(indice: pd.DatetimeIndex, hasta_objetivo: pd.Timestamp, h: int) -> np.ndarray:
    """Filas cuyo OBJETIVO (t + h) es anterior a `hasta_objetivo` (purga de frontera)."""
    return np.asarray(indice + pd.Timedelta(days=h) < pd.Timestamp(hasta_objetivo))


def decidir(rmse_campeon: float, rmse_retador: float, mejora_minima: float) -> tuple[bool, float]:
    mejora = 1 - rmse_retador / rmse_campeon
    return bool(mejora >= mejora_minima), float(mejora)
