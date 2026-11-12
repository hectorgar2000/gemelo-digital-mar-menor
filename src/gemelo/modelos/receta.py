"""Recetas de entrenamiento de los modelos de SST. La misma función entrena los modelos nuevos y los
reentrenamientos (el retador usa la receta de su campeón).

  clasica       XGBoost con los hiperparámetros tal cual (Fases 3-7).
  fisica_pesos  Elegida en scripts/experimentos_mejora.py y confirmada con previsiones reales (docs/mejora_modelos.md):
                  - aprendizaje más lento (lr 0,02) con 2,5 veces más árboles: menos ruido;
                  - más peso a los años recientes (semivida de 8 años): el mar de hoy no es el de 1998;
                  - a 30 días, media 50/50 con un modelo lineal (ridge): con tan poca señal, lo simple ayuda.
                Las variables físicas (aire-agua, flujo) no son parte de la receta: van en la lista de variables.
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from gemelo.modelos.conjunto import ModeloSST
from gemelo.modelos.sst import PARAMS_FIJOS

RECETAS = ("clasica", "fisica_pesos")
LR_LENTO, FACTOR_ARBOLES = 0.02, 2.5
SEMIVIDA_ANIOS = 8
ALPHA_RIDGE = 30.0
H_CON_LINEAL, PESO_LINEAL = 30, 0.5     # horizontes >= 30 días: media con el lineal


def pesos_recientes(indice: pd.DatetimeIndex, semivida_anios: float = SEMIVIDA_ANIOS) -> np.ndarray:
    """Peso 1 para el día más reciente; la mitad cada `semivida_anios` hacia atrás."""
    edad = (indice.max() - indice).days / 365.25
    return np.power(0.5, edad / semivida_anios)


def entrenar(X: pd.DataFrame, y: pd.Series, params: dict, n_arboles: int, receta: str, h: int) -> ModeloSST:
    """`params` = {max_depth, min_child_weight}; `n_arboles` = el de la receta clásica (la nueva lo multiplica)."""
    if receta not in RECETAS:
        raise ValueError(f"Receta desconocida '{receta}'. Opciones: {RECETAS}")
    ok = y.notna() & X.notna().all(axis=1)
    X, y = X[ok], y[ok]
    if receta == "clasica":
        return ModeloSST(xgb.XGBRegressor(**PARAMS_FIJOS, **params, n_estimators=n_arboles).fit(X, y, verbose=False))
    p = {**PARAMS_FIJOS, **params, "learning_rate": LR_LENTO, "n_estimators": int(n_arboles * FACTOR_ARBOLES)}
    arboles = xgb.XGBRegressor(**p).fit(X, y, sample_weight=pesos_recientes(X.index), verbose=False)
    if h < H_CON_LINEAL:
        return ModeloSST(arboles)
    lineal = make_pipeline(StandardScaler(), Ridge(alpha=ALPHA_RIDGE)).fit(X, y)
    return ModeloSST(arboles, ModeloSST.lineal_desde_sklearn(lineal, X.columns, PESO_LINEAL))
