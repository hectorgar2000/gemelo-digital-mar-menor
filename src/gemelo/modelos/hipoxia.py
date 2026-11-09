"""Riesgo de hipoxia en la próxima campaña (≈7 días): persistencia vs. logística vs. XGBoost.

Pocos datos (359 pares, 9 episodios) -> validación dejando un año fuera (LOYO), modelos sencillos
y regularizados, sin búsqueda de hiperparámetros (evita sobreajustar la propia validación).
Limitación: LOYO entrena con años posteriores al evaluado; con 8 años no hay alternativa razonable.
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from gemelo.modelos.comun import columnas_x

# Variables con sentido físico para la logística (interpretables una a una).
# v2 (Fase 5): se quita `viento_cuenca_media7`. En v1 tenía coeficiente POSITIVO (más viento -> más
# riesgo), lo contrario de la física (el viento mezcla y oxigena el fondo); probablemente la correlación
# estacional de las brisas de verano. El motor de escenarios lo destapó: "viento ×0,5" BAJABA el riesgo.
# Sin ella, las métricas por años no empeoran (recall 0,80 = ; precisión 0,46 -> 0,48).
VARS_LOGISTICA = [
    "ahora_o2_fondo_media", "ahora_pct_hipoxia", "ahora_temp_fondo_media", "ahora_estrat_sal",
    "ahora_estrat_temp", "ahora_turb_fondo_media", "ahora_chl_fondo_media", "tend_o2_fondo_media",
    "prec_cuenca_suma30", "prec_cuenca_suma90", "dias_hasta_siguiente",
]
PARAMS_XGB = dict(max_depth=2, n_estimators=200, learning_rate=0.05, min_child_weight=3, subsample=0.8,
                  colsample_bytree=0.8, reg_lambda=5.0, random_state=42, n_jobs=-1, eval_metric="logloss")


def nueva_logistica():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(C=0.3, class_weight="balanced", max_iter=2000))


def nuevo_xgb(y_train):
    pos = max(int(y_train.sum()), 1)
    return xgb.XGBClassifier(**PARAMS_XGB, scale_pos_weight=(len(y_train) - pos) / pos)


def preparar(df: pd.DataFrame, umbral_pct: float):
    d = df[df["y_hipoxia"].notna()].sort_index()
    return d, d["y_hipoxia"].astype(int), (d["ahora_pct_hipoxia"] > umbral_pct).astype(float)


def probabilidades_loyo(d: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Probabilidades fuera de muestra: cada año se predice con modelos entrenados sin ese año."""
    X_all = d[columnas_x(d)]
    out = pd.DataFrame(index=d.index, columns=["logistica", "xgboost"], dtype=float)
    for anio in sorted(d["anio"].unique()):
        te = d["anio"] == anio
        if y[~te].sum() == 0:
            continue
        log = nueva_logistica().fit(d.loc[~te, VARS_LOGISTICA], y[~te])
        out.loc[te, "logistica"] = log.predict_proba(d.loc[te, VARS_LOGISTICA])[:, 1]
        xg = nuevo_xgb(y[~te]).fit(X_all[~te], y[~te])
        out.loc[te, "xgboost"] = xg.predict_proba(X_all[te])[:, 1]
    return out


def episodios(y: pd.Series) -> pd.Series:
    """Id de episodio para cada fila positiva (rachas de campañas consecutivas con y=1), NaN si y=0."""
    inicio = (y == 1) & (y.shift(fill_value=0) != 1)
    return inicio.cumsum().where(y == 1)


def metricas(y: pd.Series, prob: pd.Series, umbral: float = 0.5) -> dict:
    pred = (prob >= umbral).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    ep = episodios(y)
    inicios = ep.notna() & ep.ne(ep.shift())
    return {
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "falsas_alarmas": fp,
        "pr_auc": float(average_precision_score(y, prob)) if prob.nunique() > 2 else float("nan"),
        "brier": float(brier_score_loss(y, prob.clip(0, 1))),
        "episodios_detectados": int(pred[ep.notna()].groupby(ep[ep.notna()]).max().sum()),
        "episodios_total": int(ep.nunique()),
        "inicios_anticipados": int(pred[inicios].sum()),
    }


def modelos_finales(d: pd.DataFrame, y: pd.Series):
    """Modelos entrenados con todos los datos (para explicar con coeficientes y SHAP)."""
    log = nueva_logistica().fit(d[VARS_LOGISTICA], y)
    coefs = pd.Series(log[-1].coef_[0], index=VARS_LOGISTICA).sort_values(key=np.abs, ascending=False)
    xg = nuevo_xgb(y).fit(d[columnas_x(d)], y)
    return log, coefs, xg
