"""Predicción de la anomalía de SST a h días con XGBoost y comparación con líneas base.

Objetivo: y_delta = anom(t+h) - anom(t)  (cambio respecto a hoy).
  - Persistencia      -> delta = 0
  - Climatología      -> delta = -anom(t)            (la anomalía desaparece)
  - Pers. amortiguada -> delta = (rho_h - 1)*anom(t) (la anomalía se va apagando)
Como SST(t+h) = clim(t+h) + anom(t) + delta, el error en delta es exactamente el error en SST.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import xgboost as xgb

from gemelo.modelos.comun import columnas_x, mae, rmse, skill
from gemelo.procesado.features import filas_de_particion

PARAMS_FIJOS = dict(objective="reg:squarederror", tree_method="hist", subsample=0.8,
                    colsample_bytree=0.8, learning_rate=0.05, random_state=42, n_jobs=-1)
REJILLA = [dict(max_depth=d, min_child_weight=m) for d in (3, 5) for m in (5, 20)]
# Validación de origen móvil dentro de train: se entrena con todo lo anterior y se valida en 2 años
PLIEGUES = [("2011-01-01", "2012-12-31"), ("2013-01-01", "2014-12-31"),
            ("2015-01-01", "2016-12-31"), ("2017-01-01", "2018-12-31")]
MAX_ARBOLES = 2000
PARADA_TEMPRANA = 50


@dataclass
class Resultado:
    zona: str
    h: int
    params: dict
    n_arboles: int
    metricas: dict = field(default_factory=dict)
    cuantiles_residuo: dict = field(default_factory=dict)
    modelo_final: xgb.XGBRegressor | None = None
    pred_test: pd.DataFrame | None = None


def _xy(df, zona, h):
    y = df[f"y_delta_{zona}_h{h}"]
    ok = y.notna()
    return df.loc[ok, columnas_x(df)], y[ok], df.loc[ok, f"anom_{zona}_lag0"]


def _rango(df, ini, fin, h):
    """Filas con fecha >= ini y objetivo (t+h) <= fin: purga de frontera."""
    ini, fin = pd.Timestamp(ini), pd.Timestamp(fin)
    return df[(df.index >= ini) & (df.index + pd.Timedelta(days=h) <= fin)]


def rho_amortiguada(anom_hoy: pd.Series, delta: pd.Series) -> float:
    """Coeficiente que minimiza el error de anom(t+h) ≈ rho * anom(t) (mínimos cuadrados sin constante)."""
    futura = anom_hoy + delta
    return float((anom_hoy * futura).sum() / (anom_hoy ** 2).sum())


def predicciones_base(anom_hoy: pd.Series, rho: float) -> dict:
    return {"persistencia": np.zeros(len(anom_hoy)), "climatologia": -anom_hoy.values,
            "amortiguada": (rho - 1) * anom_hoy.values}


def validacion_origen_movil(ds, zona, h):
    """Elige hiperparámetros y nº de árboles con los pliegues de origen móvil (solo datos de train)."""
    inicio = ds.index.min()
    resultados = []
    for params in REJILLA:
        errores, iters = [], []
        for ini_val, fin_val in PLIEGUES:
            tr = _rango(ds, inicio, pd.Timestamp(ini_val) - pd.Timedelta(days=1), h)
            va = _rango(ds, ini_val, fin_val, h)
            Xtr, ytr, _ = _xy(tr, zona, h)
            Xva, yva, _ = _xy(va, zona, h)
            m = xgb.XGBRegressor(**PARAMS_FIJOS, **params, n_estimators=MAX_ARBOLES,
                                 early_stopping_rounds=PARADA_TEMPRANA)
            m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
            errores.append(rmse(yva, m.predict(Xva)))
            iters.append(m.best_iteration + 1)
        resultados.append((float(np.mean(errores)), params, int(np.median(iters)), errores))
    resultados.sort(key=lambda r: r[0])
    return resultados


def entrenar_zona_horizonte(ds: pd.DataFrame, part: dict, zona: str, h: int) -> Resultado:
    cv = validacion_origen_movil(ds, zona, h)
    rmse_cv, params, n_arboles, _ = cv[0]
    res = Resultado(zona, h, params, n_arboles)
    res.metricas["cv_rmse"] = rmse_cv

    # 1) Entrenar en train -> evaluar en validación (y calibrar el intervalo con sus residuos)
    Xtr, ytr, atr = _xy(filas_de_particion(ds, "train", h, part), zona, h)
    Xva, yva, ava = _xy(filas_de_particion(ds, "validacion", h, part), zona, h)
    rho = rho_amortiguada(atr, ytr)
    m = xgb.XGBRegressor(**PARAMS_FIJOS, **params, n_estimators=n_arboles).fit(Xtr, ytr, verbose=False)
    p_va = m.predict(Xva)
    _evaluar(res, "val", yva, p_va, predicciones_base(ava, rho))
    residuo = yva.values - p_va
    res.cuantiles_residuo = {q: float(np.quantile(residuo, q)) for q in (0.05, 0.10, 0.50, 0.90, 0.95)}

    # 2) Reentrenar con train + validación -> evaluar UNA vez en test
    trva = _rango(ds, part["train"][0], part["validacion"][1], h)
    Xtv, ytv, atv = _xy(trva, zona, h)
    Xte, yte, ate = _xy(filas_de_particion(ds, "test", h, part), zona, h)
    rho_tv = rho_amortiguada(atv, ytv)
    m_final = xgb.XGBRegressor(**PARAMS_FIJOS, **params, n_estimators=n_arboles).fit(Xtv, ytv, verbose=False)
    p_te = m_final.predict(Xte)
    bases_te = predicciones_base(ate, rho_tv)
    _evaluar(res, "test", yte, p_te, bases_te)
    lo, hi = p_te + res.cuantiles_residuo[0.05], p_te + res.cuantiles_residuo[0.95]
    res.metricas["test_cobertura_90"] = float(np.mean((yte.values >= lo) & (yte.values <= hi)))
    res.metricas["rho"] = rho_tv

    res.modelo_final = m_final
    res.pred_test = pd.DataFrame({"y_delta": yte, "pred": p_te, "amortiguada": bases_te["amortiguada"],
                                  "lo90": lo, "hi90": hi}, index=yte.index)
    return res


def _evaluar(res, pref, y, p, bases):
    res.metricas[f"{pref}_rmse"] = rmse(y, p)
    res.metricas[f"{pref}_mae"] = mae(y, p)
    for nombre, pb in bases.items():
        res.metricas[f"{pref}_rmse_{nombre}"] = rmse(y, pb)
    res.metricas[f"{pref}_skill_vs_amortiguada"] = skill(rmse(y, p), rmse(y, bases["amortiguada"]))
    res.metricas[f"{pref}_skill_vs_climatologia"] = skill(rmse(y, p), rmse(y, bases["climatologia"]))


def evaluar_mhw(pred_test: pd.DataFrame, ds: pd.DataFrame, umbral_doy: pd.Series, zona: str, h: int) -> dict:
    """Olas de calor derivadas de la regresión: SST predicha > p90 del día objetivo.

    SST_pred = SST_real(t+h) + (delta_pred - delta_real). Se compara con el objetivo y_mhw (que además
    exige rachas de >= 5 días), así que es una evaluación conservadora."""
    from gemelo.procesado.features import _doy

    d = pred_test.join(ds[[f"y_sst_{zona}_h{h}", f"y_mhw_{zona}_h{h}"]]).dropna()
    fecha_obj = d.index + pd.Timedelta(days=h)
    umbral = umbral_doy.reindex(_doy(fecha_obj)).values
    real = d[f"y_mhw_{zona}_h{h}"].astype(bool).values
    out = {}
    for nombre, col in [("modelo", "pred"), ("amortiguada", "amortiguada")]:
        sst_pred = d[f"y_sst_{zona}_h{h}"].values + (d[col].values - d["y_delta"].values)
        pred = sst_pred > umbral
        tp, fp, fn = (pred & real).sum(), (pred & ~real).sum(), (~pred & real).sum()
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        out[f"mhw_{nombre}_precision"] = float(prec)
        out[f"mhw_{nombre}_recall"] = float(rec)
        out[f"mhw_{nombre}_f1"] = float(2 * prec * rec / (prec + rec)) if prec + rec else 0.0
    return out
