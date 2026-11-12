"""Banco de pruebas para mejorar los modelos de SST sin tocar el periodo de test.

Protocolo (desarrollo): origen móvil con 8 pliegues anuales (2014..2021). En cada pliegue se entrena con todo lo
anterior (purga de frontera) y se predice el año. Los errores de los 8 años se juntan y cada variante se
compara con la configuración actual ("base") con un IC del 90 % por bootstrap de bloques de 30 días.
Para h = 1, 3, 7 las variables de previsión son las de ERA5 ("previsión perfecta"), como en el entrenamiento real.

La confirmación con previsiones REALES (2024-2026) y la promoción se hacen aparte: scripts/promocionar_receta.py.
Las variantes "nombre+fis" añaden las variables de intercambio aire-mar (features.es_fisica).

Uso:
    python scripts/experimentos_mejora.py                 # todas las variantes
    python scripts/experimentos_mejora.py base lento      # solo algunas
"""
import sys
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from gemelo.config import RAIZ, cargar_config
from gemelo.modelos.comun import columnas_x
from gemelo.modelos.sst import PARAMS_FIJOS
from gemelo.procesado.features import es_fisica

PROCESSED = RAIZ / "data" / "processed"
SALIDA = RAIZ / "docs" / "figuras" / "mejora"
PLIEGUES = list(range(2014, 2022))
# hiperparámetros actuales en producción (del registro de MLflow, 2026-11-12)
ACTUAL = {("laguna", 1): (5, 20, 82), ("laguna", 3): (5, 20, 66), ("laguna", 7): (3, 5, 153), ("laguna", 30): (5, 20, 36),
          ("costa", 1): (5, 20, 122), ("costa", 3): (5, 20, 160), ("costa", 7): (3, 20, 225), ("costa", 30): (3, 5, 48)}


def xgb_actual(zona, h, **extra):
    d, mcw, n = ACTUAL[(zona, h)]
    p = {**PARAMS_FIJOS, "max_depth": d, "min_child_weight": mcw, "n_estimators": n}
    p.update(extra)
    return xgb.XGBRegressor(**p)


def pesos_recientes(indice, semivida_anios):
    edad = (indice.max() - indice).days / 365.25
    return np.power(0.5, edad / semivida_anios)


# ---------------------------------------------------------------- variantes: f(Xtr, ytr, Xte, zona, h) -> pred
def v_base(Xtr, ytr, Xte, zona, h):
    return xgb_actual(zona, h).fit(Xtr, ytr).predict(Xte)


def v_lento(Xtr, ytr, Xte, zona, h):
    """Mismo modelo, aprendizaje 2,5 veces más lento con 2,5 veces más árboles (menos ruido)."""
    n = ACTUAL[(zona, h)][2]
    return xgb_actual(zona, h, learning_rate=0.02, n_estimators=int(n * 2.5)).fit(Xtr, ytr).predict(Xte)


def v_regular(Xtr, ytr, Xte, zona, h):
    """Configuración única, más árboles y más regularización (sin parada temprana ruidosa)."""
    m = xgb.XGBRegressor(**{**PARAMS_FIJOS, "max_depth": 4, "min_child_weight": 20, "n_estimators": 400,
                            "learning_rate": 0.02, "reg_lambda": 5.0, "colsample_bytree": 0.6})
    return m.fit(Xtr, ytr).predict(Xte)


def v_pesos(Xtr, ytr, Xte, zona, h):
    """Más peso a los años recientes (semivida de 8 años): el mar cambia."""
    return xgb_actual(zona, h).fit(Xtr, ytr, sample_weight=pesos_recientes(Xtr.index, 8)).predict(Xte)


def v_huber(Xtr, ytr, Xte, zona, h):
    return xgb_actual(zona, h, objective="reg:pseudohubererror", huber_slope=1.0).fit(Xtr, ytr).predict(Xte)


def _ridge(alpha=30.0):
    return make_pipeline(StandardScaler(), Ridge(alpha=alpha))


def v_ridge(Xtr, ytr, Xte, zona, h):
    """Solo lineal: referencia (¿cuánto aporta la no linealidad de los árboles?)."""
    return _ridge().fit(Xtr, ytr).predict(Xte)


def v_hibrido(Xtr, ytr, Xte, zona, h):
    """Lineal + árboles sobre el residuo: la parte lineal SÍ extrapola fuera del rango visto."""
    lin = _ridge().fit(Xtr, ytr)
    arb = xgb_actual(zona, h).fit(Xtr, ytr - lin.predict(Xtr))
    return lin.predict(Xte) + arb.predict(Xte)


def v_hibrido_regular(Xtr, ytr, Xte, zona, h):
    lin = _ridge().fit(Xtr, ytr)
    arb = xgb.XGBRegressor(**{**PARAMS_FIJOS, "max_depth": 4, "min_child_weight": 20, "n_estimators": 400,
                              "learning_rate": 0.02, "reg_lambda": 5.0, "colsample_bytree": 0.6})
    arb.fit(Xtr, ytr - lin.predict(Xtr))
    return lin.predict(Xte) + arb.predict(Xte)


def v_media(Xtr, ytr, Xte, zona, h):
    """Media de árboles y lineal: dos modelos con errores distintos suelen compensarse."""
    return 0.5 * v_base(Xtr, ytr, Xte, zona, h) + 0.5 * v_ridge(Xtr, ytr, Xte, zona, h)


def v_lento_pesos(Xtr, ytr, Xte, zona, h):
    n = ACTUAL[(zona, h)][2]
    m = xgb_actual(zona, h, learning_rate=0.02, n_estimators=int(n * 2.5))
    return m.fit(Xtr, ytr, sample_weight=pesos_recientes(Xtr.index, 8)).predict(Xte)


VARIANTES = {"base": v_base, "lento": v_lento, "regular": v_regular, "pesos": v_pesos, "huber": v_huber,
             "ridge": v_ridge, "hibrido": v_hibrido, "hibrido_regular": v_hibrido_regular, "media": v_media,
             "lento_pesos": v_lento_pesos}


# ---------------------------------------------------------------- protocolo
def datos():
    ds = pd.read_parquet(PROCESSED / "dataset_sst.parquet")
    return ds.join(pd.read_parquet(PROCESSED / "meteo_futura_era5.parquet"))


def variables(ds, h, fisica=False):
    """Variables del modelo; las de intercambio aire-mar (es_fisica) solo si fisica=True (variantes '+fis')."""
    base = columnas_x(ds[[c for c in ds.columns if not c.startswith("prev")]])
    w = min(h, 7)
    cols = base + ([c for c in ds.columns if c.startswith(f"prev{w}_")] if h <= 7 else [])
    return [c for c in cols if fisica or not es_fisica(c)]


def origen_movil(ds, zona, h, variante, fisica=False, pliegues=PLIEGUES):
    """Predicciones fuera de muestra de una variante: Serie indexada por día base."""
    cols = variables(ds, h, fisica)
    y = ds[f"y_delta_{zona}_h{h}"]
    ok = y.notna() & ds[cols].notna().all(axis=1)
    X, y = ds.loc[ok, cols], y[ok]
    partes = []
    for anio in pliegues:
        ini, fin = pd.Timestamp(f"{anio}-01-01"), pd.Timestamp(f"{anio}-12-31")
        tr = X.index + pd.Timedelta(days=h) < ini                       # purga de frontera
        te = (X.index >= ini) & (X.index + pd.Timedelta(days=h) <= fin)
        partes.append(pd.Series(variante(X[tr], y[tr], X[te], zona, h), index=X.index[te]))
    return pd.concat(partes), y


def mejora_bootstrap(e_ref, e_mod, bloque=30, n=1000, semilla=0):
    rng = np.random.default_rng(semilla)
    inicios = np.arange(0, len(e_ref) - bloque + 1)
    nb = int(np.ceil(len(e_ref) / bloque))
    sk = []
    for _ in range(n):
        idx = (rng.choice(inicios, nb)[:, None] + np.arange(bloque)).ravel()
        sk.append(1 - np.sqrt(np.mean(e_mod[idx] ** 2)) / np.sqrt(np.mean(e_ref[idx] ** 2)))
    return np.quantile(sk, [0.05, 0.95])


if __name__ == "__main__":
    cfg = cargar_config()
    nombres = sys.argv[1:] or list(VARIANTES)
    if "base" not in nombres:
        nombres = ["base"] + nombres
    ds = datos().loc[:"2021-12-31"]          # desarrollo: NADA de 2022 en adelante
    SALIDA.mkdir(parents=True, exist_ok=True)
    filas, errores = [], {}
    for zona in ["laguna", "costa"]:
        for h in cfg["horizontes_dias"]:
            for nombre in nombres:
                t = time.time()
                funcion, _, fis = nombre.partition("+")
                pred, y = origen_movil(ds, zona, h, VARIANTES[funcion], fisica=bool(fis))   # "+fis": con variables físicas
                e = (y.loc[pred.index] - pred).values
                errores[(zona, h, nombre)] = e
                anom = ds.loc[pred.index, f"anom_{zona}_lag0"].abs().values
                extremo = anom > np.quantile(anom, 0.9)                  # días con anomalía muy alta
                fila = {"zona": zona, "h": h, "variante": nombre, "rmse": np.sqrt(np.mean(e ** 2)),
                        "rmse_extremos": np.sqrt(np.mean(e[extremo] ** 2)), "seg": round(time.time() - t, 1)}
                if nombre != "base":
                    eb = errores[(zona, h, "base")]
                    fila["mejora"] = 1 - fila["rmse"] / np.sqrt(np.mean(eb ** 2))
                    fila["ic05"], fila["ic95"] = mejora_bootstrap(eb, e)
                    fila["mejora_extremos"] = 1 - fila["rmse_extremos"] / np.sqrt(np.mean(eb[extremo] ** 2))
                filas.append(fila)
                print(f"{zona} h={h:>2} {nombre:<16} RMSE {fila['rmse']:.4f}"
                      + (f"  mejora {fila['mejora']:+.1%} [{fila['ic05']:+.1%}, {fila['ic95']:+.1%}]"
                         f"  extremos {fila['mejora_extremos']:+.1%}" if nombre != "base" else ""), flush=True)
    t = pd.DataFrame(filas)
    t.to_csv(SALIDA / f"experimentos_{'_'.join(nombres)[:60]}.csv", index=False)
    if len(nombres) > 1:
        r = t[t.variante != "base"].pivot_table(index="variante", values=["mejora", "mejora_extremos"], aggfunc="mean")
        gana = t[t.variante != "base"].assign(sig=lambda d: d.ic05 > 0).groupby("variante").sig.sum()
        print("\nMedia de los 8 modelos (mejora de RMSE frente a la configuración actual):")
        print(r.assign(modelos_con_mejora_significativa=gana).sort_values("mejora", ascending=False).round(4).to_string())
