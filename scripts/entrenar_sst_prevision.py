"""Fase 7: ¿mejora la predicción de SST si el modelo conoce la previsión meteorológica?

Para cada zona y horizonte se comparan, en la ventana de evaluación honesta (ago-2024 → sep-2026):
  A          modelo actual (sin meteorología futura)
  B_real     modelo con meteo futura, alimentado con las PREVISIONES REALES emitidas ese día (honesto)
  B_perfecta el mismo modelo con la meteorología que realmente ocurrió (ERA5): cota superior
  + persistencia y persistencia amortiguada.
B se entrena con ERA5 como "previsión perfecta" (1998-2021): no hay previsiones archivadas antes de 2024.
La mejora de B_real sobre A lleva un intervalo de confianza por bootstrap de bloques de 30 días.

Uso:
    python scripts/entrenar_sst_prevision.py
"""
import json
import time

import matplotlib
matplotlib.use("Agg")
import mlflow
import numpy as np
import pandas as pd

from gemelo.config import DATA_RAW, RAIZ, cargar_config
from gemelo.modelos import xai
from gemelo.modelos.comun import MANIFIESTO, configurar_mlflow, rmse
from gemelo.modelos.sst import entrenar_zona_horizonte, evaluar_mhw
from gemelo.procesado import meteo_futura as M
from gemelo.procesado.features import ZONAS, filas_de_particion, umbral_p90

PROCESSED = RAIZ / "data" / "processed"
SALIDA = RAIZ / "docs" / "figuras" / "fase7"


def paso(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def skill_bootstrap(err_ref, err_mod, bloque=30, n=2000, semilla=0):
    """IC 90 % de 1 - RMSE_mod/RMSE_ref remuestreando bloques de días consecutivos (errores autocorrelados)."""
    rng = np.random.default_rng(semilla)
    n_bloques = int(np.ceil(len(err_ref) / bloque))
    inicios = np.arange(0, len(err_ref) - bloque + 1)
    sk = []
    for _ in range(n):
        idx = np.concatenate([np.arange(i, i + bloque) for i in rng.choice(inicios, n_bloques)])
        sk.append(1 - np.sqrt(np.mean(err_mod[idx] ** 2)) / np.sqrt(np.mean(err_ref[idx] ** 2)))
    return np.quantile(sk, [0.05, 0.95])


def construir_tablas(cfg, ds, diario, ref):
    puntos = list(cfg["meteo"]["puntos"])
    f_era5 = M.construir(M.fuente_era5(diario), puntos, ref, ds.index)
    p = cfg["prevision"]
    prev, sesgos = {}, {}
    for pt in puntos:
        bruto = M.prevision_por_antelacion(pd.read_parquet(DATA_RAW / "meteo" / f"prevision_{pt}.parquet"), p["dias_max"])
        sesgos[pt] = M.sesgo_prevision(bruto, diario, pt, *p["calibracion"])
        prev[pt] = M.corregir(bruto, sesgos[pt])
    f_prev = M.construir(M.fuente_prevision(prev), puntos, ref, ds.index)
    f_era5.to_parquet(PROCESSED / "meteo_futura_era5.parquet")
    f_prev.to_parquet(PROCESSED / "meteo_futura_prevision.parquet")
    return f_era5, f_prev, sesgos


if __name__ == "__main__":
    cfg = cargar_config()
    part, ev = cfg["particion"], cfg["prevision"]["evaluacion"]
    ds = pd.read_parquet(PROCESSED / "dataset_sst.parquet")
    diario = pd.read_parquet(PROCESSED / "series_diarias.parquet")
    ref = pd.read_parquet(PROCESSED / "referencias_sst.parquet")
    SALIDA.mkdir(parents=True, exist_ok=True)

    paso("tablas de meteorología futura (ERA5 y previsión real corregida)")
    f_era5, f_prev, sesgos = construir_tablas(cfg, ds, diario, ref)
    for pt, s in sesgos.items():
        print(f"   sesgo previsión-ERA5 {pt} (antelación 1 / 7): " +
              ", ".join(f"{v} {s.loc[1, v]:+.2f}/{s.loc[7, v]:+.2f}" for v in ["t2m", "viento", "rad"]))

    configurar_mlflow("sst_prevision")
    filas = []
    with mlflow.start_run(run_name="sst_con_prevision") as padre:
        mlflow.log_artifact(str(MANIFIESTO))
        mlflow.log_dict({pt: s.round(4).to_dict() for pt, s in sesgos.items()}, "sesgos_prevision.json")
        for zona in ZONAS:
            umbral = umbral_p90(diario[f"sst_{zona}"], *cfg["umbrales"]["climatologia"], pct=cfg["umbrales"]["mhw_percentil"])
            for h in cfg["horizontes_dias"]:
                w = min(h, cfg["prevision"]["dias_max"])
                cols = [c for c in f_era5.columns if c.startswith(f"prev{w}_")]
                paso(f"{zona} h={h} (ventana meteo {w} d): entrenando A y B")
                rA = entrenar_zona_horizonte(ds, part, zona, h)
                rB = entrenar_zona_horizonte(ds.join(f_era5[cols]), part, zona, h)

                # filas de evaluación: test, dentro de la ventana honesta y con previsión real completa
                t = filas_de_particion(ds, "test", h, part)
                t = t[(t.index >= ev[0]) & (t.index + pd.Timedelta(days=h) <= pd.Timestamp(ev[1]))]
                vars_a = rA.modelo_final.get_booster().feature_names
                vars_b = rB.modelo_final.get_booster().feature_names
                x_real = t.join(f_prev[cols])[vars_b]
                x_perf = t.join(f_era5[cols])[vars_b]
                y = t[f"y_delta_{zona}_h{h}"]
                ok = y.notna() & t[vars_a].notna().all(axis=1) & x_real.notna().all(axis=1)
                y, idx = y[ok], y[ok].index
                pred = {
                    "persistencia": np.zeros(len(y)),
                    "amortiguada": (rA.metricas["rho"] - 1) * t.loc[idx, f"anom_{zona}_lag0"].values,
                    "A": rA.modelo_final.predict(t.loc[idx, vars_a]),
                    "B_real": rB.modelo_final.predict(x_real.loc[idx]),
                    "B_perfecta": rB.modelo_final.predict(x_perf.loc[idx]),
                }
                e = {k: y.values - v for k, v in pred.items()}
                lo, hi = skill_bootstrap(e["A"], e["B_real"])
                fila = {"zona": zona, "h": h, "n_dias": len(y), **{f"rmse_{k}": rmse(y, v) for k, v in pred.items()}}
                fila["skill_Breal_vs_A"] = 1 - fila["rmse_B_real"] / fila["rmse_A"]
                fila["skill_Breal_vs_A_ic90"] = [float(lo), float(hi)]
                fila["skill_Bperf_vs_A"] = 1 - fila["rmse_B_perfecta"] / fila["rmse_A"]
                for nombre in ["A", "B_real"]:
                    pt_ = pd.DataFrame({"y_delta": y, "pred": pred[nombre], "amortiguada": pred["amortiguada"]}, index=idx)
                    fila[f"f1_mhw_{nombre}"] = evaluar_mhw(pt_, ds, umbral, zona, h)["mhw_modelo_f1"]
                sv = xai.valores_shap(rB.modelo_final, x_real.loc[idx])
                imp = xai.importancia_por_grupo(sv, vars_b)
                fila["pct_shap_meteo_prevista"] = float(100 * imp.get("meteo_prevista", 0) / imp.sum())
                filas.append(fila)

                with mlflow.start_run(run_name=f"{zona}_h{h}", nested=True):
                    mlflow.log_params({"zona": zona, "h": h, "ventana_meteo": w, "n_arboles": rB.n_arboles, **rB.params})
                    mlflow.log_metrics({k: v for k, v in fila.items() if isinstance(v, float)})
                    mlflow.log_dict(rB.cuantiles_residuo, "cuantiles_residuo_validacion.json")
                    mlflow.xgboost.log_model(rB.modelo_final, name="modelo", input_example=x_real.loc[idx].head(3))
                paso(f"   RMSE A {fila['rmse_A']:.3f} · B_real {fila['rmse_B_real']:.3f} · B_perfecta {fila['rmse_B_perfecta']:.3f}"
                     f" · skill B_real vs A {fila['skill_Breal_vs_A']:+.1%} [IC90 {lo:+.1%}, {hi:+.1%}]"
                     f" · F1 ola A {fila['f1_mhw_A']:.2f} → B {fila['f1_mhw_B_real']:.2f}")
        tabla = pd.DataFrame(filas)
        tabla.to_csv(SALIDA / "comparacion_prevision.csv", index=False)
        mlflow.log_artifact(str(SALIDA / "comparacion_prevision.csv"))
        paso(f"run MLflow {padre.info.run_id}")
    print(tabla.drop(columns=["skill_Breal_vs_A_ic90"]).round(3).to_string(index=False))
