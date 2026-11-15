"""KPIs de acierto/fallo de los modelos de SST para el informe de proyecto.

Cada modelo se entrena con SU receta de producción pero solo con datos hasta 2021, y se evalúa fuera de muestra
(igual que la confirmación de scripts/promocionar_receta.py): h <= 7 en ago-2024 -> sep-2026 con las PREVISIONES
REALES; h = 30 en 2022 -> sep-2026. No cambia nada del registro: solo lee y escribe un CSV.

Salida: docs/figuras/informe/kpis_sst.csv

Uso:
    python scripts/kpis_informe.py
"""
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from promocionar_receta import campeon, variables

from gemelo.config import RAIZ, cargar_config
from gemelo.modelos import receta as RC
from gemelo.modelos.comun import configurar_mlflow, mae, rmse
from gemelo.procesado.features import _doy, umbral_p90

PROCESSED = RAIZ / "data" / "processed"
SALIDA = RAIZ / "docs" / "figuras" / "informe"


def clasificacion(pred: np.ndarray, real: np.ndarray) -> dict:
    """Matriz de confusión de 'habrá ola de calor el día objetivo' y sus métricas."""
    tp, fp = int((pred & real).sum()), int((pred & ~real).sum())
    fn, tn = int((~pred & real).sum()), int((~pred & ~real).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "exactitud": (tp + tn) / len(real), "precision": prec,
            "recall": rec, "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0}


if __name__ == "__main__":
    cfg = cargar_config()
    ds = pd.read_parquet(PROCESSED / "dataset_sst.parquet")
    diario = pd.read_parquet(PROCESSED / "series_diarias.parquet")
    entreno = ds.join(pd.read_parquet(PROCESSED / "meteo_futura_era5.parquet"))
    servicio = ds.join(pd.read_parquet(PROCESSED / "meteo_futura_prevision.parquet"))
    fin_entreno, ultimo = pd.Timestamp("2021-12-31"), pd.Timestamp(cfg["periodo"]["fin"])
    configurar_mlflow("mejora_receta")
    client = MlflowClient()
    filas = []
    for zona in ["laguna", "costa"]:
        umbral_doy = umbral_p90(diario[f"sst_{zona}"], *cfg["umbrales"]["climatologia"],
                                pct=cfg["umbrales"]["mhw_percentil"])
        for h in cfg["horizontes_dias"]:
            nombre = f"sst_{zona}_h{h}"
            mv, params, n_arboles = campeon(client, nombre)
            receta = mv.tags.get("receta", "clasica")
            y = ds[f"y_delta_{zona}_h{h}"]
            objetivo = ds.index + pd.Timedelta(days=h)
            ini_eval = pd.Timestamp(cfg["prevision"]["evaluacion"][0] if h <= 7 else "2022-01-01")
            cols = variables(entreno, h, receta == "fisica_pesos")
            m = RC.entrenar(entreno.loc[objetivo <= fin_entreno, cols], y[objetivo <= fin_entreno], params, n_arboles,
                            receta, h)
            ev = y.notna() & servicio[cols].notna().all(axis=1) & (ds.index >= ini_eval) & (objetivo <= ultimo)
            ev &= ds[f"y_sst_{zona}_h{h}"].notna() & ds[f"y_mhw_{zona}_h{h}"].notna()
            real_delta = y[ev].values
            pred = m.predict(servicio.loc[ev, cols])
            error = pred - real_delta  # el error del delta ES el error de la SST (clim y anomalía de hoy son conocidas)
            sst_real = ds.loc[ev, f"y_sst_{zona}_h{h}"].values
            umbral = umbral_doy.reindex(_doy(ds.index[ev] + pd.Timedelta(days=h))).values
            ola_real = ds.loc[ev, f"y_mhw_{zona}_h{h}"].astype(bool).values
            fila = {"modelo": nombre, "version": f"v{mv.version}", "receta": receta, "n_dias": int(ev.sum()),
                    "desde": str(ds.index[ev][0].date()), "hasta": str(ds.index[ev][-1].date()),
                    "pct_dias_ola": float(ola_real.mean())}
            for etiqueta, e in [("modelo", error), ("persistencia", -real_delta)]:
                c = clasificacion((sst_real + e) > umbral, ola_real)
                fila |= {f"rmse_{etiqueta}": rmse(e, 0 * e), f"mae_{etiqueta}": mae(e, 0 * e),
                         f"sesgo_{etiqueta}": float(e.mean()),
                         f"acierto_05_{etiqueta}": float((np.abs(e) <= 0.5).mean()),
                         f"acierto_10_{etiqueta}": float((np.abs(e) <= 1.0).mean()),
                         f"fallo_15_{etiqueta}": float((np.abs(e) > 1.5).mean()),
                         f"p95_abs_{etiqueta}": float(np.quantile(np.abs(e), 0.95)),
                         **{f"ola_{k}_{etiqueta}": v for k, v in c.items()}}
            filas.append(fila)
            print(f"{nombre:<15} {receta:<13} n={fila['n_dias']}  RMSE {fila['rmse_modelo']:.3f} (pers. "
                  f"{fila['rmse_persistencia']:.3f})  ±0,5 °C {fila['acierto_05_modelo']:.0%}  ±1 °C "
                  f"{fila['acierto_10_modelo']:.0%}  ola: exactitud {fila['ola_exactitud_modelo']:.0%} "
                  f"F1 {fila['ola_f1_modelo']:.2f} (pers. {fila['ola_f1_persistencia']:.2f})", flush=True)
    SALIDA.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(filas).round(4).to_csv(SALIDA / "kpis_sst.csv", index=False)
    print(f"\n-> {SALIDA / 'kpis_sst.csv'}")
