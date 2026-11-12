"""Confirma la receta 'fisica_pesos' frente a la actual y promociona los modelos que la superan.

Para cada modelo de SST (zona × horizonte):
  1. CONFIRMACIÓN honesta: las dos recetas se entrenan con datos hasta 2021 (previsión "perfecta" ERA5) y se
     evalúan fuera de muestra: h <= 7 en ago-2024 -> sep-2026 con las PREVISIONES REALES; h = 30 en 2022 -> sep-2026.
       actual: XGBoost con los hiperparámetros en producción, sin variables físicas (receta 'clasica')
       nueva:  mismas profundidades + variables físicas + receta 'fisica_pesos' (gemelo.modelos.receta)
  2. PROMOCIÓN (solo con --promover): si la mejora es >= reentrenamiento.mejora_minima y su IC90 por bloques > 0,
     se reajusta con TODOS los datos y se registra como nueva versión @produccion (la anterior queda @anterior).
     Los cuantiles del intervalo salen de los errores fuera de muestra de la confirmación.

Uso:
    python scripts/promocionar_receta.py               # solo confirma (no cambia nada)
    python scripts/promocionar_receta.py --promover    # confirma, promociona y exporta
    python scripts/promocionar_receta.py --residuos    # solo adjunta a cada modelo en producción sus errores fuera
                                                       # de muestra (los de su receta) y exporta; no crea versiones
"""
import subprocess
import sys
import time

import mlflow
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient

from gemelo.config import RAIZ, cargar_config
from gemelo.modelos import receta as RC
from gemelo.modelos.comun import MANIFIESTO, columnas_x, configurar_mlflow, mejora_bootstrap, rmse
from gemelo.procesado.features import es_fisica

PROCESSED = RAIZ / "data" / "processed"
SALIDA = RAIZ / "docs" / "figuras" / "mejora"
RECETA = "fisica_pesos"


def variables(tabla: pd.DataFrame, h: int, fisica: bool) -> list[str]:
    base = columnas_x(tabla[[c for c in tabla.columns if not c.startswith("prev")]])
    prev = [c for c in tabla.columns if c.startswith(f"prev{min(h, 7)}_")] if h <= 7 else []
    return [c for c in base + prev if fisica or not es_fisica(c)]


def campeon(client, nombre):
    mv = client.get_model_version_by_alias(nombre, "produccion")
    p = client.get_run(mv.run_id).data.params
    return mv, {"max_depth": int(p["max_depth"]), "min_child_weight": int(p["min_child_weight"])}, int(p["n_arboles"])


if __name__ == "__main__":
    promover_todo = "--promover" in sys.argv
    solo_residuos = "--residuos" in sys.argv
    cfg = cargar_config()
    minimo = cfg["reentrenamiento"]["mejora_minima"]
    ds = pd.read_parquet(PROCESSED / "dataset_sst.parquet")
    entreno = ds.join(pd.read_parquet(PROCESSED / "meteo_futura_era5.parquet"))
    servicio = ds.join(pd.read_parquet(PROCESSED / "meteo_futura_prevision.parquet"))
    fin_entreno, ultimo = pd.Timestamp("2021-12-31"), pd.Timestamp(cfg["periodo"]["fin"])
    configurar_mlflow("mejora_receta")
    client = MlflowClient()
    SALIDA.mkdir(parents=True, exist_ok=True)
    filas, promovidos = [], []
    with mlflow.start_run(run_name=f"receta_{RECETA}"):
        mlflow.log_artifact(str(MANIFIESTO))
        for zona in ["laguna", "costa"]:
            for h in cfg["horizontes_dias"]:
                nombre = f"sst_{zona}_h{h}"
                mv, params, n_arboles = campeon(client, nombre)
                y = ds[f"y_delta_{zona}_h{h}"]
                objetivo = ds.index + pd.Timedelta(days=h)
                ini_eval = pd.Timestamp(cfg["prevision"]["evaluacion"][0] if h <= 7 else "2022-01-01")
                pred = {}
                for etiqueta, receta, fisica in [("actual", "clasica", False), ("nueva", RECETA, True)]:
                    cols = variables(entreno, h, fisica)
                    m = RC.entrenar(entreno.loc[objetivo <= fin_entreno, cols], y[objetivo <= fin_entreno], params, n_arboles, receta, h)
                    ev = y.notna() & servicio[cols].notna().all(axis=1) & (ds.index >= ini_eval) & (objetivo <= ultimo)
                    pred[etiqueta] = pd.Series(m.predict(servicio.loc[ev, cols]), index=ds.index[ev])
                idx = pred["actual"].index.intersection(pred["nueva"].index)
                ea, en = (y[idx] - pred["actual"][idx]).values, (y[idx] - pred["nueva"][idx]).values
                anom = ds.loc[idx, f"anom_{zona}_lag0"].abs().values
                ext = anom > np.quantile(anom, 0.9)
                mejora = 1 - rmse(en, 0 * en) / rmse(ea, 0 * ea)
                ic05, ic95 = mejora_bootstrap(ea, en)
                pasa = bool(mejora >= minimo and ic05 > 0)
                fila = {"modelo": nombre, "campeon": f"v{mv.version}", "n": len(idx), "rmse_persistencia": rmse(y[idx], 0 * en),
                        "rmse_actual": rmse(ea, 0 * ea), "rmse_nueva": rmse(en, 0 * en), "mejora": mejora, "ic05": ic05,
                        "ic95": ic95, "mejora_extremos": 1 - rmse(en[ext], 0 * en[ext]) / rmse(ea[ext], 0 * ea[ext]),
                        "decision": "PROMOVER" if pasa else "mantener"}
                def residuos_json(errores, receta):
                    return {"fechas": [str(d.date()) for d in idx], "residuos": [round(float(e), 4) for e in errores],
                            "origen": f"receta {receta}, modelo hasta {fin_entreno.date()}, evaluado {idx[0].date()}→{idx[-1].date()}"}

                if solo_residuos:  # al modelo que está en producción, los errores de SU receta
                    receta_prod = mv.tags.get("receta", "clasica")
                    client.log_dict(mv.run_id, residuos_json(en if receta_prod == RECETA else ea, receta_prod),
                                    "residuos_validacion.json")
                    client.set_model_version_tag(nombre, mv.version, "residuos", "true")
                    promovidos.append(nombre)
                if pasa and promover_todo:
                    cols = variables(entreno, h, True)
                    final = RC.entrenar(entreno.loc[objetivo <= ultimo, cols], y[objetivo <= ultimo], params, n_arboles, RECETA, h)
                    with mlflow.start_run(run_name=nombre, nested=True) as hijo:
                        mlflow.log_params({"zona": zona, "h": h, "n_arboles": n_arboles, **params, "receta": RECETA})
                        mlflow.log_metrics({k: v for k, v in fila.items() if isinstance(v, float)})
                        mlflow.log_dict({q: float(np.quantile(en, q)) for q in (0.05, 0.10, 0.50, 0.90, 0.95)},
                                        "cuantiles_residuo_validacion.json")
                        mlflow.xgboost.log_model(final.arboles, name="modelo", input_example=servicio.loc[idx[:3], cols])
                        mlflow.log_dict(residuos_json(en, RECETA), "residuos_validacion.json")
                        if final.lineal:
                            mlflow.log_dict(final.lineal, "lineal.json")
                        nueva = mlflow.register_model(f"runs:/{hijo.info.run_id}/modelo", nombre)
                    etiquetas = {"receta": RECETA, "lineal": "true" if final.lineal else "false", "residuos": "true",
                                 "datos_hasta": str(ultimo.date()), "reentrenado": time.strftime("%Y-%m-%d"),
                                 "mejora_vs_campeon": round(mejora, 4), "usa_prevision": "true" if h <= 7 else "false"}
                    if h <= 7:
                        etiquetas["ventana_meteo"] = min(h, 7)
                    for k, v in etiquetas.items():
                        client.set_model_version_tag(nombre, nueva.version, k, str(v))
                    client.update_model_version(
                        nombre, nueva.version,
                        description=f"Receta {RECETA} (variables físicas, pesos recientes, aprendizaje lento"
                                    f"{', media con lineal' if final.lineal else ''}). Confirmación fuera de muestra "
                                    f"(modelos hasta 2021): RMSE {fila['rmse_nueva']:.3f} vs {fila['rmse_actual']:.3f} "
                                    f"({mejora:+.1%}, IC90 [{ic05:+.1%}, {ic95:+.1%}]). Reajustado hasta {ultimo.date()}.")
                    client.set_registered_model_alias(nombre, "anterior", mv.version)
                    client.set_registered_model_alias(nombre, "produccion", nueva.version)
                    fila["nueva_version"] = f"v{nueva.version}"
                    promovidos.append(nombre)
                filas.append(fila)
                print(f"{nombre:<15} actual {fila['rmse_actual']:.4f} -> nueva {fila['rmse_nueva']:.4f}  mejora {mejora:+.1%} "
                      f"[{ic05:+.1%}, {ic95:+.1%}]  extremos {fila['mejora_extremos']:+.1%}  -> {fila['decision']}"
                      + (f" ({fila['nueva_version']})" if "nueva_version" in fila else ""), flush=True)
        t = pd.DataFrame(filas)
        t.to_csv(SALIDA / "confirmacion.csv", index=False)
        mlflow.log_artifact(str(SALIDA / "confirmacion.csv"))
    print(f"\nMejora media {t.mejora.mean():+.2%} · en extremos {t.mejora_extremos.mean():+.2%} · "
          f"pasan la regla: {(t.decision == 'PROMOVER').sum()}/8 · peores: {(t.mejora < 0).sum()}/8")
    if promovidos:
        print(f"modelos actualizados: {promovidos} -> exportando")
        subprocess.run([sys.executable, str(RAIZ / "scripts" / "exportar_modelos.py")], check=True)
