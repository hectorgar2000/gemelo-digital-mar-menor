"""Reentrenamiento disparado por el monitor de deriva: campeón (@produccion) vs. retador.

    python scripts/reentrenar.py                       # los modelos que el monitor marque (config monitor.disparar_con)
    python scripts/reentrenar.py --modelos sst_costa_h30
    python scripts/reentrenar.py --forzar              # todos los modelos de SST
    python scripts/reentrenar.py --sin-promover        # solo evalúa y registra en MLflow (no cambia producción)

Para cada modelo:
  1. Retador = mismo tipo e hiperparámetros que el campeón, entrenado con todo lo anterior a la ventana de
     evaluación (últimos `reentrenamiento.ventana_evaluacion_dias` con verdad conocida), con purga de frontera.
     Las variables de previsión se entrenan con ERA5 ("previsión perfecta"), como en la Fase 7.
  2. Campeón y retador se evalúan en esa ventana con las variables de SERVICIO (previsión real).
  3. Si el retador mejora >= `mejora_minima`: se reajusta con TODOS los datos, se registra como nueva versión,
     el campeón pasa a @anterior (para poder volver atrás) y el nuevo a @produccion. Al final, se exporta.
"""
import argparse
import subprocess
import sys
import time

import mlflow
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient

from gemelo.config import RAIZ, cargar_config
from gemelo.modelos import receta as RC
from gemelo.modelos import reentrenamiento as R
from gemelo.modelos.comun import MANIFIESTO, configurar_mlflow, mejora_bootstrap, rmse
from gemelo.monitor.deriva import Monitor

PROCESSED = RAIZ / "data" / "processed"


def paso(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def campeon(client, nombre):
    mv = client.get_model_version_by_alias(nombre, "produccion")
    p = client.get_run(mv.run_id).data.params
    params = {"max_depth": int(p["max_depth"]), "min_child_weight": int(p["min_child_weight"])}
    return mv, params, int(p["n_arboles"])


def log_modelo(modelo, ejemplo):
    """Registra los árboles como modelo de MLflow y, si la hay, la parte lineal como artefacto JSON."""
    mlflow.xgboost.log_model(modelo.arboles, name="modelo", input_example=ejemplo)
    if modelo.lineal:
        mlflow.log_dict(modelo.lineal, "lineal.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelos", nargs="+")
    ap.add_argument("--forzar", action="store_true")
    ap.add_argument("--sin-promover", action="store_true")
    a = ap.parse_args()

    cfg = cargar_config()
    cr = cfg["reentrenamiento"]
    mon = Monitor()
    g = mon.g
    ultimo = g.ultima_fecha()
    if a.modelos:
        candidatos = a.modelos
    elif a.forzar:
        candidatos = list(g.manifiesto["sst"])
    else:
        inf = mon.informe()
        candidatos = inf["reentrenar"]
        paso(f"monitor {inf['fecha']}: estado {inf['estado_global']} · a reentrenar: {candidatos or 'ninguno'}")
    if not candidatos:
        sys.exit(0)

    ds = g.ds
    f_era5 = pd.read_parquet(PROCESSED / "meteo_futura_era5.parquet")
    configurar_mlflow("reentrenamiento")
    client = MlflowClient()
    filas, promovidos = [], []
    with mlflow.start_run(run_name=f"reentrenamiento_{ultimo.date()}") as padre:
        mlflow.log_artifact(str(MANIFIESTO))
        for nombre in candidatos:
            info = g.manifiesto["sst"][nombre]
            zona, h = info["zona"], info["h"]
            variables = info["variables"]
            mv, params, n_arboles = campeon(client, nombre)
            v = R.ventana_evaluacion(ultimo, h, cr["ventana_evaluacion_dias"])
            if g.datos_hasta(info) >= v.ini + pd.Timedelta(days=h):
                paso(f"{nombre}: el campeón v{mv.version} ya vio la ventana de evaluación (datos hasta "
                     f"{g.datos_hasta(info).date()}): la comparación no sería honesta, se omite")
                continue
            y = ds[f"y_delta_{zona}_h{h}"]

            # entrenamiento: estado del mar + previsión "perfecta" (ERA5); evaluación: variables de servicio
            X_entreno = ds.join(f_era5)[variables]
            eval_idx = ds.loc[v.ini:v.fin].index
            X_eval = g.matriz(eval_idx, variables)
            ok = y.loc[eval_idx].notna() & X_eval.notna().all(axis=1)
            X_eval, y_eval = X_eval[ok], y.loc[eval_idx][ok]

            m_entreno = R.mascara_entrenamiento(ds.index, v.ini, h)
            receta = info.get("receta", "clasica")   # el retador usa la receta de su campeón
            retador = RC.entrenar(X_entreno[m_entreno], y[m_entreno], params, n_arboles, receta, h)
            p_ret = retador.predict(X_eval)
            p_cam = g.modelos[nombre].predict(X_eval)
            r_cam, r_ret, r_per = rmse(y_eval, p_cam), rmse(y_eval, p_ret), rmse(y_eval, np.zeros(len(y_eval)))
            promover, mejora = R.decidir(r_cam, r_ret, cr["mejora_minima"])
            # además de mejorar >= mínimo, la mejora debe ser distinguible del ruido (IC90 por bloques > 0)
            ic05, ic95 = mejora_bootstrap(y_eval.values - p_cam, y_eval.values - p_ret)
            promover = promover and ic05 > 0
            fila = {"modelo": nombre, "campeon": f"v{mv.version}", "n_eval": len(y_eval),
                    "ventana": f"{v.ini.date()}→{v.fin.date()}", "rmse_persistencia": r_per,
                    "rmse_campeon": r_cam, "rmse_retador": r_ret, "mejora": mejora, "ic05": ic05, "ic95": ic95,
                    "decision": "PROMOVER" if promover else "mantener"}

            with mlflow.start_run(run_name=nombre, nested=True) as hijo:
                mlflow.log_params({"zona": zona, "h": h, "campeon_version": mv.version, "n_arboles": n_arboles,
                                   **params, "ventana_evaluacion": fila["ventana"]})
                mlflow.log_metrics({k: fila[k] for k in ("rmse_persistencia", "rmse_campeon", "rmse_retador", "mejora", "ic05", "ic95")})
                if promover and not a.sin_promover:
                    # reajuste con TODOS los datos con verdad conocida (incluida la ventana evaluada)
                    m_todo = R.mascara_entrenamiento(ds.index, ultimo + pd.Timedelta(days=1), h)
                    final = RC.entrenar(X_entreno[m_todo], y[m_todo], params, n_arboles, receta, h)
                    res = y_eval.values - p_ret  # errores FUERA de muestra del retador -> cuantiles del intervalo
                    mlflow.log_dict({q: float(np.quantile(res, q)) for q in (0.05, 0.10, 0.50, 0.90, 0.95)},
                                    "cuantiles_residuo_validacion.json")
                    mlflow.log_dict({"fechas": [str(d.date()) for d in y_eval.index], "residuos": [round(float(r), 4) for r in res],
                                     "origen": f"retador evaluado en {fila['ventana']}"}, "residuos_validacion.json")
                    log_modelo(final, X_eval.head(3))
                    nueva = mlflow.register_model(f"runs:/{hijo.info.run_id}/modelo", nombre)
                    etiquetas = {"datos_hasta": str(ultimo.date()), "reentrenado": time.strftime("%Y-%m-%d"),
                                 "mejora_vs_campeon": round(mejora, 4), "rmse_ventana_retador": round(r_ret, 4),
                                 "usa_prevision": "true" if info.get("usa_prevision") else "false",
                                 "receta": receta, "lineal": "true" if final.lineal else "false", "residuos": "true"}
                    if info.get("usa_prevision"):
                        etiquetas["ventana_meteo"] = info["ventana_meteo"]
                    for k, val in etiquetas.items():
                        client.set_model_version_tag(nombre, nueva.version, k, str(val))
                    client.update_model_version(
                        nombre, nueva.version,
                        description=f"Reentrenado con datos hasta {ultimo.date()} (deriva). En {fila['ventana']}: "
                                    f"RMSE {r_ret:.3f} vs {r_cam:.3f} del campeón v{mv.version} ({mejora:+.1%}).")
                    client.set_registered_model_alias(nombre, "anterior", mv.version)
                    client.set_registered_model_alias(nombre, "produccion", nueva.version)
                    fila["nueva_version"] = f"v{nueva.version}"
                    promovidos.append(nombre)
            filas.append(fila)
            paso(f"{nombre}: campeón v{mv.version} {r_cam:.3f} · retador {r_ret:.3f} · persistencia {r_per:.3f} "
                 f"· mejora {mejora:+.1%} [IC90 {ic05:+.1%}, {ic95:+.1%}] -> {fila['decision']}" + (f" ({fila.get('nueva_version')})" if promover else ""))

        tabla = pd.DataFrame(filas)
        salida = RAIZ / "docs" / "figuras" / "fase7" / "reentrenamiento.csv"
        tabla.to_csv(salida, index=False)
        mlflow.log_artifact(str(salida))
    print(tabla.round(3).to_string(index=False))

    if promovidos and not a.sin_promover:
        paso(f"promovidos: {promovidos} -> exportando modelos de servicio")
        subprocess.run([sys.executable, str(RAIZ / "scripts" / "exportar_modelos.py")], check=True)
