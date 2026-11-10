"""Registra en el MLflow Model Registry los modelos del último entrenamiento y les pone el alias
'produccion'. Es el paso de "aprobar" un modelo: la API solo sirve lo que tenga ese alias.

Uso:
    python scripts/registrar_modelos.py                     # todo
    python scripts/registrar_modelos.py --modelos hipoxia   # solo la hipoxia (nueva versión)
"""
import mlflow
from mlflow.tracking import MlflowClient

from gemelo.modelos.comun import configurar_mlflow

ALIAS = "produccion"


def ultimo_run(client, experimento, nombre):
    exp = client.get_experiment_by_name(experimento)
    runs = client.search_runs([exp.experiment_id], f"attributes.run_name = '{nombre}' and attributes.status = 'FINISHED'",
                              order_by=["attributes.start_time DESC"], max_results=1)
    if not runs:
        raise SystemExit(f"No hay runs '{nombre}' terminados en '{experimento}': entrena primero.")
    return runs[0]


def registrar(client, uri, nombre, descripcion, tags):
    mv = mlflow.register_model(uri, nombre)
    client.update_model_version(nombre, mv.version, description=descripcion)
    for k, v in tags.items():
        client.set_model_version_tag(nombre, mv.version, k, str(v))
    client.set_registered_model_alias(nombre, ALIAS, mv.version)
    print(f"  {nombre} v{mv.version} -> @{ALIAS}")


def registrar_sst(client):
    padre = ultimo_run(client, "sst_mar_menor", "sst_xgboost_delta")
    hijos = client.search_runs([padre.info.experiment_id], f"tags.mlflow.parentRunId = '{padre.info.run_id}'")
    print(f"SST: run {padre.info.run_id} con {len(hijos)} modelos")
    for r in sorted(hijos, key=lambda r: (r.data.params["zona"], int(r.data.params["h"]))):
        p, m = r.data.params, r.data.metrics
        nombre = f"sst_{p['zona']}_h{p['h']}"
        desc = (f"XGBoost del cambio de anomalía de SST ({p['zona']}, h={p['h']} d). Test: RMSE "
                f"{m['test_rmse']:.3f} °C, skill vs persistencia amortiguada {m['test_skill_vs_amortiguada']:+.3f}.")
        registrar(client, f"runs:/{r.info.run_id}/modelo", nombre, desc,
                  {"test_rmse": round(m["test_rmse"], 4), "zona": p["zona"], "h": p["h"]})


MEJORA_MINIMA_PREVISION = 0.02  # promocionar solo si con previsión REAL mejora >= 2 % al modelo actual


def registrar_sst_prevision(client):
    """Fase 7: modelos de SST con previsión meteorológica. Antes de mover @produccion, la versión actual
    queda con el alias 'sin_prevision': el servicio la usa de respaldo cuando no hay previsión ese día."""
    configurar_mlflow("sst_prevision")
    padre = ultimo_run(client, "sst_prevision", "sst_con_prevision")
    hijos = client.search_runs([padre.info.experiment_id], f"tags.mlflow.parentRunId = '{padre.info.run_id}'")
    print(f"SST con previsión: run {padre.info.run_id}")
    for r in sorted(hijos, key=lambda r: (r.data.params["zona"], int(r.data.params["h"]))):
        p, m = r.data.params, r.data.metrics
        nombre = f"sst_{p['zona']}_h{p['h']}"
        mejora = m["skill_Breal_vs_A"]
        if mejora < MEJORA_MINIMA_PREVISION:
            print(f"  {nombre}: mejora {mejora:+.1%} < {MEJORA_MINIMA_PREVISION:.0%} -> se queda la versión actual")
            continue
        actual = client.get_model_version_by_alias(nombre, ALIAS)
        if actual.tags.get("usa_prevision") == "true":
            print(f"  {nombre}: @produccion ya usa previsión (v{actual.version}), se omite")
            continue
        client.set_registered_model_alias(nombre, "sin_prevision", actual.version)
        desc = (f"XGBoost del cambio de anomalía de SST ({p['zona']}, h={p['h']} d) CON previsión meteorológica "
                f"(ventana {p['ventana_meteo']} d). Evaluación con previsiones reales ago-2024→sep-2026: RMSE "
                f"{m['rmse_B_real']:.3f} °C vs {m['rmse_A']:.3f} sin previsión ({mejora:+.1%}).")
        registrar(client, f"runs:/{r.info.run_id}/modelo", nombre, desc,
                  {"usa_prevision": "true", "ventana_meteo": p["ventana_meteo"], "rmse_prevision_real": round(m["rmse_B_real"], 4),
                   "mejora_vs_sin_prevision": round(mejora, 4)})
        print(f"     (v{actual.version} -> @sin_prevision)")


def registrar_hipoxia(client):
    configurar_mlflow("hipoxia_mar_menor")
    r = ultimo_run(client, "hipoxia_mar_menor", "hipoxia_loyo")
    m = r.data.metrics
    print(f"Hipoxia: run {r.info.run_id}")
    registrar(client, f"runs:/{r.info.run_id}/modelo_logistica", "hipoxia_logistica",
              f"Logística de riesgo de hipoxia en la próxima campaña. LOYO: recall {m['logistica_recall']:.2f}, "
              f"precisión {m['logistica_precision']:.2f}, inicios anticipados {int(m['logistica_inicios_anticipados'])}/9.",
              {"recall": round(m["logistica_recall"], 3), "precision": round(m["logistica_precision"], 3)})


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelos", nargs="+", choices=["sst", "sst_prevision", "hipoxia"], default=["sst", "hipoxia"],
                    help="qué familias registrar (por defecto: sst e hipoxia)")
    args = ap.parse_args()
    configurar_mlflow("sst_mar_menor")
    cliente = MlflowClient()
    if "sst" in args.modelos:
        registrar_sst(cliente)
    if "sst_prevision" in args.modelos:
        registrar_sst_prevision(cliente)
    if "hipoxia" in args.modelos:
        registrar_hipoxia(cliente)
