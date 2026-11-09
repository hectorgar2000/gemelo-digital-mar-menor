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
    ap.add_argument("--modelos", nargs="+", choices=["sst", "hipoxia"], default=["sst", "hipoxia"],
                    help="qué familias registrar (por defecto, todas)")
    args = ap.parse_args()
    configurar_mlflow("sst_mar_menor")
    cliente = MlflowClient()
    if "sst" in args.modelos:
        registrar_sst(cliente)
    if "hipoxia" in args.modelos:
        registrar_hipoxia(cliente)
