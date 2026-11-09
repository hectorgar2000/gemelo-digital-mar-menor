"""Exporta los modelos con alias @produccion del Model Registry a models/servicio/: un artefacto
portable (sin rutas absolutas ni dependencia de MLflow) que es lo que carga la API y la imagen Docker.

    MLflow Registry (@produccion) --exportar--> models/servicio/  --montar--> API / Docker

Uso:
    python scripts/exportar_modelos.py
"""
import json
import shutil
import subprocess
from datetime import datetime, timezone

import joblib
import mlflow
import sklearn
import xgboost
from mlflow.tracking import MlflowClient

from gemelo.config import RAIZ, cargar_config
from gemelo.modelos.comun import configurar_mlflow
from gemelo.procesado.features import ZONAS

DESTINO = RAIZ / "models" / "servicio"
ALIAS = "produccion"


def info_version(client, nombre):
    mv = client.get_model_version_by_alias(nombre, ALIAS)
    return mv, client.get_run(mv.run_id)


if __name__ == "__main__":
    cfg = cargar_config()
    configurar_mlflow("sst_mar_menor")
    client = MlflowClient()
    if DESTINO.exists():
        shutil.rmtree(DESTINO)
    DESTINO.mkdir(parents=True)

    manifiesto = {
        "creado": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=RAIZ,
                                     capture_output=True, text=True).stdout.strip(),
        "versiones": {"xgboost": xgboost.__version__, "scikit-learn": sklearn.__version__},
        "sst": {}, "hipoxia": {},
    }

    for zona in ZONAS:
        for h in cfg["horizontes_dias"]:
            nombre = f"sst_{zona}_h{h}"
            mv, run = info_version(client, nombre)
            modelo = mlflow.xgboost.load_model(f"models:/{nombre}@{ALIAS}")
            fichero = f"{nombre}.json"
            modelo.save_model(DESTINO / fichero)  # formato nativo JSON de XGBoost: portable
            ruta_q = client.download_artifacts(run.info.run_id, "cuantiles_residuo_validacion.json")
            with open(ruta_q, encoding="utf-8") as f:
                cuantiles = json.load(f)
            manifiesto["sst"][nombre] = {
                "zona": zona, "h": h, "fichero": fichero, "version_registro": int(mv.version),
                "run_id": run.info.run_id, "variables": modelo.get_booster().feature_names,
                "cuantiles_residuo_validacion": cuantiles,
                "metricas_test": {k: round(v, 4) for k, v in run.data.metrics.items() if k.startswith("test_")},
            }
            print(f"  {nombre} v{mv.version} -> {fichero}")

    mv, run = info_version(client, "hipoxia_logistica")
    log = mlflow.sklearn.load_model(f"models:/hipoxia_logistica@{ALIAS}")
    joblib.dump(log, DESTINO / "hipoxia_logistica.joblib")
    servicio = json.load(open(client.download_artifacts(run.info.run_id, "servicio.json"), encoding="utf-8"))
    manifiesto["hipoxia"] = {
        "fichero": "hipoxia_logistica.joblib", "version_registro": int(mv.version), "run_id": run.info.run_id,
        "variables": servicio["variables"], "umbral_alerta": servicio["umbral_alerta"],
        "umbral_pct_hipoxia": cfg["umbrales"]["hipoxia_area_pct"],
        "metricas_loyo": {k: round(v, 4) for k, v in run.data.metrics.items() if k.startswith("logistica_")},
    }
    print(f"  hipoxia_logistica v{mv.version} -> hipoxia_logistica.joblib")

    (DESTINO / "servicio.json").write_text(json.dumps(manifiesto, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Manifiesto: {DESTINO / 'servicio.json'}")
