"""Fase 3: entrena XGBoost de anomalía de SST (laguna/costa × horizontes), compara con líneas base,
deriva olas de calor, calcula SHAP agrupado y lo registra todo en MLflow.

Uso:
    python scripts/entrenar_sst.py
    mlflow ui --backend-store-uri sqlite:///mlflow.db     # para ver los experimentos
"""
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import pandas as pd

from gemelo.config import RAIZ, cargar_config
from gemelo.modelos import xai
from gemelo.modelos.comun import FIGURAS, MANIFIESTO, configurar_mlflow
from gemelo.modelos.sst import entrenar_zona_horizonte, evaluar_mhw
from gemelo.procesado.features import ZONAS, umbral_p90

SHAP_HORIZONTES = [1, 7, 30]  # horizontes con gráficos SHAP detallados


def paso(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def log_modelo(modelo, ejemplo):
    try:  # MLflow 3 usa `name`; MLflow 2, `artifact_path`
        mlflow.xgboost.log_model(modelo, name="modelo", input_example=ejemplo)
    except TypeError:
        mlflow.xgboost.log_model(modelo, artifact_path="modelo", input_example=ejemplo)


if __name__ == "__main__":
    cfg = cargar_config()
    ds = pd.read_parquet(RAIZ / "data" / "processed" / "dataset_sst.parquet")
    diario = pd.read_parquet(RAIZ / "data" / "interim" / "diario.parquet")
    FIGURAS.mkdir(parents=True, exist_ok=True)
    configurar_mlflow("sst_mar_menor")

    filas, importancias = [], {}
    with mlflow.start_run(run_name="sst_xgboost_delta") as padre:
        mlflow.log_artifact(str(MANIFIESTO))
        mlflow.log_dict(cfg, "config_zona.json")
        for zona in ZONAS:
            umbral = umbral_p90(diario[f"sst_{zona}"], *cfg["umbrales"]["climatologia"],
                                pct=cfg["umbrales"]["mhw_percentil"])
            for h in cfg["horizontes_dias"]:
                paso(f"{zona} h={h}: validación de origen móvil + entrenamiento")
                r = entrenar_zona_horizonte(ds, cfg["particion"], zona, h)
                r.metricas.update(evaluar_mhw(r.pred_test, ds, umbral, zona, h))

                Xte = ds.loc[r.pred_test.index, r.modelo_final.get_booster().feature_names]
                shap_vals = xai.valores_shap(r.modelo_final, Xte)
                imp = xai.importancia_por_grupo(shap_vals, Xte.columns)
                importancias[(zona, h)] = imp

                with mlflow.start_run(run_name=f"{zona}_h{h}", nested=True):
                    mlflow.log_params({"zona": zona, "h": h, "objetivo": f"y_delta_{zona}_h{h}",
                                       "n_arboles": r.n_arboles, **r.params})
                    mlflow.log_metrics(r.metricas)
                    mlflow.log_dict(r.cuantiles_residuo, "cuantiles_residuo_validacion.json")
                    mlflow.log_dict(imp.round(4).to_dict(), "shap_por_grupo.json")
                    log_modelo(r.modelo_final, Xte.head(3))
                    if h in SHAP_HORIZONTES:
                        for nombre, fig in xai.figuras_shap(shap_vals, Xte, imp, r.pred_test, zona, h).items():
                            ruta = FIGURAS / f"{nombre}.png"
                            fig.savefig(ruta, dpi=110, bbox_inches="tight")
                            plt.close(fig)
                            mlflow.log_artifact(str(ruta), "figuras")

                m = r.metricas
                filas.append({"zona": zona, "h": h, **{k: round(v, 3) for k, v in m.items()}})
                paso(f"   RMSE test {m['test_rmse']:.3f} · amortiguada {m['test_rmse_amortiguada']:.3f} · "
                     f"skill {m['test_skill_vs_amortiguada']:+.3f} · cobertura90 {m['test_cobertura_90']:.2f} · "
                     f"F1 MHW {m['mhw_modelo_f1']:.2f} (amort. {m['mhw_amortiguada_f1']:.2f})")

        tabla = pd.DataFrame(filas)
        tabla.to_csv(FIGURAS / "metricas_sst.csv", index=False)
        mlflow.log_artifact(str(FIGURAS / "metricas_sst.csv"))
        fig = xai.mapa_grupos_por_horizonte(importancias)
        fig.savefig(FIGURAS / "shap_grupos_por_horizonte.png", dpi=110, bbox_inches="tight")
        mlflow.log_artifact(str(FIGURAS / "shap_grupos_por_horizonte.png"), "figuras")
        paso(f"run MLflow {padre.info.run_id}")

    cols = ["zona", "h", "val_rmse", "val_rmse_amortiguada", "val_skill_vs_amortiguada", "test_rmse",
            "test_rmse_persistencia", "test_rmse_climatologia", "test_rmse_amortiguada",
            "test_skill_vs_amortiguada", "test_skill_vs_climatologia", "test_cobertura_90",
            "mhw_modelo_f1", "mhw_amortiguada_f1", "mhw_modelo_recall", "mhw_modelo_precision"]
    print(tabla[cols].to_string(index=False))
    print(json.dumps({f"{z}_h{h}": v.head(4).round(3).to_dict() for (z, h), v in importancias.items()},
                     indent=1, ensure_ascii=False))
