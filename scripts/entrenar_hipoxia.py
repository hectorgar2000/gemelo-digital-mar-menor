"""Fase 3: modelo de riesgo de hipoxia en la próxima campaña, con validación dejando un año fuera.

Uso:
    python scripts/entrenar_hipoxia.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import shap

from gemelo.config import RAIZ, cargar_config
from gemelo.modelos import hipoxia as H
from gemelo.modelos import xai
from gemelo.modelos.comun import FIGURAS, MANIFIESTO, columnas_x, configurar_mlflow

if __name__ == "__main__":
    cfg = cargar_config()
    FIGURAS.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(RAIZ / "data" / "processed" / "dataset_hipoxia.parquet")
    d, y, persist = H.preparar(df, cfg["umbrales"]["hipoxia_area_pct"])
    configurar_mlflow("hipoxia_mar_menor")

    with mlflow.start_run(run_name="hipoxia_loyo"):
        mlflow.log_artifact(str(MANIFIESTO))
        mlflow.log_params({"validacion": "dejar un año fuera", "vars_logistica": ",".join(H.VARS_LOGISTICA),
                           **{f"xgb_{k}": v for k, v in H.PARAMS_XGB.items()}})
        probs = H.probabilidades_loyo(d, y)
        probs["persistencia"] = persist
        filas = {}
        for nombre in ["persistencia", "logistica", "xgboost"]:
            m = H.metricas(y, probs[nombre])
            filas[nombre] = m
            mlflow.log_metrics({f"{nombre}_{k}": v for k, v in m.items() if pd.notna(v)})
        tabla = pd.DataFrame(filas).T
        tabla.to_csv(FIGURAS / "metricas_hipoxia.csv")
        mlflow.log_artifact(str(FIGURAS / "metricas_hipoxia.csv"))
        print(tabla.round(3).to_string())

        # Detalle por episodio: ¿qué prob. daba cada modelo en el inicio de cada episodio?
        ep = H.episodios(y)
        inicios = ep.notna() & ep.ne(ep.shift())
        det = probs[inicios].round(2).assign(pct_hipoxia_siguiente=d.loc[inicios, "y_pct_hipoxia"].round(1),
                                             pct_ahora=d.loc[inicios, "ahora_pct_hipoxia"].round(1))
        print("\nInicio de cada episodio (probabilidad fuera de muestra):\n" + det.to_string())
        det.to_csv(FIGURAS / "hipoxia_inicios_episodio.csv")

        # Explicación: coeficientes de la logística + SHAP agrupado del XGBoost (entrenados con todo)
        log, coefs, xg = H.modelos_finales(d, y)
        print("\nCoeficientes logística (variables estandarizadas):\n" + coefs.round(2).to_string())
        mlflow.log_dict(coefs.round(4).to_dict(), "coeficientes_logistica.json")
        X = d[columnas_x(d)]
        sv = shap.TreeExplainer(xg)(X)
        imp = xai.importancia_por_grupo(sv, X.columns)
        print("\nSHAP XGBoost por grupo:\n" + imp.round(3).to_string())

        fig, axs = plt.subplots(1, 2, figsize=(14, 5))
        coefs.sort_values().plot.barh(ax=axs[0], color=["#c0392b" if v > 0 else "#2e86c1" for v in coefs.sort_values()])
        axs[0].set_title("Logística: coeficiente (↑ riesgo si > 0)")
        imp.sort_values().plot.barh(ax=axs[1], color="#3b75af")
        axs[1].set_title("XGBoost: media |SHAP| por grupo (log-odds)")
        fig.tight_layout()
        fig.savefig(FIGURAS / "hipoxia_explicacion.png", dpi=110, bbox_inches="tight")
        mlflow.log_artifact(str(FIGURAS / "hipoxia_explicacion.png"), "figuras")

        fig, ax = plt.subplots(figsize=(14, 4))
        d["y_pct_hipoxia"].div(100).plot(ax=ax, color="k", lw=1, label="% fondo en hipoxia (siguiente campaña)/100")
        probs[["logistica", "xgboost"]].plot(ax=ax, lw=1)
        ax.axhline(0.5, ls="--", c="grey")
        ax.set_title("Probabilidad de hipoxia fuera de muestra (año dejado fuera)")
        ax.legend(loc="upper left")
        fig.savefig(FIGURAS / "hipoxia_probabilidades.png", dpi=110, bbox_inches="tight")
        mlflow.log_artifact(str(FIGURAS / "hipoxia_probabilidades.png"), "figuras")
