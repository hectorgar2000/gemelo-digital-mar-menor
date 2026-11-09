"""Explicabilidad (XAI) con SHAP: importancia global por grupos y explicación de días concretos.

Nota para interpretar: los modelos de SST predicen el CAMBIO de anomalía (delta). Un SHAP positivo
significa "esta variable empuja a que el agua se caliente más de lo que dice la persistencia".
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from gemelo.procesado.features import grupo_variable


def valores_shap(modelo, X: pd.DataFrame) -> shap.Explanation:
    return shap.TreeExplainer(modelo)(X)


def agrupar(shap_vals: shap.Explanation, columnas) -> pd.DataFrame:
    """Suma los SHAP de las variables de un mismo grupo (fila a fila). La suma es válida porque
    SHAP es aditivo: el grupo aporta exactamente la suma de sus variables."""
    df = pd.DataFrame(shap_vals.values, columns=columnas)
    return df.T.groupby([grupo_variable(c) for c in columnas]).sum().T


def importancia_por_grupo(shap_vals, columnas) -> pd.Series:
    return agrupar(shap_vals, columnas).abs().mean().sort_values(ascending=False)


def figuras_shap(shap_vals, X, imp_grupos, pred_test, zona, h) -> dict:
    figs = {}
    # 1) Importancia global por grupos
    fig, ax = plt.subplots(figsize=(7, 4))
    imp_grupos.sort_values().plot.barh(ax=ax, color="#3b75af")
    ax.set_xlabel("media |SHAP| (°C de cambio de anomalía)")
    ax.set_title(f"Importancia por grupo · {zona} · h={h} d (test)")
    figs[f"shap_grupos_{zona}_h{h}"] = fig

    # 2) Beeswarm de las 15 variables más influyentes
    plt.figure()
    shap.plots.beeswarm(shap_vals, max_display=15, show=False)
    fig = plt.gcf()
    fig.suptitle(f"SHAP por variable · {zona} · h={h} d (test)", y=1.02)
    figs[f"shap_beeswarm_{zona}_h{h}"] = fig

    # 3) Explicaciones locales: un ACIERTO (mayor calentamiento predicho y real a la vez)
    #    y un FALLO (mayor error). El fallo sirve para depurar: ¿en qué se apoyó el modelo?
    pred, real = pred_test["pred"].values, pred_test["y_delta"].values
    casos = {"acierto": int(np.argmax(np.minimum(pred, real))), "fallo": int(np.argmax(np.abs(pred - real)))}
    for tipo, idx in casos.items():
        plt.figure()
        shap.plots.waterfall(shap_vals[idx], max_display=12, show=False)
        fig = plt.gcf()
        fig.suptitle(f"{tipo.upper()} · {pred_test.index[idx].date()} · {zona} · h={h}: "
                     f"cambio predicho {pred[idx]:+.2f} °C (real {real[idx]:+.2f} °C)", y=1.02)
        figs[f"shap_{tipo}_{zona}_h{h}"] = fig
    return figs


def mapa_grupos_por_horizonte(importancias: dict):
    tabla = pd.DataFrame({f"{z} h{h}": imp for (z, h), imp in importancias.items()}).fillna(0)
    tabla = tabla.div(tabla.sum(), axis=1).mul(100)  # % del total de cada modelo
    tabla = tabla.loc[tabla.mean(axis=1).sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(tabla.values, cmap="Blues", aspect="auto")
    ax.set_xticks(range(tabla.shape[1]), tabla.columns, rotation=45, ha="right")
    ax.set_yticks(range(tabla.shape[0]), tabla.index)
    for i in range(tabla.shape[0]):
        for j in range(tabla.shape[1]):
            ax.text(j, i, f"{tabla.values[i, j]:.0f}", ha="center", va="center", fontsize=8,
                    color="white" if tabla.values[i, j] > 40 else "black")
    ax.set_title("% de la importancia SHAP por grupo, zona y horizonte (test)")
    fig.colorbar(im, ax=ax, shrink=0.8)
    return fig
