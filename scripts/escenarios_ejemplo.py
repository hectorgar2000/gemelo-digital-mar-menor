"""Fase 5: escenarios de ejemplo (laguna, inicio del verano de 2024) con gráfico y tabla.

Uso:
    python scripts/escenarios_ejemplo.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gemelo.config import RAIZ
from gemelo.servicio.escenarios import MotorEscenarios, Perturbacion
from gemelo.servicio.motor import Gemelo

FECHA, CAMPANA, ZONA, HORIZONTES = "2024-06-15", "2024-06-10", "laguna", [1, 3, 7, 30]
ESCENARIOS = {
    "Base (real)": [],
    "Episodio agua +1,5 °C (30 d)": [Perturbacion("temp_agua", 1.5, 0.3, 30)],
    "Clima de fondo +1,5 °C (delta)": [Perturbacion("calentamiento_base", 1.5, 0.3)],
    "Viento ×0,5 (7 d)": [Perturbacion("viento", 0.5, 0.1, 7)],
    "Aire +2 °C (14 d)": [Perturbacion("temp_aire", 2.0, 0.5, 14)],
    "Lluvia ×3 (30 d)": [Perturbacion("lluvia", 3.0, 0.5, 30)],
}
SALIDA = RAIZ / "docs" / "figuras" / "fase5"

if __name__ == "__main__":
    SALIDA.mkdir(parents=True, exist_ok=True)
    g = Gemelo()
    me = MotorEscenarios(g)
    filas, sst = [], {}
    for nombre, perts in ESCENARIOS.items():
        r = me.simular_sst(ZONA, HORIZONTES, FECHA, perts)
        hx = me.simular_hipoxia(CAMPANA, perts)
        sst[nombre] = r["horizontes"]
        for x in r["horizontes"]:
            filas.append({"escenario": nombre, "h": x["h"], "sst_p50": x["escenario"]["sst_p50"],
                          "sst_p05": x["escenario"]["sst_p05"], "sst_p95": x["escenario"]["sst_p95"],
                          "dif_p50": x["diferencia_sst"]["p50"], "prob_ola_calor": x["escenario"]["prob_ola_calor"],
                          "n_fuera_rango": len(x["fuera_de_rango"]),
                          "prob_hipoxia_p50": hx["escenario"]["prob_hipoxia"]["p50"]})
    t = pd.DataFrame(filas)
    t.to_csv(SALIDA / "escenarios_ejemplo.csv", index=False)
    print(t.to_string(index=False))

    # Gráfico: SST prevista (p05-p95) y probabilidad de ola de calor por escenario y horizonte
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    n = len(ESCENARIOS)
    for i, (nombre, hs) in enumerate(sst.items()):
        x = np.arange(len(HORIZONTES)) + (i - n / 2) * 0.13
        p50 = [h["escenario"]["sst_p50"] for h in hs]
        err = [[p - h["escenario"]["sst_p05"] for p, h in zip(p50, hs)],
               [h["escenario"]["sst_p95"] - p for p, h in zip(p50, hs)]]
        axs[0].errorbar(x, p50, yerr=err, fmt="o", capsize=3, label=nombre)
        axs[1].bar(x, [h["escenario"]["prob_ola_calor"] for h in hs], width=0.12, label=nombre)
    umbrales = [h["umbral_ola_calor"] for h in sst["Base (real)"]]
    axs[0].plot(np.arange(len(HORIZONTES)), umbrales, "k--", lw=1, label="umbral ola de calor (p90)")
    for ax, tit in zip(axs, ["SST prevista en la laguna (mediana y 90 %)", "Probabilidad de ola de calor"]):
        ax.set_xticks(range(len(HORIZONTES)), [f"h={h} d" for h in HORIZONTES])
        ax.set_title(f"{tit} · base {FECHA}")
    axs[0].set_ylabel("°C")
    axs[1].set_ylim(0, 1)
    axs[1].legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(SALIDA / "escenarios_laguna.png", dpi=110, bbox_inches="tight")
    print(f"figura: {SALIDA / 'escenarios_laguna.png'}")
