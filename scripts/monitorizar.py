"""Monitor de deriva del gemelo.

Uso:
    python scripts/monitorizar.py                    # informe del último día con datos
    python scripts/monitorizar.py --fecha 2025-08-31 # informe "como si" fuera ese día
    python scripts/monitorizar.py --historico        # control mensual 2022 -> último día (tabla + figura)

Salidas:
    data/processed/monitor.json                  último informe (lo sirve la API en GET /monitor)
    data/processed/monitor_historico.parquet     con --historico
    docs/figuras/fase7/monitor_historico.png     con --historico
"""
import argparse
import json

import pandas as pd

from gemelo.config import RAIZ
from gemelo.monitor.deriva import Monitor

PROCESSED = RAIZ / "data" / "processed"
FIGURAS = RAIZ / "docs" / "figuras" / "fase7"


def imprimir(inf):
    print(f"\n=== Monitor {inf['fecha']} · estado global: {inf['estado_global']} · reentrenar: {inf['reentrenar'] or 'nada'}")
    for n, m in inf["modelos"].items():
        r, e = m["rendimiento"], m["entradas"]
        skill = f"{r['skill']:+.1%}" if r["skill"] is not None else "  —  "
        cob = f"{r['cobertura']:.0%}" if r["cobertura"] is not None else "—"
        print(f"  {n:<15} {m['estado']:<6} v{m['version']} {'prev' if m['usa_prevision'] else '    '} "
              f"skill vs pers {skill:>7} · cobertura {cob:>4} · PSI máx {e['psi_max'] or 0:.2f} "
              f"({e['variable_psi_max']}) · fuera de rango {e['fuera_rango'] or 0:.0%}")
        for mot in m["motivos"]:
            print(f"      - {mot}")
    print(f"  previsión: {inf['prevision']['estado']} {inf['prevision'].get('motivos') or ''}")
    print(f"  hipoxia:   {inf['hipoxia']['estado']} {inf['hipoxia'].get('motivos') or ''}")


def figura_historico(h: pd.DataFrame, ruta):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    modelos = sorted(h["modelo"].unique(), key=lambda n: (n.split("_")[1], int(n.split("_h")[1])))
    color = {"OK": "#2ca02c", "AVISO": "#f2b134", "DERIVA": "#d62728"}
    fig, axs = plt.subplots(3, 1, figsize=(14, 10), sharex=True, gridspec_kw={"height_ratios": [2, 1.3, 1.3]})
    for i, n in enumerate(modelos):
        d = h[h["modelo"] == n]
        axs[0].scatter(d["fecha"], [i] * len(d), c=[color[e] for e in d["estado"]], s=60, marker="s")
    axs[0].set_yticks(range(len(modelos)), modelos)
    axs[0].set_title("Estado de cada modelo en el control mensual (verde OK · amarillo AVISO · rojo DERIVA)")
    for n in modelos:
        d = h[h["modelo"] == n]
        axs[1].plot(d["fecha"], d["skill"], lw=1, label=n)
        axs[2].plot(d["fecha"], d["cobertura"], lw=1)
    axs[1].axhline(0, c="k", lw=1)
    axs[1].set_ylabel("skill vs persistencia")
    axs[1].legend(ncol=4, fontsize=7)
    axs[2].axhline(0.9, c="grey", ls="--", lw=1)
    axs[2].axhline(0.8, c="r", ls="--", lw=1)
    axs[2].set_ylabel("cobertura intervalo 90 %")
    fig.tight_layout()
    fig.savefig(ruta, dpi=110, bbox_inches="tight")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fecha")
    ap.add_argument("--historico", action="store_true")
    ap.add_argument("--calibrar", action="store_true", help="recalcula los umbrales PSI con años normales")
    a = ap.parse_args()
    mon = Monitor()
    if a.calibrar:
        ini, fin = mon.cm["anios_calibracion"]
        umbrales = mon.calibrar(range(ini, fin + 1))
        (PROCESSED / "monitor_calibracion.json").write_text(json.dumps(umbrales, indent=1), encoding="utf-8")
        print(f"Umbrales PSI calibrados (p{mon.cm['psi_percentil_calibracion']} en {ini}-{fin}):", umbrales)
        mon = Monitor()  # recargar con la calibración
    inf = mon.informe(a.fecha)
    imprimir(inf)
    if not a.fecha:
        (PROCESSED / "monitor.json").write_text(json.dumps(inf, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"\ninforme guardado en {PROCESSED / 'monitor.json'}")
    if a.historico:
        fechas = pd.date_range("2022-01-31", mon.g.ultima_fecha(), freq="ME")
        h = mon.historico(fechas)
        h.to_parquet(PROCESSED / "monitor_historico.parquet")
        FIGURAS.mkdir(parents=True, exist_ok=True)
        figura_historico(h, FIGURAS / "monitor_historico.png")
        resumen = h.pivot_table(index=h["fecha"].dt.year, columns="estado", values="modelo", aggfunc="count", fill_value=0)
        print("\nControles por año y estado (8 modelos × meses):\n" + resumen.to_string())
        print(f"figura: {FIGURAS / 'monitor_historico.png'}")
