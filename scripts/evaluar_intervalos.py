"""Cobertura real del intervalo del 90 % servido, por estación, con dos métodos:
  - móvil 365 d (Fase 4): errores de los últimos 365 días
  - estacional en cascada (Fase 7): errores de la misma época (±45 d) de los últimos 3 años
Ambos sobre lo que habría servido la API cada día (modelo v2 con previsión o respaldo), test 2023-2026.

Uso:
    python scripts/evaluar_intervalos.py
"""
import numpy as np
import pandas as pd

from gemelo.config import RAIZ
from gemelo.monitor import metricas as D
from gemelo.monitor.deriva import Monitor

ESTACION = {12: "invierno", 1: "invierno", 2: "invierno", 3: "primavera", 4: "primavera", 5: "primavera",
            6: "verano", 7: "verano", 8: "verano", 9: "otoño", 10: "otoño", 11: "otoño"}

if __name__ == "__main__":
    mon = Monitor()
    g = mon.g
    filas = []
    for zona in g.zonas:
        for h in g.horizontes:
            s = mon.serie(zona, h)  # estacional en cascada (lo que sirve la API ahora)
            w = s.loc["2023-01-01":].dropna(subset=["y", "pred", "lo", "hi"])
            # método anterior (móvil 365 d) con las mismas predicciones servidas
            q = g.manifiesto["sst"][f"sst_{zona}_h{h}"]["cuantiles_residuo_validacion"]
            movil = D.intervalos_conformales((s["y"] - s["pred"]).dropna(), h, 365, 60, (q["0.05"], q["0.95"]))
            mv = movil.reindex(w.index)
            for est, idx in w.groupby(w.index.month.map(ESTACION)).groups.items():
                a = w.loc[idx]
                filas.append({"zona": zona, "h": h, "estacion": est, "n": len(a),
                              "cob_movil": D.cobertura(a["y"], a["pred"], mv.loc[idx, "lo"], mv.loc[idx, "hi"]),
                              "cob_estacional": D.cobertura(a["y"], a["pred"], a["lo"], a["hi"]),
                              "ancho_movil": float((mv.loc[idx, "hi"] - mv.loc[idx, "lo"]).mean()),
                              "ancho_estacional": float((a["hi"] - a["lo"]).mean()),
                              "is_movil": D.interval_score(a["y"], a["pred"], mv.loc[idx, "lo"], mv.loc[idx, "hi"]),
                              "is_estacional": D.interval_score(a["y"], a["pred"], a["lo"], a["hi"])})
    t = pd.DataFrame(filas)
    t.to_csv(RAIZ / "docs" / "figuras" / "fase7" / "cobertura_intervalos_estacional.csv", index=False)
    resumen = t.groupby("estacion")[["cob_movil", "cob_estacional", "ancho_movil", "ancho_estacional",
                                      "is_movil", "is_estacional"]].mean()
    print("Media de los 8 modelos por estación (test 2023-2026):")
    print(resumen.loc[["invierno", "primavera", "verano", "otoño"]].round(3).to_string())
    print("\nPeor caso (mínima cobertura por estación):")
    print(t.groupby("estacion")[["cob_movil", "cob_estacional"]].min().round(3).to_string())
    print(f"\nGlobal: cobertura móvil {np.average(t.cob_movil, weights=t.n):.3f} · estacional "
          f"{np.average(t.cob_estacional, weights=t.n):.3f} | interval score móvil "
          f"{np.average(t.is_movil, weights=t.n):.3f} · estacional {np.average(t.is_estacional, weights=t.n):.3f}")
    gana = (t.is_estacional < t.is_movil).mean()
    print(f"El estacional tiene mejor interval score en {gana:.0%} de las combinaciones modelo × estación")
