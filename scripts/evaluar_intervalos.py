"""Compara la cobertura real del intervalo del 90 % en test con dos métodos:
  - cuantiles de validación fijos (Fase 3)
  - conformal con ventana móvil de 365 días (Fase 4, el que usa la API)

Se evalúa desde 2023 para que la ventana móvil contenga solo errores fuera de entrenamiento.
Uso:
    python scripts/evaluar_intervalos.py
"""
import numpy as np
import pandas as pd

from gemelo.config import RAIZ
from gemelo.servicio.motor import Gemelo

if __name__ == "__main__":
    g = Gemelo()
    filas = []
    for zona in g.zonas:
        for h in g.horizontes:
            nombre, m, info = g._modelo(zona, h)
            q = info["cuantiles_residuo_validacion"]
            ds = g.ds.loc["2023-01-01":]
            y = ds[f"y_delta_{zona}_h{h}"]
            ok = y.notna() & ds[info["variables"]].notna().all(axis=1)
            fechas = ds.index[ok]
            pred = pd.Series(m.predict(ds.loc[ok, info["variables"]]), index=fechas)
            res_todos = g.ds[f"y_delta_{zona}_h{h}"] - pd.Series(
                m.predict(g.ds[info["variables"]]), index=g.ds.index)
            dentro_fijo, dentro_movil, ancho_fijo, ancho_movil = [], [], [], []
            for f in fechas:
                real = y[f]
                lo, hi = pred[f] + q["0.05"], pred[f] + q["0.95"]
                dentro_fijo.append(lo <= real <= hi)
                ancho_fijo.append(hi - lo)
                # misma lógica causal que Gemelo.residuos_recientes, vectorizada sobre residuos precalculados
                ventana = res_todos.loc[f - pd.Timedelta(days=365 + h): f - pd.Timedelta(days=h)].dropna()
                qlo, qhi = np.quantile(ventana, [0.05, 0.95])
                dentro_movil.append(pred[f] + qlo <= real <= pred[f] + qhi)
                ancho_movil.append(qhi - qlo)
            filas.append({"zona": zona, "h": h, "n": len(fechas),
                          "cobertura_fija": np.mean(dentro_fijo), "ancho_fijo": np.mean(ancho_fijo),
                          "cobertura_movil": np.mean(dentro_movil), "ancho_movil": np.mean(ancho_movil)})
    t = pd.DataFrame(filas).round(3)
    t.to_csv(RAIZ / "docs" / "figuras" / "fase4_cobertura_intervalos.csv", index=False)
    print(t.to_string(index=False))
