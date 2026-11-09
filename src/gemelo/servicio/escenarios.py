"""Motor de escenarios "what-if" del gemelo, con incertidumbre (Monte Carlo) y aviso fuera de rango.

Idea clave: NO se tocan las variables derivadas (lags, medias, rachas...). Se perturba la SERIE DIARIA de
origen durante los últimos `dias` y se recalculan todas las variables con el mismo código del pipeline
(features.construir_dataset_sst). Así un "+2 °C de aire durante 14 días" mueve de forma coherente la
anomalía de hoy, las medias de 3 y 7 días, etc.

Dos formas de "calentar el agua", conceptualmente distintas:
  - temp_agua           episodio: el agua ha estado +x °C los últimos días. El modelo lo trata como una
                        anomalía y (habitualmente) predice que revierte.
  - calentamiento_base  cambio de fondo del clima (método delta): se desplaza la climatología +x °C. El
                        modelo no cambia sus variables (las anomalías son las mismas) y la SST prevista
                        sube x; el umbral de ola de calor (base 1991-2020) NO se mueve.

Incertidumbre:
  - de la perturbación: valor ~ Normal(valor, incertidumbre), muestreado por hipercubo latino;
  - del modelo: errores recientes (los mismos del intervalo conformal de /predict).
La DIFERENCIA escenario - base se calcula emparejada (mismo error del modelo en ambos), así que solo
refleja la respuesta del modelo a la perturbación y la incertidumbre de esta.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm, qmc

from gemelo.config import cargar_config
from gemelo.procesado.features import _doy, construir_dataset_sst, grupo_variable

DIAS_TRAMO = 400  # historia necesaria para recalcular lags (30), medias (90) y días desde lluvia (365)

DRIVERS = {
    "temp_agua": {"tipo": "suma", "unidad": "°C", "series": ["sst_laguna", "sst_costa", "thetao_costa"],
                  "insitu": ["ahora_temp_sup_media", "ahora_temp_fondo_media"],
                  "descripcion": "Episodio de agua más cálida/fría (+x °C) durante los últimos `dias`."},
    "calentamiento_base": {"tipo": "suma", "unidad": "°C", "series": [], "insitu": ["ahora_temp_sup_media",
                           "ahora_temp_fondo_media"],
                           "descripcion": "Cambio de clima de fondo (método delta): desplaza la climatología "
                                          "+x °C. `dias` no aplica."},
    "temp_aire": {"tipo": "suma", "unidad": "°C", "prefijos": ["t2m_", "t2max_"], "insitu": [],
                  "descripcion": "Aire más cálido/frío (+x °C) durante los últimos `dias`."},
    "lluvia": {"tipo": "factor", "unidad": "×", "prefijos": ["prec_"], "insitu": [],
               "descripcion": "Lluvia multiplicada por x durante los últimos `dias` (2 = el doble)."},
    "viento": {"tipo": "factor", "unidad": "×", "prefijos": ["viento_", "vientomax_"], "insitu": [],
               "descripcion": "Viento multiplicado por x durante los últimos `dias`."},
    "radiacion": {"tipo": "factor", "unidad": "×", "prefijos": ["rad_"], "insitu": [],
                  "descripcion": "Radiación solar multiplicada por x durante los últimos `dias`."},
    "clorofila": {"tipo": "factor", "unidad": "×", "series": ["logchl_laguna", "logchl_costa"], "log": True,
                  "insitu": ["ahora_chl_sup_media", "ahora_chl_fondo_media"],
                  "descripcion": "Clorofila multiplicada por x (satélite durante `dias` y última campaña in situ)."},
}
_FORZAMIENTO_PASADO = ("Perturba el forzamiento del PASADO RECIENTE, no una previsión: el agua observada esos "
                       "días no cambia, así que el efecto suele ser pequeño. Una ola de calor atmosférica FUTURA "
                       "requeriría un modelo con previsión meteorológica como entrada (mejora de la Fase 7).")
NOTAS = {
    "temp_agua": "Episodio: el modelo ha aprendido que las anomalías tienden a revertir, así que parte del "
                 "calentamiento se disipa con el horizonte.",
    "calentamiento_base": "Método delta: supone que la dinámica del mar no cambia con el clima de fondo. El umbral "
                          "de ola de calor se mantiene en la base estándar 1991-2020.",
    "temp_aire": _FORZAMIENTO_PASADO, "lluvia": _FORZAMIENTO_PASADO,
    "viento": _FORZAMIENTO_PASADO + " En la hipoxia no tiene efecto: el modelo v2 no usa el viento porque en v1 la "
              "relación aprendida tenía el signo contrario a la física (más viento -> más riesgo).",
    "radiacion": _FORZAMIENTO_PASADO,
    "clorofila": "En la SST apenas influye (<3 % de la importancia SHAP). En la hipoxia actúa mediante la clorofila "
                 "in situ de la última campaña.",
}
NO_DISPONIBLES = {
    "nitratos": "No es una entrada de los modelos: el modelo biogeoquímico de Copernicus no cubre la laguna "
                "y la UPCT no publica nutrientes en el servidor usado.",
    "fosfatos": "Igual que nitratos.",
    "vertidos": "No hay series de vertidos/caudales; la lluvia en la cuenca es el único indicador disponible.",
}


@dataclass
class Perturbacion:
    driver: str
    valor: float
    incertidumbre: float = 0.0
    dias: int = 30


def columnas_driver(driver: str, columnas) -> list[str]:
    d = DRIVERS[driver]
    if "series" in d:
        return [c for c in d["series"] if c in columnas]
    return [c for c in columnas if any(c.startswith(p) for p in d["prefijos"])]


def muestrear(perts: list[Perturbacion], n: int, semilla: int) -> np.ndarray:
    """Matriz (n, n_perts) de valores. Hipercubo latino: cubre la distribución con pocas muestras."""
    if not perts:
        return np.zeros((1, 0))
    if all(p.incertidumbre == 0 for p in perts):
        return np.array([[p.valor for p in perts]])
    u = qmc.LatinHypercube(d=len(perts), seed=semilla).random(n)
    vals = np.column_stack([norm.ppf(u[:, j], p.valor, max(p.incertidumbre, 1e-12)) for j, p in enumerate(perts)])
    for j, p in enumerate(perts):
        if DRIVERS[p.driver]["tipo"] == "factor":
            vals[:, j] = np.clip(vals[:, j], 0.0, None)  # un factor no puede ser negativo
    return vals


def perturbar_series(tramo: pd.DataFrame, fecha: pd.Timestamp, perts, valores) -> pd.DataFrame:
    t = tramo.copy()
    for p, v in zip(perts, valores):
        d = DRIVERS[p.driver]
        cols = columnas_driver(p.driver, t.columns)
        if not cols:
            continue
        ventana = t.index > fecha - pd.Timedelta(days=p.dias)
        if d["tipo"] == "suma":
            t.loc[ventana, cols] += v
        elif d.get("log"):
            t.loc[ventana, cols] += np.log10(max(v, 1e-6))  # factor sobre clorofila = suma en log10
        else:
            t.loc[ventana, cols] *= v
    return t


def desplazamiento_base(perts, valores) -> float:
    return float(sum(v for p, v in zip(perts, valores) if p.driver == "calentamiento_base"))


class MotorEscenarios:
    def __init__(self, gemelo):
        self.g = gemelo
        self.cfg = cargar_config()
        self.diario = gemelo.diario
        # Rango visto en entrenamiento (train + validación: lo que vieron los modelos finales)
        fin = gemelo.fin_entrenamiento_sst
        entreno = gemelo.ds.loc[:fin]
        self.rango_sst = entreno.quantile([0.0, 1.0], numeric_only=True)
        hip = gemelo.ds_hip[gemelo.ds_hip["y_hipoxia"].notna()]
        self.rango_hip = hip.quantile([0.0, 1.0], numeric_only=True)
        self.sst_max_observada = {z: float(gemelo.diario[f"sst_{z}"].max()) for z in gemelo.zonas}

    # ------------------------------------------------------------------ variables
    def variables_escenario(self, fecha, perts, valores) -> pd.DataFrame:
        """Filas de variables (una por muestra) recalculadas desde las series perturbadas."""
        tramo = self.diario.loc[fecha - pd.Timedelta(days=DIAS_TRAMO):fecha]
        filas = []
        for v in valores:
            t = perturbar_series(tramo, fecha, perts, v)
            filas.append(construir_dataset_sst(t, self.cfg, self.g.ref).loc[fecha])
        return pd.DataFrame(filas).reset_index(drop=True)

    def _fuera_de_rango(self, X: pd.DataFrame, rango: pd.DataFrame, variables) -> list[dict]:
        avisos = []
        for v in variables:
            lo, hi = float(rango.loc[0.0, v]), float(rango.loc[1.0, v])
            frac = float(((X[v] < lo) | (X[v] > hi)).mean())
            if frac > 0:
                avisos.append({"variable": v, "valor_central": round(float(X[v].median()), 3),
                               "rango_entrenamiento": [round(lo, 3), round(hi, 3)],
                               "fraccion_muestras_fuera": round(frac, 3)})
        return avisos

    # ------------------------------------------------------------------ SST
    def simular_sst(self, zona, horizontes, fecha, perts, n_muestras=30, semilla=42) -> dict:
        fecha = self.g._fecha(fecha)
        valores = muestrear(perts, n_muestras, semilla)
        central = np.array([[p.valor for p in perts]])
        X_esc = self.variables_escenario(fecha, perts, valores)
        X_cen = self.variables_escenario(fecha, perts, central)
        X_base = self.variables_escenario(fecha, [], np.zeros((1, 0)))
        base_shift = np.array([desplazamiento_base(perts, v) for v in valores])
        resultados = []
        for h in horizontes:
            nombre, m, info = self.g._modelo(zona, h)
            vars_m = info["variables"]
            res, metodo = self.g.residuos_recientes(zona, h, fecha)
            fecha_obj = fecha + pd.Timedelta(days=h)
            doy = int(_doy(pd.DatetimeIndex([fecha_obj]))[0])
            clim = float(self.g.ref.loc[doy, f"clim_{zona}"])
            umbral = float(self.g.ref.loc[doy, f"p90_obj_{zona}"])

            def sst(X, shift):
                return clim + X[f"anom_{zona}_lag0"].values + m.predict(X[vars_m]) + shift

            sst_base = float(sst(X_base, 0.0)[0])
            sst_esc = sst(X_esc, base_shift)                          # una por muestra de perturbación
            dist_base = sst_base + res                                # incertidumbre del modelo
            dist_esc = (sst_esc[:, None] + res[None, :]).ravel()      # perturbación × modelo
            dif = sst_esc - sst_base                                  # emparejada: sin error del modelo

            # Explicación del cambio: SHAP (escenario central - base), agrupado
            expl = self.g._explicador(nombre)
            d_shap = expl(X_cen[vars_m]).values[0] - expl(X_base[vars_m]).values[0]
            cambio = pd.Series(d_shap, index=vars_m).groupby([grupo_variable(c) for c in vars_m]).sum()
            cambio = cambio[cambio.abs() > 1e-4].sort_values(key=np.abs, ascending=False)

            avisos = self._fuera_de_rango(X_esc, self.rango_sst, vars_m)
            sst_max = self.sst_max_observada[zona]
            if np.quantile(dist_esc, 0.5) > sst_max:
                avisos.append({"variable": f"sst_prevista_{zona}", "valor_central": round(float(np.median(dist_esc)), 3),
                               "rango_entrenamiento": [None, round(sst_max, 3)],
                               "nota": "SST prevista por encima del máximo observado desde 1982."})
            sst_cen = float(sst(X_cen, desplazamiento_base(perts, [p.valor for p in perts]))[0])
            resultados.append({
                "h": h, "fecha_objetivo": str(fecha_obj.date()), "umbral_ola_calor": round(umbral, 3),
                # sst_puntual = predicción del modelo (igual que /predict). Los cuantiles añaden los errores
                # recientes, cuya mediana no es 0 si el modelo tiene sesgo reciente (deriva).
                "base": {"sst_puntual": round(sst_base, 3), **_resumen(dist_base, umbral)},
                "escenario": {"sst_puntual": round(sst_cen, 3), **_resumen(dist_esc, umbral)},
                "diferencia_sst": {k: round(float(np.quantile(dif, q)), 3) for k, q in (("p05", .05), ("p50", .5), ("p95", .95))},
                "cambio_explicado_por_grupo": [{"grupo": g, "contribucion": round(float(v), 4)} for g, v in cambio.items()],
                "desplazamiento_base_c": round(float(np.median(base_shift)), 3),
                "fuera_de_rango": avisos, "metodo_intervalo": metodo,
                "modelo": {"nombre": nombre, "version_registro": info["version_registro"]},
            })
        return {"zona": zona, "fecha_base": str(fecha.date()), "n_muestras_perturbacion": len(valores),
                "notas": notas(perts), "horizontes": resultados}

    # ------------------------------------------------------------------ hipoxia
    def simular_hipoxia(self, fecha_campana, perts, n_muestras=30, semilla=42, dias_hasta_siguiente=7) -> dict:
        g = self.g
        info = g.manifiesto["hipoxia"]
        fc = g.campanas.max() if fecha_campana is None else pd.Timestamp(fecha_campana).normalize()
        g.predecir_hipoxia(fc)  # valida la fecha (lanza ErrorGemelo con mensaje claro)
        variables = info["variables"]
        valores = muestrear(perts, n_muestras, semilla)

        def filas(vals_matriz):
            diarias = self.variables_escenario(fc, perts, vals_matriz)
            out = []
            for i, v in enumerate(vals_matriz):
                x = g.ds_hip.loc[fc, variables].copy()
                for c in variables:                  # variables diarias (lluvia, viento...) recalculadas
                    if c in diarias.columns:
                        x[c] = diarias.loc[i, c]
                for p, val in zip(perts, v):         # variables in situ de la campaña
                    for c in DRIVERS[p.driver]["insitu"]:
                        if c in x.index:
                            x[c] = x[c] * val if DRIVERS[p.driver]["tipo"] == "factor" else x[c] + val
                x["dias_hasta_siguiente"] = dias_hasta_siguiente
                out.append(x)
            return pd.DataFrame(out)[variables]

        # base = las mismas perturbaciones con valor neutro (suma 0, factor 1) = estado real
        neutros = np.array([[1.0 if DRIVERS[p.driver]["tipo"] == "factor" else 0.0 for p in perts]])
        X_base = filas(neutros)
        X_esc = filas(valores)
        X_cen = filas(np.array([[p.valor for p in perts]]))
        p_base = float(g.hipoxia.predict_proba(X_base)[0, 1])
        p_esc = g.hipoxia.predict_proba(X_esc)[:, 1]

        imp, esc, log = g.hipoxia[0], g.hipoxia[1], g.hipoxia[-1]
        z = lambda X: esc.transform(imp.transform(X))[0]
        dcontrib = pd.Series(log.coef_[0] * (z(X_cen) - z(X_base)), index=variables)
        dcontrib = dcontrib[dcontrib.abs() > 1e-4].sort_values(key=np.abs, ascending=False)
        return {
            "fecha_campana": str(fc.date()), "umbral_alerta": info["umbral_alerta"], "notas": notas(perts),
            "base": {"prob_hipoxia": round(p_base, 3), "alerta": p_base >= info["umbral_alerta"]},
            "escenario": {"prob_hipoxia": {k: round(float(np.quantile(p_esc, q)), 3)
                                           for k, q in (("p05", .05), ("p50", .5), ("p95", .95))},
                          "prob_alerta": round(float(np.mean(p_esc >= info["umbral_alerta"])), 3)},
            "cambio_explicado_log_odds": [{"variable": v, "contribucion": round(float(c), 3)} for v, c in dcontrib.items()],
            "fuera_de_rango": self._fuera_de_rango(X_esc, self.rango_hip, variables),
        }


def notas(perts) -> dict:
    return {p.driver: NOTAS[p.driver] for p in perts}


def _resumen(dist: np.ndarray, umbral: float) -> dict:
    q = np.quantile(dist, [0.05, 0.5, 0.95])
    return {"sst_p05": round(float(q[0]), 3), "sst_p50": round(float(q[1]), 3), "sst_p95": round(float(q[2]), 3),
            "prob_ola_calor": round(float(np.mean(dist > umbral)), 3)}
