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
from gemelo.procesado import meteo_futura as M
from gemelo.procesado.features import _doy, construir_dataset_sst, grupo_variable

DIAS_TRAMO = 400  # historia necesaria para recalcular lags (30), medias (90) y días desde lluvia (365)

DRIVERS = {
    "temp_agua": {"tipo": "suma", "unidad": "°C", "series": ["sst_laguna", "sst_costa", "thetao_costa"],
                  "insitu": ["ahora_temp_sup_media", "ahora_temp_fondo_media"],
                  "descripcion": "Episodio de agua más cálida/fría (+x °C) durante los últimos `dias`, en laguna y "
                                 "costa a la vez (y en la última campaña in situ)."},
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
_FORZAMIENTO = ("Con periodo 'futuro' (por defecto) perturba la PREVISIÓN de los próximos días (máx. 7), que usan "
                "los modelos de 1, 3 y 7 días (no el de 30 ni la hipoxia). Con 'pasado' perturba los últimos días "
                "observados: el agua de esos días no cambia, así que el efecto suele ser pequeño.")
NOTAS = {
    "temp_agua": "Episodio: el modelo ha aprendido que las anomalías tienden a revertir, así que parte del "
                 "calentamiento se disipa con el horizonte.",
    "calentamiento_base": "Método delta: supone que la dinámica del mar no cambia con el clima de fondo. El umbral "
                          "de ola de calor se mantiene en la base estándar 1991-2020.",
    "temp_aire": _FORZAMIENTO, "lluvia": _FORZAMIENTO,
    "viento": _FORZAMIENTO + " En la hipoxia no tiene efecto: el modelo v2 no usa el viento porque en v1 la "
              "relación aprendida tenía el signo contrario a la física (más viento -> más riesgo).",
    "radiacion": _FORZAMIENTO,
    "clorofila": "En la SST apenas influye (<3 % de la importancia SHAP). En la hipoxia actúa mediante la clorofila "
                 "in situ de la última campaña.",
}
NO_DISPONIBLES = {
    "nitratos": "No es una entrada de los modelos: el modelo biogeoquímico de Copernicus no cubre la laguna "
                "y la UPCT no publica nutrientes en el servidor usado.",
    "fosfatos": "Igual que nitratos.",
    "vertidos": "No hay series de vertidos/caudales; la lluvia en la cuenca es el único indicador disponible.",
}


# Drivers meteorológicos que también pueden perturbar la PREVISIÓN (días futuros) y sobre qué variables
# diarias de la previsión actúan (ver procesado.meteo_futura).
METEO_FUTURO = {"temp_aire": ["t2m", "t2max"], "lluvia": ["prec"],
                "viento": ["viento", "vientomax", "viento_u", "viento_v"], "radiacion": ["rad"]}
PERIODOS = ("pasado", "futuro", "ambos")


@dataclass
class Perturbacion:
    driver: str
    valor: float
    incertidumbre: float = 0.0
    dias: int = 30
    periodo: str | None = None   # None -> 'futuro' para la meteorología, 'pasado' para el resto

    def __post_init__(self):
        if self.periodo is None:
            self.periodo = "futuro" if self.driver in METEO_FUTURO else "pasado"
        if self.periodo not in PERIODOS:
            raise ValueError(f"periodo debe ser uno de {PERIODOS}")
        if self.periodo != "pasado" and self.driver not in METEO_FUTURO:
            raise ValueError(f"'{self.driver}' solo admite periodo 'pasado': no hay previsión de esa variable.")

    @property
    def en_pasado(self):
        return self.periodo in ("pasado", "ambos")

    @property
    def en_futuro(self):
        return self.periodo in ("futuro", "ambos")


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
    """Perturba las series OBSERVADAS (pasado reciente) de las perturbaciones con periodo pasado/ambos."""
    t = tramo.copy()
    for p, v in zip(perts, valores):
        if not p.en_pasado:
            continue
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


def fuente_prevision_perturbada(prev, fecha: pd.Timestamp, perts, valores):
    """Fuente para meteo_futura.construir con la PREVISIÓN de (fecha, fecha + dias] perturbada."""
    base = M.fuente_prevision(prev)
    cambios = [(p, v) for p, v in zip(perts, valores) if p.en_futuro]

    def valor_dia(var, punto, k):
        s = base(var, punto, k)
        aplicables = [(p, v) for p, v in cambios if var in METEO_FUTURO[p.driver]]
        if not aplicables:
            return s
        s = s.copy()
        for p, v in aplicables:
            ventana = (s.index > fecha) & (s.index <= fecha + pd.Timedelta(days=min(p.dias, 7)))
            s[ventana] = s[ventana] + v if DRIVERS[p.driver]["tipo"] == "suma" else s[ventana] * v
        return s
    return valor_dia


def desplazamiento_base(perts, valores) -> float:
    return float(sum(v for p, v in zip(perts, valores) if p.driver == "calentamiento_base"))


class MotorEscenarios:
    def __init__(self, gemelo):
        self.g = gemelo
        self.cfg = cargar_config()
        self.diario = gemelo.diario
        # Rango visto en entrenamiento, POR MODELO (hasta su `datos_hasta`: los reentrenados ven más).
        # Las variables de previsión se entrenaron con ERA5: su rango es el de meteo_futura_era5.
        from gemelo.servicio.motor import DIR_DATOS
        ruta = DIR_DATOS / "meteo_futura_era5.parquet"
        self._entreno = gemelo.ds.join(pd.read_parquet(ruta)) if ruta.exists() else gemelo.ds
        self._rangos = {}
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

    def variables_prevision(self, fecha, perts, valores) -> pd.DataFrame | None:
        """Variables prev{w}_* (una fila por muestra) desde la previsión real del día, perturbada."""
        if self.g.prev_diaria is None:
            return None
        puntos = list(self.cfg["meteo"]["puntos"])
        filas = [M.construir(fuente_prevision_perturbada(self.g.prev_diaria, fecha, perts, v), puntos, self.g.ref,
                             pd.DatetimeIndex([fecha])).iloc[0] for v in valores]
        return pd.DataFrame(filas).reset_index(drop=True)

    def rango_modelo(self, info) -> pd.DataFrame:
        hasta = self.g.datos_hasta(info)
        if hasta not in self._rangos:
            self._rangos[hasta] = self._entreno.loc[:hasta].quantile([0.0, 1.0], numeric_only=True)
        return self._rangos[hasta]

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
        # previsión meteorológica (solo la usan los modelos con previsión y si hay previsión ese día)
        P = {"esc": self.variables_prevision(fecha, perts, valores),
             "cen": self.variables_prevision(fecha, perts, central),
             "base": self.variables_prevision(fecha, [], np.zeros((1, 0)))}
        base_shift = np.array([desplazamiento_base(perts, v) for v in valores])
        resultados, alguno_con_prevision = [], False
        for h in horizontes:
            clave, m, info, usa_prev = self.g.elegir(zona, h, fecha)
            alguno_con_prevision |= usa_prev
            vars_m = info["variables"]
            if usa_prev:  # añadir las columnas de previsión a cada tabla de variables
                X_esc_h, X_cen_h, X_base_h = (pd.concat([X, P[k]], axis=1) for X, k in
                                              ((X_esc, "esc"), (X_cen, "cen"), (X_base, "base")))
            else:
                X_esc_h, X_cen_h, X_base_h = X_esc, X_cen, X_base
            res, metodo = self.g.residuos_recientes(zona, h, fecha)
            fecha_obj = fecha + pd.Timedelta(days=h)
            doy = int(_doy(pd.DatetimeIndex([fecha_obj]))[0])
            clim = float(self.g.ref.loc[doy, f"clim_{zona}"])
            umbral = float(self.g.ref.loc[doy, f"p90_obj_{zona}"])

            def sst(X, shift):
                return clim + X[f"anom_{zona}_lag0"].values + m.predict(X[vars_m]) + shift

            sst_base = float(sst(X_base_h, 0.0)[0])
            sst_esc = sst(X_esc_h, base_shift)                        # una por muestra de perturbación
            dist_base = sst_base + res                                # incertidumbre del modelo
            dist_esc = (sst_esc[:, None] + res[None, :]).ravel()      # perturbación × modelo
            dif = sst_esc - sst_base                                  # emparejada: sin error del modelo

            # Explicación del cambio: SHAP (escenario central - base), agrupado
            expl = self.g._explicador(clave)
            d_shap = expl(X_cen_h[vars_m]).values[0] - expl(X_base_h[vars_m]).values[0]
            cambio = pd.Series(d_shap, index=vars_m).groupby([grupo_variable(c) for c in vars_m]).sum()
            cambio = cambio[cambio.abs() > 1e-4].sort_values(key=np.abs, ascending=False)

            avisos = self._fuera_de_rango(X_esc_h, self.rango_modelo(info), vars_m)
            sst_max = self.sst_max_observada[zona]
            if np.quantile(dist_esc, 0.5) > sst_max:
                avisos.append({"variable": f"sst_prevista_{zona}", "valor_central": round(float(np.median(dist_esc)), 3),
                               "rango_entrenamiento": [None, round(sst_max, 3)],
                               "nota": "SST prevista por encima del máximo observado desde 1982."})
            sst_cen = float(sst(X_cen_h, desplazamiento_base(perts, [p.valor for p in perts]))[0])
            resultados.append({
                "h": h, "fecha_objetivo": str(fecha_obj.date()), "umbral_ola_calor": round(umbral, 3),
                # sst_puntual = predicción del modelo (igual que /predict). Los cuantiles añaden los errores
                # recientes, cuya mediana no es 0 si el modelo tiene sesgo reciente (deriva).
                "base": {"sst_puntual": round(sst_base, 3), **_resumen(dist_base, umbral)},
                "escenario": {"sst_puntual": round(sst_cen, 3), **_resumen(dist_esc, umbral)},
                "diferencia_sst": {k: round(float(np.quantile(dif, q)), 3) for k, q in (("p05", .05), ("p50", .5), ("p95", .95))},
                "cambio_explicado_por_grupo": [{"grupo": g, "contribucion": round(float(v), 4)} for g, v in cambio.items()],
                "desplazamiento_base_c": round(float(np.median(base_shift)), 3),
                "fuera_de_rango": avisos, "metodo_intervalo": metodo, "usa_prevision_meteo": usa_prev,
                "modelo": {"nombre": clave.split("__")[0], "version_registro": info["version_registro"],
                           "variante": "con_prevision" if usa_prev else "sin_prevision"},
            })
        avisos_n = notas(perts)
        if any(p.en_futuro for p in perts) and not alguno_con_prevision:
            avisos_n["sin_prevision"] = (f"No hay previsión meteorológica archivada para {fecha.date()}: las "
                                         "perturbaciones del FUTURO no tienen efecto en este día.")
        return {"zona": zona, "fecha_base": str(fecha.date()), "n_muestras_perturbacion": len(valores),
                "notas": avisos_n, "horizontes": resultados}

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
