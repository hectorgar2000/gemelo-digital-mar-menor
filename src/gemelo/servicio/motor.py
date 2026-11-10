"""Motor del gemelo: carga los modelos exportados y el estado del mar, y responde preguntas.

No depende de MLflow ni de HTTP: lo usan la API (Fase 4), el motor de escenarios (Fase 5) y el
dashboard (Fase 6).

Reconstrucción de la SST prevista (el modelo predice el CAMBIO de anomalía, delta):
    SST(t+h) = climatología(t+h) + anomalía(t) + delta
"""
import json
from functools import cached_property
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import xgboost as xgb

from gemelo.config import RAIZ, cargar_config
from gemelo.procesado.features import _doy, grupo_variable
from gemelo.procesado.meteo_futura import desde_tabla


def es_prevision(variable: str) -> bool:
    """Variables prev{w}_* de meteorología prevista (Fase 7)."""
    return variable.startswith("prev") and variable[4:5].isdigit()

DIR_MODELOS = RAIZ / "models" / "servicio"
DIR_DATOS = RAIZ / "data" / "processed"
VENTANA_CONFORMAL_DIAS = 365   # residuos recientes para el intervalo (se adapta a la deriva)
MIN_RESIDUOS = 60
DIAS_SIGUIENTE_CAMPANA = 7     # supuesto por defecto para la hipoxia (mediana real: 7 días)


class ErrorGemelo(ValueError):
    """Petición válida en forma pero imposible de responder (fecha sin datos, modelo inexistente...)."""


class Gemelo:
    def __init__(self, dir_modelos: Path = DIR_MODELOS, dir_datos: Path = DIR_DATOS):
        self.manifiesto = json.loads((dir_modelos / "servicio.json").read_text(encoding="utf-8"))

        def cargar(fichero):
            m = xgb.XGBRegressor()
            m.load_model(dir_modelos / fichero)
            return m

        # Por modelo: el principal (@produccion) y, si usa previsión, su respaldo sin previsión.
        self.modelos = {n: cargar(i["fichero"]) for n, i in self.manifiesto["sst"].items()}
        self.respaldos = {n: cargar(i["respaldo"]["fichero"]) for n, i in self.manifiesto["sst"].items()
                          if i.get("respaldo")}
        self.hipoxia = joblib.load(dir_modelos / self.manifiesto["hipoxia"]["fichero"])
        self.ds = pd.read_parquet(dir_datos / "dataset_sst.parquet")
        self.ds_hip = pd.read_parquet(dir_datos / "dataset_hipoxia.parquet")
        self.ref = pd.read_parquet(dir_datos / "referencias_sst.parquet")
        self.diario = pd.read_parquet(dir_datos / "series_diarias.parquet")  # series de origen (escenarios)
        # Previsión meteorológica real (Fase 7): variables prev{w}_* por día base y previsión diaria por
        # antelación (para escenarios). Opcionales: sin ellas, se usan los modelos de respaldo.
        ruta_prev = dir_datos / "meteo_futura_prevision.parquet"
        self.f_prev = pd.read_parquet(ruta_prev) if ruta_prev.exists() else None
        ruta_diaria = dir_datos / "prevision_diaria.parquet"
        self.prev_diaria = desde_tabla(pd.read_parquet(ruta_diaria)) if ruta_diaria.exists() else None
        self._explicadores = {}
        # Los modelos finales se entrenaron con train + validación: para esas fechas, una
        # "predicción" del pasado no es honesta (el modelo ya vio la respuesta).
        part = cargar_config()["particion"]
        self.fin_entrenamiento_sst = pd.Timestamp(part["validacion"][1])

    # ------------------------------------------------------------------ utilidades
    @property
    def zonas(self):
        return sorted({i["zona"] for i in self.manifiesto["sst"].values()})

    @property
    def horizontes(self):
        return sorted({i["h"] for i in self.manifiesto["sst"].values()})

    @cached_property
    def fechas_validas(self) -> pd.DatetimeIndex:
        """Días con todas las variables del estado del mar disponibles (sin contar la previsión)."""
        variables = sorted({v for i in self.manifiesto["sst"].values() for v in i["variables"] if not es_prevision(v)})
        return self.ds.index[self.ds[variables].notna().all(axis=1)]

    def ultima_fecha(self) -> pd.Timestamp:
        return self.fechas_validas.max()

    def _fecha(self, fecha) -> pd.Timestamp:
        f = self.ultima_fecha() if fecha is None else pd.Timestamp(fecha).normalize()
        if f not in self.fechas_validas:
            raise ErrorGemelo(f"Sin datos completos para {f.date()}. Rango disponible: "
                              f"{self.fechas_validas.min().date()} a {self.fechas_validas.max().date()}.")
        return f

    def _modelo(self, zona, h):
        nombre = f"sst_{zona}_h{h}"
        if nombre not in self.modelos:
            raise ErrorGemelo(f"No hay modelo para zona={zona}, h={h}. Zonas: {self.zonas}; horizontes: {self.horizontes}.")
        return nombre, self.modelos[nombre], self.manifiesto["sst"][nombre]

    def hay_prevision(self, fecha, ventana: int) -> bool:
        if self.f_prev is None or fecha not in self.f_prev.index:
            return False
        cols = [c for c in self.f_prev.columns if c.startswith(f"prev{ventana}_")]
        return bool(self.f_prev.loc[fecha, cols].notna().all())

    def elegir(self, zona, h, fecha):
        """Modelo que se usa ese día: el principal si no necesita previsión o si la hay completa;
        si no, su respaldo sin previsión. Devuelve (clave, modelo, ficha, usa_prevision)."""
        nombre, m, info = self._modelo(zona, h)
        if info.get("usa_prevision") and not self.hay_prevision(fecha, info["ventana_meteo"]):
            if nombre not in self.respaldos:
                raise ErrorGemelo(f"No hay previsión meteorológica para {fecha.date()} ni modelo de respaldo.")
            return f"{nombre}__sin_prevision", self.respaldos[nombre], info["respaldo"], False
        return nombre, m, info, bool(info.get("usa_prevision"))

    def matriz(self, fechas, variables) -> pd.DataFrame:
        """Variables de entrada para esas fechas: estado del mar (dataset_sst) + previsión si el modelo la usa."""
        X = self.ds.loc[fechas, [v for v in variables if not es_prevision(v)]]
        prev = [v for v in variables if es_prevision(v)]
        if prev:
            X = X.join(self.f_prev[prev])
        return X[variables]

    def fila(self, fecha, variables) -> pd.DataFrame:
        return self.matriz([fecha], variables)

    # ------------------------------------------------------------------ lectura del estado
    def serie_sst(self, zona: str, desde=None, hasta=None, max_dias: int = 1500) -> pd.DataFrame:
        """SST diaria observada con su climatología, umbral de ola de calor y anomalía."""
        if zona not in self.zonas:
            raise ErrorGemelo(f"Zona desconocida '{zona}'. Zonas: {self.zonas}.")
        hasta = self.ultima_fecha() if hasta is None else pd.Timestamp(hasta)
        desde = hasta - pd.Timedelta(days=365) if desde is None else pd.Timestamp(desde)
        if (hasta - desde).days > max_dias:
            raise ErrorGemelo(f"Rango demasiado largo: máximo {max_dias} días por petición.")
        s = self.diario.loc[desde:hasta, f"sst_{zona}"].dropna()
        doy = _doy(s.index)
        clim = self.ref[f"clim_{zona}"].reindex(doy).values
        return pd.DataFrame({"sst": s.values, "climatologia": clim,
                             "umbral_ola_calor": self.ref[f"p90_obj_{zona}"].reindex(doy).values,
                             "anomalia": s.values - clim}, index=s.index)

    def estado(self, fecha=None) -> dict:
        """Foto del gemelo un día: SST, anomalía y si hay ola de calor en curso (causal: solo días <= fecha)."""
        fecha = self._fecha(fecha)
        cfg = cargar_config()
        zonas = {}
        for z in self.zonas:
            s = self.serie_sst(z, fecha - pd.Timedelta(days=120), fecha)
            sobre = (s["sst"] > s["umbral_ola_calor"]).values[::-1]
            racha = int(np.argmin(sobre)) if not sobre.all() else len(sobre)
            hoy = s.iloc[-1]
            zonas[z] = {"sst": round(float(hoy["sst"]), 2), "anomalia": round(float(hoy["anomalia"]), 2),
                        "umbral_ola_calor": round(float(hoy["umbral_ola_calor"]), 2),
                        "dias_sobre_umbral": racha,
                        "ola_calor_en_curso": racha >= cfg["umbrales"]["mhw_dias_min"],
                        "caja": cfg["subzonas"][z]}
        ult = self.campanas[self.campanas <= fecha]
        campana = None
        if len(ult):
            c = self.ds_hip.loc[ult.max()]
            campana = {"fecha": str(ult.max().date()), "pct_fondo_hipoxia": round(float(c["ahora_pct_hipoxia"]), 1),
                       "o2_fondo_media": round(float(c["ahora_o2_fondo_media"]), 2),
                       "temp_fondo_media": round(float(c["ahora_temp_fondo_media"]), 2),
                       "dias_desde_campana": int((fecha - ult.max()).days)}
        return {"fecha": str(fecha.date()), "zonas": zonas, "ultima_campana_hipoxia": campana,
                # la celda ERA5 real (no el punto pedido): es lo que representan los datos
                "puntos_meteo": {k: p.get("celda", {"lat": p["lat"], "lon": p["lon"]})
                                 for k, p in cfg["meteo"]["puntos"].items()},
                "rango_fechas": [str(self.fechas_validas.min().date()), str(self.fechas_validas.max().date())]}

    def historial_campanas(self) -> pd.DataFrame:
        cols = {"ahora_pct_hipoxia": "pct_fondo_hipoxia", "ahora_o2_fondo_media": "o2_fondo_media",
                "ahora_o2_fondo_min": "o2_fondo_min", "ahora_temp_fondo_media": "temp_fondo_media",
                "ahora_sal_fondo_media": "sal_fondo_media"}
        return self.ds_hip[list(cols)].rename(columns=cols)

    # ------------------------------------------------------------------ intervalo
    def residuos_recientes(self, zona, h, fecha) -> tuple[np.ndarray, str]:
        """Residuos (real - predicho) de los últimos VENTANA días cuyo resultado YA se conocía en `fecha`
        (t + h <= fecha): conformal causal que se adapta a la deriva. Si no hay suficientes, usa los
        cuantiles de validación guardados al entrenar. Usa el MISMO modelo y la misma fuente de variables
        (con previsión real, si es el caso) que la predicción de ese día."""
        _, m, info, _ = self.elegir(zona, h, fecha)
        ini = fecha - pd.Timedelta(days=VENTANA_CONFORMAL_DIAS + h)
        fin = fecha - pd.Timedelta(days=h)
        fechas = self.ds.loc[ini:fin].index
        X = self.matriz(fechas, info["variables"])
        y = self.ds.loc[fechas, f"y_delta_{zona}_h{h}"]
        ok = y.notna() & X.notna().all(axis=1)
        if ok.sum() >= MIN_RESIDUOS:
            res = y[ok].values - m.predict(X[ok])
            return res, f"conformal_ventana_{VENTANA_CONFORMAL_DIAS}d"
        q = info["cuantiles_residuo_validacion"]
        return np.array([q["0.05"], q["0.5"], q["0.95"]]), "cuantiles_validacion"

    # ------------------------------------------------------------------ predicción SST
    def predecir(self, zona: str, h: int, fecha=None, X: pd.DataFrame | None = None) -> dict:
        """Predicción de SST a h días desde `fecha`. Con X se puede pasar una fila modificada
        (lo usará el motor de escenarios); por defecto, el estado real del gemelo ese día."""
        fecha = self._fecha(fecha)
        clave, m, info, usa_prev = self.elegir(zona, h, fecha)
        X = self.fila(fecha, info["variables"]) if X is None else X[info["variables"]]
        delta = float(m.predict(X)[0])

        fecha_obj = fecha + pd.Timedelta(days=h)
        doy_obj = int(_doy(pd.DatetimeIndex([fecha_obj]))[0])
        clim_obj = float(self.ref.loc[doy_obj, f"clim_{zona}"])
        umbral = float(self.ref.loc[doy_obj, f"p90_obj_{zona}"])
        anom_hoy = float(X[f"anom_{zona}_lag0"].iloc[0])
        sst_prev = clim_obj + anom_hoy + delta

        res, metodo = self.residuos_recientes(zona, h, fecha)
        lo, hi = np.quantile(res, [0.05, 0.95])
        real = self.ds.get(f"y_sst_{zona}_h{h}", pd.Series(dtype=float)).get(fecha, np.nan)
        return {
            "zona": zona, "h": h, "fecha_base": str(fecha.date()), "fecha_objetivo": str(fecha_obj.date()),
            "sst_hoy": round(float(X[f"sst_{zona}"].iloc[0]), 3),
            "sst_prevista": round(sst_prev, 3),
            "intervalo_90": [round(sst_prev + lo, 3), round(sst_prev + hi, 3)],
            "anomalia_hoy": round(anom_hoy, 3), "cambio_anomalia_previsto": round(delta, 3),
            "umbral_ola_calor": round(umbral, 3),
            "ola_calor_prevista": bool(sst_prev > umbral),
            # P(SST > umbral) usando la distribución empírica de errores recientes
            "prob_superar_umbral": round(float(np.mean(sst_prev + res > umbral)), 3),
            "metodo_intervalo": metodo,
            "sst_real": None if pd.isna(real) else round(float(real), 3),  # solo si ya se conoce (backtesting)
            "visto_en_entrenamiento": bool(fecha_obj <= self.fin_entrenamiento_sst),
            "usa_prevision_meteo": usa_prev,
            "modelo": {"nombre": clave.split("__")[0], "version_registro": info["version_registro"],
                       "variante": "con_prevision" if usa_prev else "sin_prevision"},
        }

    # ------------------------------------------------------------------ explicación SST
    def _explicador(self, clave):
        """Explicador SHAP por modelo (clave = nombre o nombre__sin_prevision para el respaldo)."""
        if clave not in self._explicadores:
            nombre, _, variante = clave.partition("__")
            modelo = self.respaldos[nombre] if variante else self.modelos[nombre]
            self._explicadores[clave] = shap.TreeExplainer(modelo)
        return self._explicadores[clave]

    def explicar(self, zona: str, h: int, fecha=None, top: int = 10, X: pd.DataFrame | None = None) -> dict:
        fecha = self._fecha(fecha)
        clave, m, info, usa_prev = self.elegir(zona, h, fecha)
        X = self.fila(fecha, info["variables"]) if X is None else X[info["variables"]]
        sv = self._explicador(clave)(X)
        contrib = pd.Series(sv.values[0], index=info["variables"])
        grupos = contrib.groupby([grupo_variable(c) for c in contrib.index]).sum()
        grupos = grupos.reindex(grupos.abs().sort_values(ascending=False).index)
        principales = contrib.reindex(contrib.abs().sort_values(ascending=False).index[:top])
        return {
            "zona": zona, "h": h, "fecha_base": str(fecha.date()), "usa_prevision_meteo": usa_prev,
            "que_se_explica": "cambio de anomalía de SST previsto (°C) respecto a hoy",
            "valor_base": round(float(np.ravel(sv.base_values)[0]), 4),
            "prediccion": round(float(m.predict(X)[0]), 4),
            "grupos": [{"grupo": g, "contribucion": round(float(v), 4)} for g, v in grupos.items()],
            "variables": [{"variable": v, "valor": round(float(X[v].iloc[0]), 4), "grupo": grupo_variable(v),
                           "contribucion": round(float(c), 4)} for v, c in principales.items()],
            "nota": "contribución > 0: empuja a calentar más de lo que diría la persistencia; < 0: a enfriar. "
                    "Explica lo que usa el modelo, no una relación causal.",
        }

    # ------------------------------------------------------------------ hipoxia
    @property
    def campanas(self) -> pd.DatetimeIndex:
        return self.ds_hip.index

    def predecir_hipoxia(self, fecha=None, dias_hasta_siguiente: int = DIAS_SIGUIENTE_CAMPANA) -> dict:
        info = self.manifiesto["hipoxia"]
        fecha = self.campanas.max() if fecha is None else pd.Timestamp(fecha).normalize()
        if fecha not in self.campanas:
            cercana = self.campanas[np.argmin(np.abs((self.campanas - fecha).days))]
            raise ErrorGemelo(f"No hay campaña de muestreo el {fecha.date()} (la más cercana: {cercana.date()}). "
                              f"Rango: {self.campanas.min().date()} a {self.campanas.max().date()}.")
        X = self.ds_hip.loc[[fecha], info["variables"]].copy()
        X["dias_hasta_siguiente"] = dias_hasta_siguiente
        prob = float(self.hipoxia.predict_proba(X)[0, 1])

        # Explicación exacta de la logística: contribución = coeficiente × valor estandarizado (log-odds)
        imp, esc, log = self.hipoxia[0], self.hipoxia[1], self.hipoxia[-1]
        z = esc.transform(imp.transform(X))[0]
        contrib = pd.Series(log.coef_[0] * z, index=info["variables"]).sort_values(key=np.abs, ascending=False)
        pct_ahora = float(self.ds_hip.loc[fecha, "ahora_pct_hipoxia"])
        real = self.ds_hip.loc[fecha, "y_hipoxia"]
        return {
            "fecha_campana": str(fecha.date()), "dias_hasta_siguiente": dias_hasta_siguiente,
            "prob_hipoxia_siguiente": round(prob, 3),
            "alerta_temprana": bool(prob >= info["umbral_alerta"]),
            "hipoxia_actual": bool(pct_ahora > info["umbral_pct_hipoxia"]),
            "pct_fondo_hipoxia_actual": round(pct_ahora, 2),
            "contribuciones_log_odds": [{"variable": v, "valor": round(float(X[v].iloc[0]), 3),
                                         "contribucion": round(float(c), 3)} for v, c in contrib.items()],
            "hipoxia_real_siguiente": None if pd.isna(real) else bool(real),
            # la logística final se entrenó con TODAS las campañas que tienen siguiente
            "visto_en_entrenamiento": bool(pd.notna(real)),
            "modelo": {"nombre": "hipoxia_logistica", "version_registro": info["version_registro"]},
            "nota": f"Logística como alerta temprana (validación por años: recall "
                    f"{info['metricas_loyo'].get('logistica_recall', float('nan')):.2f}, precisión "
                    f"{info['metricas_loyo'].get('logistica_precision', float('nan')):.2f}). Interpretar junto a "
                    f"'hipoxia_actual', que es más precisa pero no anticipa inicios.",
        }
