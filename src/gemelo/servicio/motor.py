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
        self.modelos = {}
        for nombre, info in self.manifiesto["sst"].items():
            m = xgb.XGBRegressor()
            m.load_model(dir_modelos / info["fichero"])
            self.modelos[nombre] = m
        self.hipoxia = joblib.load(dir_modelos / self.manifiesto["hipoxia"]["fichero"])
        self.ds = pd.read_parquet(dir_datos / "dataset_sst.parquet")
        self.ds_hip = pd.read_parquet(dir_datos / "dataset_hipoxia.parquet")
        self.ref = pd.read_parquet(dir_datos / "referencias_sst.parquet")
        self.diario = pd.read_parquet(dir_datos / "series_diarias.parquet")  # series de origen (escenarios)
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
        """Días con todas las variables de entrada disponibles (en ambas zonas)."""
        info = next(iter(self.manifiesto["sst"].values()))
        return self.ds.index[self.ds[info["variables"]].notna().all(axis=1)]

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

    def fila(self, fecha, variables) -> pd.DataFrame:
        return self.ds.loc[[fecha], variables]

    # ------------------------------------------------------------------ intervalo
    def residuos_recientes(self, zona, h, fecha) -> tuple[np.ndarray, str]:
        """Residuos (real - predicho) de los últimos VENTANA días cuyo resultado YA se conocía en `fecha`
        (t + h <= fecha): conformal causal que se adapta a la deriva. Si no hay suficientes, usa los
        cuantiles de validación guardados al entrenar."""
        nombre, m, info = self._modelo(zona, h)
        ini = fecha - pd.Timedelta(days=VENTANA_CONFORMAL_DIAS + h)
        fin = fecha - pd.Timedelta(days=h)
        tramo = self.ds.loc[ini:fin]
        y = tramo[f"y_delta_{zona}_h{h}"]
        ok = y.notna() & tramo[info["variables"]].notna().all(axis=1)
        if ok.sum() >= MIN_RESIDUOS:
            res = y[ok].values - m.predict(tramo.loc[ok, info["variables"]])
            return res, f"conformal_ventana_{VENTANA_CONFORMAL_DIAS}d"
        q = info["cuantiles_residuo_validacion"]
        return np.array([q["0.05"], q["0.5"], q["0.95"]]), "cuantiles_validacion"

    # ------------------------------------------------------------------ predicción SST
    def predecir(self, zona: str, h: int, fecha=None, X: pd.DataFrame | None = None) -> dict:
        """Predicción de SST a h días desde `fecha`. Con X se puede pasar una fila modificada
        (lo usará el motor de escenarios); por defecto, el estado real del gemelo ese día."""
        fecha = self._fecha(fecha)
        nombre, m, info = self._modelo(zona, h)
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
            "modelo": {"nombre": nombre, "version_registro": info["version_registro"]},
        }

    # ------------------------------------------------------------------ explicación SST
    def _explicador(self, nombre):
        if nombre not in self._explicadores:
            self._explicadores[nombre] = shap.TreeExplainer(self.modelos[nombre])
        return self._explicadores[nombre]

    def explicar(self, zona: str, h: int, fecha=None, top: int = 10, X: pd.DataFrame | None = None) -> dict:
        fecha = self._fecha(fecha)
        nombre, m, info = self._modelo(zona, h)
        X = self.fila(fecha, info["variables"]) if X is None else X[info["variables"]]
        sv = self._explicador(nombre)(X)
        contrib = pd.Series(sv.values[0], index=info["variables"])
        grupos = contrib.groupby([grupo_variable(c) for c in contrib.index]).sum()
        grupos = grupos.reindex(grupos.abs().sort_values(ascending=False).index)
        principales = contrib.reindex(contrib.abs().sort_values(ascending=False).index[:top])
        return {
            "zona": zona, "h": h, "fecha_base": str(fecha.date()),
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
