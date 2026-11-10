"""Monitor de deriva de los modelos en servicio.

Para una fecha de control F (lo que "sabría" el sistema ese día) y cada modelo de SST:
  - ENTRADAS: las variables de los últimos `ventana_dias` frente a las del entrenamiento EN LA MISMA ÉPOCA DEL
    AÑO (PSI de las variables más importantes) y fracción de días fuera del rango visto.
  - RENDIMIENTO: en los días cuya verdad ya se conoce (t + h <= F), error frente a la persistencia y cobertura
    real del intervalo del 90 % tal y como lo sirve la API (conformal móvil, mismo modelo/variante).
Además, deriva del sesgo de la previsión meteorológica y estado de los datos de hipoxia.
Reproduce lo que habría servido la API cada día (v2 con previsión si la había; si no, el respaldo).
"""
import json

import numpy as np
import pandas as pd

from gemelo.config import cargar_config
from gemelo.monitor import metricas as D
from gemelo.servicio.motor import DIR_DATOS, Gemelo

DESDE_MONITOR = "2021-06-01"  # los intervalos solo se reconstruyen desde aquí (el histórico empieza en 2022)

ORDEN = {"OK": 0, "AVISO": 1, "DERIVA": 2, "SIN_DATOS": 1}


class Monitor:
    def __init__(self, gemelo: Gemelo | None = None):
        self.g = gemelo or Gemelo()
        self.cfg = cargar_config()
        self.cm = self.cfg["monitor"]
        # Referencia = lo que vieron los modelos al entrenar (hasta fin de validación); las variables de
        # previsión se entrenaron con ERA5, así que su referencia es la tabla ERA5.
        total = self.g.ds
        ruta = DIR_DATOS / "meteo_futura_era5.parquet"
        if ruta.exists():
            total = total.join(pd.read_parquet(ruta))
        self._ref_total = total
        self.ref = total.loc[:self.g.fin_entrenamiento_sst]   # referencia base (calibración)
        self._rangos, self._series = {}, {}

    def referencia(self, info) -> pd.DataFrame:
        """Lo que vio ESE modelo al entrenar (los reentrenados ven datos más recientes)."""
        return self._ref_total.loc[:self.g.datos_hasta(info)]

    def rango(self, info):
        clave = self.g.datos_hasta(info)
        if clave not in self._rangos:
            r = self.referencia(info)
            self._rangos[clave] = (r.min(numeric_only=True), r.max(numeric_only=True))
        return self._rangos[clave]

    # ------------------------------------------------------------------ series de lo servido
    def _variantes(self, nombre):
        info = self.g.manifiesto["sst"][nombre]
        out = {"principal": (self.g.modelos[nombre], info)}
        if info.get("respaldo"):
            out["respaldo"] = (self.g.respaldos[nombre], info["respaldo"])
        return out

    def serie(self, zona: str, h: int) -> pd.DataFrame:
        """Para cada día: verdad, predicción servida, intervalo conformal servido y variante usada."""
        clave = (zona, h)
        if clave in self._series:
            return self._series[clave]
        nombre = f"sst_{zona}_h{h}"
        y = self.g.ds[f"y_delta_{zona}_h{h}"]
        por_variante = {}
        for var, (m, info) in self._variantes(nombre).items():
            X = self.g.matriz(self.g.ds.index, info["variables"])
            ok = X.notna().all(axis=1)
            pred = pd.Series(np.nan, index=X.index)
            pred[ok] = m.predict(X[ok])
            q = info["cuantiles_residuo_validacion"]
            fechas = X.index[(X.index >= DESDE_MONITOR) & ok.values]
            hasta = self.g.datos_hasta(info)
            res = y - pred
            res = res[res.index + pd.Timedelta(days=h) > hasta]   # solo errores fuera de muestra (como la API)
            # misma función (cascada estacional) que usa la API en Gemelo.residuos_recientes
            iv = D.intervalos_cascada(res, fechas, h, self.g.cfg_intervalo,
                                      (q["0.05"], q["0.95"])).reindex(X.index)
            por_variante[var] = pd.DataFrame({"pred": pred, "lo": iv["lo"], "hi": iv["hi"], "metodo": iv["metodo"],
                                              "disponible": ok, "datos_hasta": hasta})
        servido = por_variante["principal"].copy()
        servido["variante"] = "principal"
        if "respaldo" in por_variante:
            usar_resp = ~servido["disponible"]
            cols = ["pred", "lo", "hi", "metodo", "datos_hasta"]
            servido.loc[usar_resp, cols] = por_variante["respaldo"].loc[usar_resp, cols]
            servido.loc[usar_resp, "variante"] = "respaldo"
        servido["y"] = y
        self._series[clave] = servido
        return servido

    # ------------------------------------------------------------------ control de un día
    def _top(self, nombre, variante):
        m = self._variantes(nombre)[variante][0]
        ganancia = pd.Series(m.get_booster().get_score(importance_type="total_gain"))
        return list(ganancia.sort_values(ascending=False).index[: self.cm["top_variables"]])

    def _psis(self, X: pd.DataFrame, fechas, top, ref=None, excluir_anio=None) -> dict:
        ref = self.ref if ref is None else ref
        mask = D.mascara_estacional(ref.index, fechas, self.cm["margen_estacional_dias"])
        if excluir_anio is not None:  # calibración: la ventana no puede estar en su propia referencia
            mask &= ~np.isin(ref.index.year, excluir_anio)
        # Los contadores (días desde la última lluvia fuerte, racha sobre el umbral) crecen de 1 en 1: en una
        # ventana de 90 días su distribución SIEMPRE parece desplazada. No tiene sentido el PSI; sí el rango.
        psis = {v: D.psi(ref.loc[mask, v], X[v]) for v in top if not es_contador(v)}
        return {k: round(v, 3) for k, v in sorted(psis.items(), key=lambda kv: -np.nan_to_num(kv[1]))}

    def _entradas(self, nombre, fecha, variante):
        info = self._variantes(nombre)[variante][1]
        top = self._top(nombre, variante)
        fechas = self.g.ds.loc[fecha - pd.Timedelta(days=self.cm["ventana_dias"] - 1):fecha].index
        X = self.g.matriz(fechas, top).dropna()
        psis = self._psis(X, fechas, top, ref=self.referencia(info))
        psi_max = max((v for v in psis.values() if np.isfinite(v)), default=np.nan)
        minimo, maximo = self.rango(info)
        return {"psi_max": psi_max, "variable_psi_max": next(iter(psis), None), "psi_top": psis,
                "psi_umbral": self.umbral_psi(nombre),
                "fuera_rango": D.fraccion_fuera_de_rango(X, minimo, maximo)}

    # ------------------------------------------------------------------ calibración de umbrales PSI
    def calibrar(self, anios=range(2010, 2019)) -> dict:
        """Umbral PSI por modelo = percentil del PSI máximo mensual en años NORMALES (dentro del entrenamiento),
        dejando el año de cada ventana fuera de su referencia. Así el aviso salta por algo inusual, no por la
        variabilidad habitual (las anomalías persisten semanas y una ventana de 90 días nunca es 'típica')."""
        umbrales = {}
        fechas_ctrl = pd.date_range(f"{min(anios)}-01-31", f"{max(anios)}-12-31", freq="ME")
        for nombre in self.g.manifiesto["sst"]:
            top = self._top(nombre, "principal")
            valores = []
            for f in fechas_ctrl:
                fechas = self.ref.loc[f - pd.Timedelta(days=self.cm["ventana_dias"] - 1):f].index
                X = self.ref.loc[fechas, top].dropna()
                if len(X) < 30:
                    continue
                psis = self._psis(X, fechas, top, excluir_anio=f.year)
                valores.append(max(v for v in psis.values() if np.isfinite(v)))
            umbrales[nombre] = round(float(np.percentile(valores, self.cm["psi_percentil_calibracion"])), 3)
        return umbrales

    def umbral_psi(self, nombre) -> float:
        cal = self._calibracion()
        return cal.get(nombre, self.cm["psi_deriva"])

    def _calibracion(self) -> dict:
        if not hasattr(self, "_cal"):
            ruta = DIR_DATOS / "monitor_calibracion.json"
            self._cal = json.loads(ruta.read_text(encoding="utf-8")) if ruta.exists() else {}
        return self._cal

    def _rendimiento(self, zona, h, fecha):
        s = self.serie(zona, h)
        ini = fecha - pd.Timedelta(days=self.cm["ventana_dias"] + h - 1)
        w = s.loc[ini:fecha - pd.Timedelta(days=h)].dropna(subset=["y", "pred", "lo", "hi"])
        # solo días FUERA de muestra: si el modelo se reentrenó con ellos, su error no dice nada
        w = w[w.index + pd.Timedelta(days=h) > pd.to_datetime(w["datos_hasta"])]
        if len(w) < 30:
            return {"n": len(w), "rmse": np.nan, "rmse_persistencia": np.nan, "skill": np.nan, "cobertura": np.nan,
                    "sesgo": np.nan, "nota": "rendimiento aún no evaluable: menos de 30 días fuera de muestra "
                                             "(p. ej. recién reentrenado)"}
        rmse = float(np.sqrt(np.mean((w["y"] - w["pred"]) ** 2)))
        rmse_p = float(np.sqrt(np.mean(w["y"] ** 2)))  # persistencia: cambio de anomalía = 0
        return {"n": len(w), "rmse": rmse, "rmse_persistencia": rmse_p, "skill": 1 - rmse / rmse_p,
                "cobertura": D.cobertura(w["y"], w["pred"], w["lo"], w["hi"]),
                "sesgo": float((w["y"] - w["pred"]).mean())}

    def modelo(self, zona, h, fecha) -> dict:
        nombre = f"sst_{zona}_h{h}"
        variante = self.serie(zona, h).loc[fecha, "variante"] if fecha in self.serie(zona, h).index else "principal"
        if variante == "principal" and not self.serie(zona, h).loc[fecha, "disponible"] and \
                "respaldo" in self._variantes(nombre):
            variante = "respaldo"
        ent = self._entradas(nombre, fecha, variante)
        ren = self._rendimiento(zona, h, fecha)
        est, motivos = D.estado(ren["skill"], ren["cobertura"], ent["psi_max"], ent["fuera_rango"],
                                {**self.cm, "psi_deriva": ent["psi_umbral"]})
        info = self._variantes(nombre)[variante][1]
        return {"estado": est, "motivos": motivos, "version": info["version_registro"],
                "datos_hasta": str(self.g.datos_hasta(info).date()),
                "usa_prevision": bool(info.get("usa_prevision")), "rendimiento": _redondear(ren),
                "entradas": _redondear(ent)}

    def prevision(self, fecha) -> dict:
        """¿Ha cambiado el sesgo de la previsión (ya corregido con feb-jul 2024) en la ventana reciente?"""
        if self.g.prev_diaria is None:
            return {"estado": "SIN_DATOS", "motivos": ["no hay previsiones descargadas"]}
        ini = fecha - pd.Timedelta(days=self.cm["ventana_dias"] - 1)
        sesgos, motivos = {}, []
        for punto, por_k in self.g.prev_diaria.items():
            for k in (1, 7):
                d = por_k[k].loc[ini:fecha]
                if len(d) < 30:
                    continue
                for var, umbral in self.cm["sesgo_prevision_max"].items():
                    b = float((d[var] - self.g.diario[f"{var}_{punto}"].reindex(d.index)).mean())
                    sesgos[f"{var}_{punto}_d{k}"] = round(b, 3)
                    if abs(b) > umbral:
                        motivos.append(f"sesgo de {var} en {punto} a {k} d: {b:+.2f} (umbral ±{umbral})")
        if not sesgos:
            return {"estado": "SIN_DATOS", "motivos": [f"sin previsión archivada antes de {fecha.date()}"]}
        return {"estado": "AVISO" if motivos else "OK", "motivos": motivos, "sesgos": sesgos}

    def hipoxia(self, fecha) -> dict:
        previas = self.g.campanas[self.g.campanas <= fecha]
        ultima = previas.max() if len(previas) else None
        dias = (fecha - ultima).days if ultima is not None else None
        if ultima is None or dias > 30:
            return {"estado": "SIN_DATOS", "ultima_campana": None if ultima is None else str(ultima.date()),
                    "motivos": [f"sin campañas in situ en los últimos {dias} días: el modelo no se puede vigilar ni "
                                "usar con datos actuales" if dias else "sin campañas"]}
        return {"estado": "OK", "ultima_campana": str(ultima.date()), "motivos": []}

    def informe(self, fecha=None) -> dict:
        fecha = self.g.ultima_fecha() if fecha is None else pd.Timestamp(fecha)
        modelos = {f"sst_{z}_h{h}": self.modelo(z, h, fecha) for z in self.g.zonas for h in self.g.horizontes}
        prev, hip = self.prevision(fecha), self.hipoxia(fecha)
        peor = max([m["estado"] for m in modelos.values()] + [prev["estado"]], key=ORDEN.get)
        # Enfriamiento: no proponer reentrenar un modelo que ya vio datos muy recientes (acaba de reentrenarse);
        # su ventana de evaluación ya no sería honesta y se reentrenaría en bucle.
        espera = pd.Timedelta(days=self.cfg["reentrenamiento"]["dias_minimos_entre_reentrenos"])
        reentrenar, en_espera = [], []
        for n, m in modelos.items():
            if m["estado"] not in self.cm["disparar_con"]:
                continue
            (en_espera if fecha - pd.Timestamp(m["datos_hasta"]) < espera else reentrenar).append(n)
        return {"fecha": str(fecha.date()), "estado_global": peor, "modelos": modelos, "prevision": prev,
                "hipoxia": hip, "reentrenar": reentrenar, "en_espera_tras_reentreno": en_espera}

    def historico(self, fechas) -> pd.DataFrame:
        filas = []
        for f in fechas:
            inf = self.informe(f)
            for n, m in inf["modelos"].items():
                filas.append({"fecha": pd.Timestamp(f), "modelo": n, "estado": m["estado"], "version": m["version"],
                              "usa_prevision": m["usa_prevision"], **m["rendimiento"],
                              "psi_max": m["entradas"]["psi_max"], "fuera_rango": m["entradas"]["fuera_rango"],
                              "variable_psi_max": m["entradas"]["variable_psi_max"],
                              "motivos": "; ".join(m["motivos"])})
        return pd.DataFrame(filas)


def es_contador(variable: str) -> bool:
    return variable.startswith(("dias_desde", "racha_"))


def _redondear(d):
    return {k: (round(float(v), 4) if isinstance(v, (float, np.floating)) and np.isfinite(v) else
                (None if isinstance(v, (float, np.floating)) else v)) for k, v in d.items()}
