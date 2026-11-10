"""Variables de METEOROLOGÍA FUTURA (días t+1 .. t+w) para usar previsión como entrada (Fase 7).

¡Ojo! Estas variables miran al futuro A PROPÓSITO: representan la previsión meteorológica disponible el
día t. Por eso NO van en dataset_sst (cuyas variables solo usan datos <= t) sino en tablas aparte, y se
construyen desde dos fuentes con el MISMO código:

  - "era5":      lo que realmente pasó (previsión perfecta). Para entrenar (no hay previsiones archivadas
                 antes de 2024) y como cota superior.
  - "prevision": lo que se PREVEÍA el día t para t+k (Open-Meteo Previous Runs), con el sesgo sistemático
                 corregido en el periodo de calibración. Para la evaluación honesta y para el servicio.

Nunca usan la temperatura del agua futura: solo aire, viento, lluvia y radiación.
"""
import numpy as np
import pandas as pd

from gemelo.procesado.features import _doy

VENTANAS = (1, 3, 7)
# variable diaria -> cómo se agrega en la ventana
AGREGACION = {"t2m_anom": "mean", "t2max": "max", "prec": "sum", "viento": "mean", "vientomax": "max",
              "rad": "mean", "viento_u": "mean", "viento_v": "mean"}
CORREGIR_SESGO = ["t2m", "t2max", "viento", "vientomax", "rad", "viento_u", "viento_v"]  # la lluvia no (aditivo no tiene sentido)


# ------------------------------------------------------------------ fuentes: valor diario por antelación
def diaria_desde_horaria(horaria: pd.DataFrame, k: int) -> pd.DataFrame:
    """Agregados diarios (UTC) de la previsión emitida k días antes. Mismas definiciones que ERA5 diario."""
    s = lambda v: horaria[f"{v}_previous_day{k}"]
    dia = horaria.index.floor("D")
    vel, dir_ = s("wind_speed_10m"), np.deg2rad(s("wind_direction_10m"))
    # dirección dominante = dirección del viento medio vectorial (como el "dominant" diario de Open-Meteo)
    u_h, v_h = -vel * np.sin(dir_), -vel * np.cos(dir_)
    g = pd.DataFrame({"t2m": s("temperature_2m"), "prec": s("precipitation"), "viento": vel,
                      "rad_wm2": s("shortwave_radiation"), "u_h": u_h, "v_h": v_h}).groupby(dia)
    d = pd.DataFrame({
        "t2m": g["t2m"].mean(), "t2max": g["t2m"].max(), "prec": g["prec"].sum(min_count=1),
        "viento": g["viento"].mean(), "vientomax": g["viento"].max(),
        "rad": g["rad_wm2"].sum(min_count=1) * 3600 / 1e6,          # W/m² horario -> MJ/m² diario (como ERA5)
    })
    dom = np.arctan2(-g["u_h"].mean(), -g["v_h"].mean())            # dirección dominante (radianes)
    d["viento_u"] = -d["viento"] * np.sin(dom)                       # igual que series.series_meteo con ERA5
    d["viento_v"] = -d["viento"] * np.cos(dom)
    completos = g["t2m"].count() == 24                               # días con las 24 horas
    return d[completos]


def prevision_por_antelacion(horaria: pd.DataFrame, dias_max: int = 7) -> dict[int, pd.DataFrame]:
    return {k: diaria_desde_horaria(horaria, k) for k in range(1, dias_max + 1)}


def sesgo_prevision(prev: dict[int, pd.DataFrame], era5: pd.DataFrame, punto: str, ini: str, fin: str) -> pd.DataFrame:
    """Sesgo medio (previsión - ERA5) por variable y antelación en el periodo de calibración."""
    filas = {}
    for k, d in prev.items():
        obs = era5[[f"{v}_{punto}" for v in CORREGIR_SESGO]].rename(columns=lambda c: c.rsplit("_", 1)[0])
        dif = (d[CORREGIR_SESGO] - obs.reindex(d.index)).loc[ini:fin]
        filas[k] = dif.mean()
    return pd.DataFrame(filas).T  # índice = antelación k


def sesgo_causal(error: pd.Series, k: int, metodo: str, retraso: int = 5, dias_movil: int = 60,
                 margen_estacional: int = 45) -> pd.Series:
    """Sesgo a restar a la previsión de antelación k para cada fecha OBJETIVO T, usando SOLO errores que ya se
    conocían al emitirla (objetivos <= T - k - retraso; `retraso` = días que tarda en publicarse ERA5).

    error  = previsión - ERA5 por fecha objetivo (antelación k).
    metodo:
      'movil'              media de los últimos `dias_movil` días conocidos: se adapta a saltos (cambio de modelo).
      'estacional'         media de los errores de la misma época del año (± margen) de toda la historia conocida.
      'estacional_movil'   estacional + media móvil de lo que el estacional no explica (las dos cosas).
    """
    e = error.dropna().asfreq("D")
    desfase = pd.Timedelta(days=k + retraso)
    if metodo == "movil":
        return e.rolling(f"{dias_movil}D", min_periods=20).mean().shift(freq=desfase).reindex(error.index)

    # estacional causal: para cada T, errores con fecha <= T - desfase y día del año a <= margen
    fechas = e.dropna().index
    valores = e.dropna().values
    dias = fechas.values.astype("datetime64[D]").astype(np.int64)
    doy = _doy(fechas)
    estacional = pd.Series(np.nan, index=error.index)
    for t in error.index:
        limite = (t - desfase).value // 86_400_000_000_000
        d = np.abs(doy - _doy(pd.DatetimeIndex([t]))[0])
        sel = (dias <= limite) & (np.minimum(d, 365 - d) <= margen_estacional)
        if sel.sum() >= 20:
            estacional[t] = valores[sel].mean()
    if metodo == "estacional":
        return estacional
    resto = (error - estacional).dropna().asfreq("D")
    movil_resto = resto.rolling(f"{dias_movil}D", min_periods=20).mean().shift(freq=desfase).reindex(error.index)
    return estacional + movil_resto.fillna(0.0)


def corregir_causal(prev: dict[int, pd.DataFrame], era5: pd.DataFrame, punto: str, metodo: str,
                    **kw) -> dict[int, pd.DataFrame]:
    """Corrige cada variable y antelación con su sesgo causal. Donde aún no hay historia suficiente, NaN
    (mejor no usar la previsión que usarla mal corregida: el servicio pasa al modelo de respaldo)."""
    out = {}
    for k, d in prev.items():
        d = d.copy()
        for v in CORREGIR_SESGO:
            err = d[v] - era5[f"{v}_{punto}"].reindex(d.index)
            d[v] = d[v] - sesgo_causal(err, k, metodo, **kw)
        out[k] = d
    return out


def corregir(prev: dict[int, pd.DataFrame], sesgo: pd.DataFrame) -> dict[int, pd.DataFrame]:
    out = {}
    for k, d in prev.items():
        d = d.copy()
        d[CORREGIR_SESGO] = d[CORREGIR_SESGO] - sesgo.loc[k, CORREGIR_SESGO]
        out[k] = d
    return out


# ------------------------------------------------------------------ construcción de variables
def construir(valor_dia, puntos, ref: pd.DataFrame, indice: pd.DatetimeIndex, ventanas=VENTANAS) -> pd.DataFrame:
    """Variables prev{w}_{var}_{punto} para cada día base t del índice.

    valor_dia(var, punto, k) -> Serie indexada por FECHA OBJETIVO con el valor que se conoce en t = objetivo - k
    (para ERA5 no depende de k; para la previsión, es la emitida k días antes)."""
    cols = {}
    idx = pd.DatetimeIndex(indice)
    for p in puntos:
        for w in ventanas:
            por_var = {v: [] for v in AGREGACION}
            for k in range(1, w + 1):
                objetivo = idx + pd.Timedelta(days=k)
                clim = ref[f"clim_t2m_{p}"].reindex(_doy(objetivo)).values
                for v in AGREGACION:
                    base = "t2m" if v == "t2m_anom" else v
                    x = valor_dia(base, p, k).reindex(objetivo).values
                    por_var[v].append(x - clim if v == "t2m_anom" else x)
            for v, agg in AGREGACION.items():
                m = np.vstack(por_var[v])   # (w, n)
                with np.errstate(all="ignore"):
                    # NaN si falta algún día de la ventana: mejor no predecir que inventar
                    r = {"mean": np.mean, "max": np.max, "sum": np.sum}[agg](m, axis=0)
                cols[f"prev{w}_{v}_{p}"] = r
    return pd.DataFrame(cols, index=idx)


def fuente_era5(diario: pd.DataFrame):
    return lambda var, p, k: diario[f"{var}_{p}"]


def fuente_prevision(prev_por_punto: dict[str, dict[int, pd.DataFrame]]):
    return lambda var, p, k: prev_por_punto[p][k][var]


# ------------------------------------------------------------------ preparación (pipeline y servicio)
def preparar(cfg: dict, diario: pd.DataFrame, ref: pd.DataFrame, indice: pd.DatetimeIndex):
    """Tablas de variables futuras desde ERA5 y desde la previsión real corregida.
    Devuelve (f_era5, f_prev, prev_corregida, sesgos); las tres últimas son None si no hay previsiones."""
    from gemelo.config import DATA_RAW
    puntos, p = list(cfg["meteo"]["puntos"]), cfg["prevision"]
    corr = p.get("correccion", {"metodo": "fija"})
    f_era5 = construir(fuente_era5(diario), puntos, ref, indice)
    prev, sesgos = {}, {}
    for pt in puntos:
        ruta = DATA_RAW / "meteo" / f"prevision_{pt}.parquet"
        if not ruta.exists():
            return f_era5, None, None, None
        bruto = prevision_por_antelacion(pd.read_parquet(ruta), p["dias_max"])
        sesgos[pt] = sesgo_prevision(bruto, diario, pt, *p["calibracion"])  # se guarda siempre (referencia)
        if corr["metodo"] == "fija":
            prev[pt] = corregir(bruto, sesgos[pt])
        else:
            prev[pt] = corregir_causal(bruto, diario, pt, corr["metodo"], retraso=corr["retraso_dias"],
                                       dias_movil=corr["dias_movil"], margen_estacional=corr["margen_estacional"])
    f_prev = construir(fuente_prevision(prev), puntos, ref, indice)
    return f_era5, f_prev, prev, sesgos


def a_tabla(prev: dict[str, dict[int, pd.DataFrame]]) -> pd.DataFrame:
    """{punto: {antelación: diario}} -> tabla larga (para guardar en Parquet)."""
    partes = [d.assign(punto=pt, antelacion=k).rename_axis("fecha").reset_index()
              for pt, por_k in prev.items() for k, d in por_k.items()]
    return pd.concat(partes, ignore_index=True)


def desde_tabla(tabla: pd.DataFrame) -> dict[str, dict[int, pd.DataFrame]]:
    out = {}
    for (pt, k), g in tabla.groupby(["punto", "antelacion"]):
        out.setdefault(pt, {})[int(k)] = g.drop(columns=["punto", "antelacion"]).set_index("fecha").sort_index()
    return out
