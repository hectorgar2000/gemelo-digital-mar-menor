"""interim -> processed: ingeniería de variables, objetivos por horizonte y partición temporal.

Regla de oro (anti-fuga): una variable del día t solo puede usar datos de días <= t.
Los objetivos son lo único que mira al futuro (t + h).
"""
import numpy as np
import pandas as pd

ZONAS = ["laguna", "costa"]
LAGS_SST = [0, 1, 2, 3, 7, 14, 30]
VENTANAS = [3, 7, 14, 30]
LLUVIA_FUERTE_MM = 30  # día de lluvia intensa (la DANA de 2019 dejó 84 y 90 mm)
TOPE_DIAS_DESDE = 365


# ---------------------------------------------------------------- climatología y anomalías
def _doy(idx: pd.DatetimeIndex) -> np.ndarray:
    """Día del año 1..365, con el 29 de febrero contado como 28 (evita un día con la mitad de datos)."""
    d = idx.day_of_year.values.copy()
    bisiesto_tras_feb = idx.is_leap_year & (idx.month > 2)
    d[bisiesto_tras_feb] -= 1
    return d


def climatologia(serie: pd.Series, ini: str, fin: str, suavizado: int = 31) -> pd.Series:
    """Media por día del año en [ini, fin], suavizada con ventana circular. Devuelve índice 1..365."""
    ref = serie[ini:fin].dropna()
    media = ref.groupby(_doy(ref.index)).mean().reindex(range(1, 366)).interpolate()
    # ventana circular: diciembre empalma con enero
    ext = pd.concat([media.iloc[-suavizado:], media, media.iloc[:suavizado]])
    return ext.rolling(suavizado, center=True).mean().iloc[suavizado:-suavizado]


def anomalia(serie: pd.Series, clim: pd.Series) -> pd.Series:
    return serie - clim.reindex(_doy(serie.index)).values


# ---------------------------------------------------------------- olas de calor marinas
def umbral_p90(serie: pd.Series, ini: str, fin: str, pct: int = 90, ventana: int = 5) -> pd.Series:
    """Percentil `pct` por día del año usando ±`ventana` días alrededor (Hobday et al., 2016)."""
    ref = serie[ini:fin].dropna()
    doy = _doy(ref.index)
    umbral = {}
    for d in range(1, 366):
        dist = np.abs(((doy - d + 182) % 365) - 182)
        umbral[d] = ref[dist <= ventana].quantile(pct / 100)
    return pd.Series(umbral)


def flag_mhw(serie: pd.Series, umbral: pd.Series, min_dias: int = 5) -> pd.Series:
    """True si el día pertenece a una racha de >= min_dias por encima del umbral.
    OJO: mira días futuros dentro de la racha -> solo vale como OBJETIVO, nunca como variable."""
    sobre = pd.Series(serie.values > umbral.reindex(_doy(serie.index)).values, index=serie.index)
    grupo = (sobre != sobre.shift()).cumsum()
    largo = sobre.groupby(grupo).transform("size")
    return (sobre & (largo >= min_dias)).where(serie.notna())


def racha_sobre_umbral(serie: pd.Series, umbral: pd.Series) -> pd.Series:
    """Días consecutivos por encima del umbral HASTA hoy (causal: sí vale como variable)."""
    sobre = (serie.values > umbral.reindex(_doy(serie.index)).values).astype(int)
    racha = np.zeros(len(sobre), dtype=int)
    for i, s in enumerate(sobre):
        racha[i] = (racha[i - 1] + 1) if (s and i > 0) else s
    return pd.Series(racha, index=serie.index)


def dias_desde(evento: pd.Series, tope: int = TOPE_DIAS_DESDE) -> pd.Series:
    """Días transcurridos desde el último True hasta hoy (0 = hoy). Causal. Con tope."""
    cuenta = np.full(len(evento), tope, dtype=int)
    ultimo = None
    for i, e in enumerate(evento.values):
        if e:
            ultimo = i
        if ultimo is not None:
            cuenta[i] = min(i - ultimo, tope)
    return pd.Series(cuenta, index=evento.index)


# ---------------------------------------------------------------- dataset SST / MHW
def construir_dataset_sst(diario: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    part = cfg["particion"]
    u = cfg["umbrales"]
    # Climatología para ANOMALÍAS: solo datos anteriores a la validación (sin fuga)
    clim_ini, clim_fin = str(diario.index.min().date()), str(pd.Timestamp(part["validacion"][0]) - pd.Timedelta(days=1))
    X = {}
    Y = {}

    for z in ZONAS:
        sst = diario[f"sst_{z}"]
        anom = anomalia(sst, climatologia(sst, clim_ini, clim_fin))
        # Dos umbrales p90 distintos a propósito:
        #  - p90_obj: definición estándar de MHW (Hobday, base 1991-2020) -> solo para el OBJETIVO
        #  - p90_var: mismo cálculo pero solo con datos anteriores a validación -> para VARIABLES
        #    (base 1991-2020 incluye 2019-2020, que son validación: sería fuga)
        p90_obj = umbral_p90(sst, *u["climatologia"], pct=u["mhw_percentil"])
        p90_var = umbral_p90(sst, clim_ini, clim_fin, pct=u["mhw_percentil"])

        X[f"sst_{z}"] = sst
        for l in LAGS_SST:
            X[f"anom_{z}_lag{l}"] = anom.shift(l)
        for w in VENTANAS:
            X[f"anom_{z}_media{w}"] = anom.rolling(w).mean()
        X[f"anom_{z}_tend7"] = anom - anom.shift(7)
        X[f"racha_p90_{z}"] = racha_sobre_umbral(sst, p90_var)
        X[f"logchl_{z}"] = diario[f"logchl_{z}"]
        X[f"logchl_{z}_media7"] = diario[f"logchl_{z}"].rolling(7).mean()
        X[f"logchl_{z}_media30"] = diario[f"logchl_{z}"].rolling(30).mean()

        mhw = flag_mhw(sst, p90_obj, u["mhw_dias_min"])
        for h in cfg["horizontes_dias"]:
            Y[f"y_anom_{z}_h{h}"] = anom.shift(-h)
            Y[f"y_sst_{z}_h{h}"] = sst.shift(-h)
            Y[f"y_mhw_{z}_h{h}"] = mhw.shift(-h)

    X["dif_laguna_costa"] = diario["sst_laguna"] - diario["sst_costa"]

    # Física del modelo (costa)
    for c in ["thetao_costa", "bottomT_costa", "mld_costa"]:
        X[c] = diario[c]
    X["mld_costa_media7"] = diario["mld_costa"].rolling(7).mean()

    # Meteorología ERA5
    for p in cfg["meteo"]["puntos"]:
        t2m = diario[f"t2m_{p}"]
        t2m_anom = anomalia(t2m, climatologia(t2m, clim_ini, clim_fin))
        X[f"t2m_anom_{p}"] = t2m_anom
        X[f"t2m_anom_{p}_media3"] = t2m_anom.rolling(3).mean()
        X[f"t2m_anom_{p}_media7"] = t2m_anom.rolling(7).mean()
        X[f"t2max_{p}"] = diario[f"t2max_{p}"]
        for w in [1, 3, 7, 14, 30, 60, 90]:
            X[f"prec_{p}_suma{w}"] = diario[f"prec_{p}"].rolling(w).sum()
        X[f"dias_desde_lluvia_fuerte_{p}"] = dias_desde(diario[f"prec_{p}"] >= LLUVIA_FUERTE_MM)
        for w in [3, 7]:
            X[f"viento_{p}_media{w}"] = diario[f"viento_{p}"].rolling(w).mean()
            X[f"viento_u_{p}_media{w}"] = diario[f"viento_u_{p}"].rolling(w).mean()
            X[f"viento_v_{p}_media{w}"] = diario[f"viento_v_{p}"].rolling(w).mean()
        X[f"vientomax_{p}"] = diario[f"vientomax_{p}"]
        X[f"rad_{p}_media7"] = diario[f"rad_{p}"].rolling(7).mean()

    # Calendario: seno/coseno para que el 31-dic y el 1-ene queden "cerca"
    doy = _doy(diario.index)
    X["doy_sin"] = np.sin(2 * np.pi * doy / 365)
    X["doy_cos"] = np.cos(2 * np.pi * doy / 365)

    df = pd.concat([pd.DataFrame(X), pd.DataFrame(Y)], axis=1)
    df["particion"] = asignar_particion(df.index, part)
    return df.loc[cfg["periodo"]["inicio"]:cfg["periodo"]["fin"]]


# ---------------------------------------------------------------- dataset hipoxia
def construir_dataset_hipoxia(campanas: pd.DataFrame, dataset_sst: pd.DataFrame, cfg: dict,
                              gap_min: int = 3, gap_max: int = 14) -> pd.DataFrame:
    """Una fila por campaña t_i; objetivo = estado en la campaña siguiente t_{i+1}.
    Solo se usan pares con separación entre gap_min y gap_max días (horizonte ~1 semana)."""
    c = campanas.sort_index()
    sig = c.shift(-1)
    gap = (c.index.to_series().shift(-1) - c.index.to_series()).dt.days

    df = c.add_prefix("ahora_")
    df["dias_hasta_siguiente"] = gap.values
    df["y_pct_hipoxia"] = sig["pct_hipoxia"].values
    df["y_o2_fondo_min"] = sig["o2_fondo_min"].values
    df["y_hipoxia"] = (sig["pct_hipoxia"] > cfg["umbrales"]["hipoxia_area_pct"]).astype(float).values
    df.loc[sig["pct_hipoxia"].isna().values, "y_hipoxia"] = np.nan

    cols_diarias = [col for col in dataset_sst.columns
                    if not col.startswith("y_") and col != "particion"
                    and ("laguna" in col or "cuenca" in col or col.startswith("doy"))]
    df = df.join(dataset_sst[cols_diarias], how="left")
    df = df[(df["dias_hasta_siguiente"] >= gap_min) & (df["dias_hasta_siguiente"] <= gap_max)]
    df["anio"] = df.index.year  # para validación cruzada dejando un año fuera
    return df


# ---------------------------------------------------------------- partición
def asignar_particion(idx: pd.DatetimeIndex, part: dict) -> pd.Series:
    etiquetas = pd.Series("fuera", index=idx)
    for nombre in ["train", "validacion", "test"]:
        ini, fin = part[nombre]
        etiquetas[(idx >= ini) & (idx <= fin)] = nombre
    return etiquetas


def filas_de_particion(df: pd.DataFrame, particion: str, h: int, part: dict) -> pd.DataFrame:
    """Filas de una partición cuyo OBJETIVO (t + h) también cae dentro de ella (purga de frontera).

    Sin esto, los últimos h días de train tendrían como etiqueta valores de validación:
    el modelo 'vería' validación durante el entrenamiento."""
    ini, fin = pd.Timestamp(part[particion][0]), pd.Timestamp(part[particion][1])
    fecha_obj = df.index + pd.Timedelta(days=h)
    return df[(df.index >= ini) & (fecha_obj <= fin)]
