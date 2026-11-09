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
def periodo_climatologia(diario: pd.DataFrame, cfg: dict) -> tuple[str, str]:
    """Climatología para ANOMALÍAS: solo datos anteriores a la validación (sin fuga)."""
    fin = pd.Timestamp(cfg["particion"]["validacion"][0]) - pd.Timedelta(days=1)
    return str(diario.index.min().date()), str(fin.date())


def referencias_sst(diario: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Tablas por día del año (1..365) que comparten entrenamiento y servicio:
    clim_{zona}     climatología de las anomalías
    p90_obj_{zona}  umbral de ola de calor (definición estándar, base 1991-2020)
    p90_var_{zona}  umbral para la variable racha_p90 (solo datos anteriores a validación)"""
    u = cfg["umbrales"]
    ini, fin = periodo_climatologia(diario, cfg)
    ref = {}
    for z in ZONAS:
        sst = diario[f"sst_{z}"]
        ref[f"clim_{z}"] = climatologia(sst, ini, fin)
        # Dos umbrales p90 distintos a propósito: la base estándar 1991-2020 incluye 2019-2020
        # (validación), así que solo puede definir el OBJETIVO, nunca una variable (sería fuga).
        ref[f"p90_obj_{z}"] = umbral_p90(sst, *u["climatologia"], pct=u["mhw_percentil"])
        ref[f"p90_var_{z}"] = umbral_p90(sst, ini, fin, pct=u["mhw_percentil"])
    df = pd.DataFrame(ref)
    df.index.name = "dia_del_anio"
    return df


def construir_dataset_sst(diario: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    part = cfg["particion"]
    u = cfg["umbrales"]
    clim_ini, clim_fin = periodo_climatologia(diario, cfg)
    ref = referencias_sst(diario, cfg)
    X = {}
    Y = {}

    for z in ZONAS:
        sst = diario[f"sst_{z}"]
        anom = anomalia(sst, ref[f"clim_{z}"])
        p90_obj, p90_var = ref[f"p90_obj_{z}"], ref[f"p90_var_{z}"]

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
            # y_delta: cambio de anomalía respecto a hoy. Es el objetivo recomendado para los
            # árboles: no necesitan extrapolar niveles nunca vistos (test ~1 °C más cálido que train).
            Y[f"y_delta_{z}_h{h}"] = anom.shift(-h) - anom
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
    # tendencia desde la campaña ANTERIOR (causal)
    df["tend_o2_fondo_media"] = c["o2_fondo_media"].diff().values
    df["tend_pct_hipoxia"] = c["pct_hipoxia"].diff().values
    df["dias_desde_anterior"] = c.index.to_series().diff().dt.days.values
    df["dias_hasta_siguiente"] = gap.values
    df["y_pct_hipoxia"] = sig["pct_hipoxia"].values
    df["y_o2_fondo_min"] = sig["o2_fondo_min"].values
    df["y_hipoxia"] = (sig["pct_hipoxia"] > cfg["umbrales"]["hipoxia_area_pct"]).astype(float).values
    df.loc[sig["pct_hipoxia"].isna().values, "y_hipoxia"] = np.nan

    cols_diarias = [col for col in dataset_sst.columns
                    if not col.startswith("y_") and col != "particion"
                    and ("laguna" in col or "cuenca" in col or col.startswith("doy"))]
    df = df.join(dataset_sst[cols_diarias], how="left")
    # Se conserva la ÚLTIMA campaña (sin siguiente, dias_hasta_siguiente = NaN): no tiene objetivo y no
    # se usa para entrenar, pero es justo la que necesita el servicio para predecir el futuro.
    gap_ok = df["dias_hasta_siguiente"].between(gap_min, gap_max)
    df = df[gap_ok | df["dias_hasta_siguiente"].isna()]
    df["anio"] = df.index.year  # para validación cruzada dejando un año fuera
    return df


# ---------------------------------------------------------------- grupos (para SHAP agrupado)
GRUPOS = [  # (grupo, función que reconoce la columna); el primero que encaja gana
    ("calendario", lambda c: c.startswith("doy_")),
    ("oxigeno_in_situ", lambda c: "o2_" in c or "hipoxia" in c or "anoxia" in c),
    ("estratificacion", lambda c: "estrat" in c or "mld_" in c),
    ("agua_in_situ", lambda c: c.startswith("ahora_")),
    ("muestreo", lambda c: c.startswith("dias_hasta") or c.startswith("dias_desde_anterior")),
    ("temp_laguna", lambda c: "laguna" in c and ("anom" in c or "sst" in c or "racha" in c)),
    ("temp_costa", lambda c: ("costa" in c and ("anom" in c or "sst" in c or "racha" in c or "thetao" in c
                                               or "bottomT" in c)) or c == "dif_laguna_costa"),
    ("clorofila_satelite", lambda c: "logchl" in c),
    ("lluvia", lambda c: "prec_" in c or "lluvia" in c),
    ("viento", lambda c: "viento" in c),
    ("aire_y_radiacion", lambda c: c.startswith(("t2m", "t2max", "rad_"))),
]


def grupo_variable(col: str) -> str:
    for nombre, encaja in GRUPOS:
        if encaja(col):
            return nombre
    return "otros"


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
