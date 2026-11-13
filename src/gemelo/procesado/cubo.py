"""Cubo 4D compacto de la laguna (lat, lon, profundidad, tiempo) para el visor CesiumJS (Fase 7).

A partir de los mapas L2 in situ de la UPCT (200 x 200 puntos, 10 niveles de 0,5 a 5 m, una campaña por semana):
  - se promedia en bloques de `factor` x `factor` puntos (por defecto 4 -> 50 x 50), exigiendo que al menos la
    mitad del bloque sea agua (si no, los bloques de la orilla con 1 punto mojado darían medias engañosas);
  - se guardan solo los puntos con dato, en formato largo (una fila por campaña, nivel y punto);
  - mismo control de calidad de salinidad que el procesado de campañas (mar-may 2017 -> NaN).
"""
import pandas as pd

# xarray y el procesado in situ se importan DENTRO de las funciones: la API (imagen sin xarray) importa este
# módulo solo por UNIDADES.
UNIDADES = {"o2": "mg/L", "temp": "°C", "sal": "PSU", "chl": "µg/L", "turb": "NTU"}


def reducir(da, factor: int):
    """Media por bloques factor x factor con al menos la mitad del bloque con dato."""
    bloques = {"longitude": factor, "latitude": factor}
    media = da.coarsen(bloques, boundary="trim").mean()
    n = da.notnull().coarsen(bloques, boundary="trim").sum()
    return media.where(n >= factor * factor / 2)


def construir_cubo(cfg: dict, factor: int = 4) -> pd.DataFrame:
    from gemelo.procesado.insitu import PREFIJO, cargar_cubo
    partes = []
    for variable in cfg["upct"]["variables"]:
        p = PREFIJO[variable]
        da = reducir(cargar_cubo(variable), factor).astype("float32")
        if variable == "Salinidad":  # QC: campañas con salinidad superficial media no plausible
            sup = da.isel(level=0).mean(["latitude", "longitude"])
            malas = sup < cfg["upct"]["sal_sup_min_plausible"]
            da = da.where(~malas)
        df = da.rename(p).to_dataframe()[[p]]
        partes.append(df)
        del da
    cubo = pd.concat(partes, axis=1).dropna(how="all").reset_index()
    cubo = cubo.rename(columns={"time": "fecha", "level": "profundidad", "longitude": "lon", "latitude": "lat"})
    for c in ("lon", "lat", "profundidad"):
        cubo[c] = cubo[c].astype("float32")
    return cubo.sort_values(["fecha", "profundidad", "lat", "lon"]).reset_index(drop=True)


def resumen_por_campana(cubo: pd.DataFrame, umbral_hipoxia: float, umbral_anoxia: float) -> pd.DataFrame:
    """Indicadores 3D por campaña. Las celdas tienen el mismo tamaño, así que el % de celdas = % de volumen."""
    g = cubo.dropna(subset=["o2"]).groupby("fecha")
    return pd.DataFrame({
        "celdas": g.size(),
        "pct_volumen_hipoxia": g["o2"].apply(lambda s: 100 * float((s < umbral_hipoxia).mean())),
        "pct_volumen_anoxia": g["o2"].apply(lambda s: 100 * float((s < umbral_anoxia).mean())),
        "o2_medio": g["o2"].mean(),
        "temp_media": cubo.groupby("fecha")["temp"].mean(),
    }).round(3)


def contorno_laguna(cubo: pd.DataFrame) -> list[list[float]]:
    """Contorno aproximado de la laguna como polígono [[lon, lat], ...] (para el mapa del dashboard).
    Para cada fila de latitud de la rejilla se toma el punto de agua más al oeste y el más al este: se sube
    por la orilla oeste y se baja por la este. Sencillo y suficiente para una laguna sin islas grandes."""
    sup = cubo[cubo["profundidad"] == cubo["profundidad"].min()][["lon", "lat"]].drop_duplicates()
    medio_paso = float(sup["lon"].drop_duplicates().sort_values().diff().median()) / 2   # borde de la celda, no su centro
    filas = sup.groupby("lat")["lon"].agg(["min", "max"]).sort_index()
    oeste = [[round(float(r["min"] - medio_paso), 5), round(float(lat), 5)] for lat, r in filas.iterrows()]
    este = [[round(float(r["max"] + medio_paso), 5), round(float(lat), 5)] for lat, r in filas.iloc[::-1].iterrows()]
    return oeste + este + [oeste[0]]
