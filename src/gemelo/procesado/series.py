"""raw -> interim: de cubos (tiempo, lat, lon) a series diarias medias por subzona.

Esta es nuestra "rejilla común": cada fuente tiene su propia rejilla (0,05°, 1 km, 1/24°, 0,25°),
pero todas se reducen a la media de la misma subzona y al mismo paso diario.
"""
import numpy as np
import pandas as pd
import xarray as xr

from gemelo.config import DATA_RAW

RAW_CMEMS = DATA_RAW / "copernicus"
MAX_HUECO_DIAS = 3  # huecos más largos no se rellenan: se dejan como NaN


def media_subzona(da: xr.DataArray, caja: dict) -> pd.Series:
    """Media espacial de `da` dentro de la caja {lat: [min, max], lon: [min, max]}."""
    sel = da.where(
        (da.latitude >= caja["lat"][0]) & (da.latitude <= caja["lat"][1])
        & (da.longitude >= caja["lon"][0]) & (da.longitude <= caja["lon"][1])
    )
    serie = sel.mean(["latitude", "longitude"]).to_series()
    serie.index = pd.DatetimeIndex(serie.index).normalize()
    return serie


def _abrir(nombre: str) -> xr.Dataset:
    ds = xr.open_dataset(RAW_CMEMS / f"{nombre}.nc")
    return ds.isel(depth=0, drop=True) if "depth" in ds.dims else ds


def series_copernicus(subzonas: dict) -> pd.DataFrame:
    sst = _abrir("sst")["analysed_sst"] - 273.15  # K -> °C
    chl = np.log10(_abrir("clorofila")["CHL"])   # la clorofila es log-normal: trabajamos en log10
    temp = _abrir("fisica_temp")
    mld = _abrir("fisica_mld")["mlotst"]

    cols = {}
    for z, caja in subzonas.items():
        cols[f"sst_{z}"] = media_subzona(sst, caja)
        cols[f"logchl_{z}"] = media_subzona(chl, caja)
    # el modelo no tiene celdas en la laguna: solo costa
    cols["thetao_costa"] = media_subzona(temp["thetao"], subzonas["costa"])
    cols["bottomT_costa"] = media_subzona(temp["bottomT"], subzonas["costa"])
    cols["mld_costa"] = media_subzona(mld, subzonas["costa"])
    return pd.DataFrame(cols)


def series_meteo(puntos) -> pd.DataFrame:
    nombres = {
        "temperature_2m_mean": "t2m", "temperature_2m_max": "t2max", "precipitation_sum": "prec",
        "wind_speed_10m_mean": "viento", "wind_speed_10m_max": "vientomax",
        "shortwave_radiation_sum": "rad",
    }
    partes = []
    for p in puntos:
        df = pd.read_parquet(DATA_RAW / "meteo" / f"era5_{p}.parquet")
        # la dirección es circular (359° ≈ 1°): se pasa a componentes u/v del viento
        rad_dir = np.deg2rad(df["wind_direction_10m_dominant"])
        df["viento_u"] = -df["wind_speed_10m_mean"] * np.sin(rad_dir)
        df["viento_v"] = -df["wind_speed_10m_mean"] * np.cos(rad_dir)
        df = df.drop(columns="wind_direction_10m_dominant").rename(columns=nombres)
        partes.append(df.add_suffix(f"_{p}"))
    out = pd.concat(partes, axis=1)
    out.index = pd.DatetimeIndex(out.index).normalize()
    return out


def construir_diario(cfg: dict) -> tuple[pd.DataFrame, dict]:
    """Tabla diaria con todas las series. Devuelve (tabla, informe de huecos rellenados)."""
    df = pd.concat([series_copernicus(cfg["subzonas"]), series_meteo(cfg["meteo"]["puntos"])], axis=1)
    dias = pd.date_range(df.index.min(), cfg["periodo"]["fin"], freq="D")
    df = df.reindex(dias)
    df.index.name = "fecha"

    informe = {}
    for c in df.columns:
        antes = int(df[c].isna().sum())
        # Relleno CAUSAL (arrastrar el último valor conocido), no interpolación: interpolar
        # usaría el valor de mañana para rellenar hoy, cosa imposible en predicción real.
        df[c] = df[c].ffill(limit=MAX_HUECO_DIAS)
        informe[c] = {"nan_antes": antes, "rellenados": antes - int(df[c].isna().sum())}
    return df, informe
