"""raw -> interim: del cubo de O2 de la UPCT a una tabla con una fila por campaña."""
import numpy as np
import pandas as pd
import xarray as xr

from gemelo.config import DATA_RAW


def campanas_oxigeno(cfg: dict) -> pd.DataFrame:
    u = cfg["umbrales"]
    ds = xr.open_dataset(DATA_RAW / "upct" / "oxigeno_l2.nc", decode_times=False)
    fechas = pd.to_datetime(ds.time.values, unit="D", origin="1970-01-01")

    # Limpieza 1: fechas imposibles (hay un registro con fecha 1970-01-01)
    # .load() PRIMERO (~600 MB en memoria): indexar con una lista de posiciones sobre el
    # fichero comprimido sin cargar lo relee paso a paso y tarda ~6 min; en memoria, segundos.
    validas = fechas >= "2000-01-01"
    o2 = ds["Oxigeno"].load().isel(time=np.where(validas)[0]).assign_coords(time=fechas[validas].normalize())
    # Limpieza 2: campañas duplicadas el mismo día -> media
    o2 = o2.groupby("time").mean().sortby("time")
    # Los 0,0 mg/L se mantienen: la anoxia de 2019 fue real; lo más probable es que la
    # interpolación recorte en 0 valores ligeramente negativos.

    # Fondo = nivel más profundo con dato en cada punto (máscara de todo el periodo)
    tiene = o2.notnull().any("time")
    nivel = xr.DataArray(np.arange(o2.sizes["level"]), dims="level")
    idx_fondo = (tiene * nivel).where(tiene).max("level")
    fondo = o2.isel(level=idx_fondo.fillna(0).astype(int)).where(idx_fondo.notnull())

    puntos = fondo.notnull().sum(["latitude", "longitude"])
    df = pd.DataFrame({
        "o2_sup_media": o2.isel(level=0).mean(["latitude", "longitude"]).to_series(),
        "o2_fondo_media": fondo.mean(["latitude", "longitude"]).to_series(),
        "o2_fondo_min": fondo.min(["latitude", "longitude"]).to_series(),
        "pct_hipoxia": (100 * (fondo < u["hipoxia_mg_l"]).sum(["latitude", "longitude"]) / puntos).to_series(),
        "pct_anoxia": (100 * (fondo < u["anoxia_mg_l"]).sum(["latitude", "longitude"]) / puntos).to_series(),
    })
    df.index.name = "fecha"
    return df
