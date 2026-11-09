"""raw -> interim: de los cubos in situ de la UPCT a una tabla con una fila por campaña.

Por cada variable se calcula la media en superficie (0,5 m) y en el fondo (nivel más profundo con
dato en cada punto). Para el oxígeno, además, el mínimo de fondo y el % del área en hipoxia/anoxia.
La estratificación (diferencia fondo - superficie) es lo que impide que el oxígeno llegue al fondo.
"""
import numpy as np
import pandas as pd
import xarray as xr

from gemelo.ingesta.upct import ruta_upct

PREFIJO = {"Oxigeno": "o2", "Temperatura": "temp", "Salinidad": "sal", "Clorofila": "chl", "Turbidez": "turb"}


def cargar_cubo(variable: str) -> xr.DataArray:
    """Cubo (time, level, lon, lat) limpio: fechas válidas, un paso por día, ordenado."""
    ds = xr.open_dataset(ruta_upct(variable), decode_times=False)
    fechas = pd.to_datetime(ds.time.values, unit="D", origin="1970-01-01")
    validas = fechas >= "2000-01-01"  # hay registros con fecha 1970-01-01
    # .load() PRIMERO: indexar con posiciones sobre el NetCDF comprimido sin cargar tarda minutos.
    da = ds[variable].load().isel(time=np.where(validas)[0])
    da = da.assign_coords(time=fechas[validas].normalize())
    return da.groupby("time").mean().sortby("time")  # campañas duplicadas el mismo día -> media


def valor_fondo(da: xr.DataArray) -> xr.DataArray:
    """Valor en el nivel más profundo con dato de cada punto (máscara de todo el periodo)."""
    tiene = da.notnull().any("time")
    nivel = xr.DataArray(np.arange(da.sizes["level"]), dims="level")
    idx = (tiene * nivel).where(tiene).max("level")
    return da.isel(level=idx.fillna(0).astype(int)).where(idx.notnull())


def _media(da):
    return da.mean(["latitude", "longitude"]).to_series()


def campanas_insitu(cfg: dict) -> pd.DataFrame:
    u = cfg["umbrales"]
    cols = {}
    for variable in cfg["upct"]["variables"]:
        p = PREFIJO[variable]
        cubo = cargar_cubo(variable)
        fondo = valor_fondo(cubo)
        cols[f"{p}_sup_media"] = _media(cubo.isel(level=0))
        cols[f"{p}_fondo_media"] = _media(fondo)
        if variable == "Oxigeno":
            # Los 0,0 mg/L se mantienen: la anoxia de 2019 fue real (probable recorte en 0 de la interpolación).
            puntos = fondo.notnull().sum(["latitude", "longitude"])
            cols["o2_fondo_min"] = fondo.min(["latitude", "longitude"]).to_series()
            cols["pct_hipoxia"] = (100 * (fondo < u["hipoxia_mg_l"]).sum(["latitude", "longitude"]) / puntos).to_series()
            cols["pct_anoxia"] = (100 * (fondo < u["anoxia_mg_l"]).sum(["latitude", "longitude"]) / puntos).to_series()

    # La tabla se indexa por las campañas de OXÍGENO (las del objetivo); el resto se alinea a ellas.
    df = pd.DataFrame(cols).dropna(subset=["o2_fondo_media"])
    # Control de calidad (ver config upct.sal_sup_min_plausible): NaN, nunca se inventa un valor
    dudosos = df["sal_sup_media"] < cfg["upct"]["sal_sup_min_plausible"]
    if dudosos.any():
        print(f"    QC: {int(dudosos.sum())} campañas con salinidad superficial no plausible -> NaN "
              f"({', '.join(str(d.date()) for d in df.index[dudosos])})")
    df.loc[dudosos, "sal_sup_media"] = np.nan
    df["estrat_sal"] = df["sal_fondo_media"] - df["sal_sup_media"]     # >0: agua más salada (densa) abajo
    df["estrat_temp"] = df["temp_sup_media"] - df["temp_fondo_media"]  # >0: agua más cálida arriba
    df.index.name = "fecha"
    return df
