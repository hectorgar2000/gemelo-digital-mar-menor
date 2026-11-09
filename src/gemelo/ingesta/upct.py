"""Descarga de las variables in situ del Servidor de Datos Científicos del Mar Menor (UPCT).

El servidor publica mapas interpolados (L2) por OPeNDAP, un fichero por variable. Se descarga
por bloques de tiempo para no pedir ~600 MB de golpe.
"""
import xarray as xr

from gemelo.config import DATA_RAW, cargar_config

BLOQUE = 50  # pasos de tiempo por petición


def ruta_upct(variable: str):
    return DATA_RAW / "upct" / f"{variable.lower()}_l2.nc"


def descargar_variable(variable: str, url_base: str, sobrescribir: bool = False):
    salida = ruta_upct(variable)
    salida.parent.mkdir(parents=True, exist_ok=True)
    if salida.exists() and not sobrescribir:
        print(f"[{variable}] ya existe {salida.name}, se omite")
        return salida

    # decode_times=False: hay registros con fecha errónea; se decodifica y filtra en el procesado.
    remoto = xr.open_dataset(f"{url_base}/{variable}.nc", decode_times=False)
    n = remoto.sizes["time"]
    trozos = []
    for i in range(0, n, BLOQUE):
        print(f"[{variable}] pasos {i}-{min(i + BLOQUE, n) - 1} de {n}", flush=True)
        trozos.append(remoto[variable].isel(time=slice(i, i + BLOQUE)).load())
    da = xr.concat(trozos, dim="time")
    da.to_dataset().to_netcdf(salida, encoding={variable: {"zlib": True, "complevel": 4}})
    print(f"[{variable}] guardado {salida} ({salida.stat().st_size / 1e6:.1f} MB)")
    return salida


def descargar_todo(sobrescribir: bool = False):
    cfg = cargar_config()["upct"]
    for v in cfg["variables"]:
        descargar_variable(v, cfg["thredds_opendap"], sobrescribir)
