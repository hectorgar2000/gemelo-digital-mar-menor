"""Descarga del oxígeno disuelto in situ del Servidor de Datos Científicos del Mar Menor (UPCT).

El servidor publica mapas interpolados (L2) por OPeNDAP. Se descarga por bloques de tiempo
para no cargar los ~600 MB del fichero en memoria de golpe.
"""
import xarray as xr

from gemelo.config import DATA_RAW, cargar_config

BLOQUE = 50  # pasos de tiempo por petición


def descargar_oxigeno(sobrescribir: bool = False):
    cfg = cargar_config()["upct"]
    destino = DATA_RAW / "upct"
    destino.mkdir(parents=True, exist_ok=True)
    salida = destino / "oxigeno_l2.nc"
    if salida.exists() and not sobrescribir:
        print(f"ya existe {salida}, se omite")
        return salida

    # decode_times=False: hay al menos un registro con fecha errónea; se decodifica y filtra luego.
    remoto = xr.open_dataset(cfg["thredds_opendap"], decode_times=False)
    n = remoto.sizes["time"]
    trozos = []
    for i in range(0, n, BLOQUE):
        print(f"  pasos {i}-{min(i + BLOQUE, n) - 1} de {n}")
        trozos.append(remoto[cfg["variable"]].isel(time=slice(i, i + BLOQUE)).load())
    da = xr.concat(trozos, dim="time")
    da.to_dataset().to_netcdf(salida, encoding={cfg["variable"]: {"zlib": True, "complevel": 4}})
    print(f"guardado {salida} ({salida.stat().st_size / 1e6:.1f} MB)")
    return salida
