"""Descarga de meteorología diaria ERA5 desde la API de archivo de Open-Meteo."""
import json
import time
import urllib.error
import urllib.parse
import urllib.request

import pandas as pd

from gemelo.config import DATA_RAW, cargar_config


def _pedir(url: str, intentos: int = 4, espera_s: int = 65) -> dict:
    """GET con reintentos: una petición de 45 años 'pesa' mucho en el límite por minuto
    de Open-Meteo y puede devolver 429; basta con esperar al minuto siguiente."""
    for i in range(intentos):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code != 429 or i == intentos - 1:
                raise
            print(f"  límite de peticiones (429), espero {espera_s} s…")
            time.sleep(espera_s)


def descargar_meteo(sobrescribir: bool = False):
    cfg = cargar_config()
    m = cfg["meteo"]
    destino = DATA_RAW / "meteo"
    destino.mkdir(parents=True, exist_ok=True)

    for nombre, p in m["puntos"].items():
        salida = destino / f"era5_{nombre}.parquet"
        if salida.exists() and not sobrescribir:
            print(f"[{nombre}] ya existe {salida.name}, se omite")
            continue
        params = {
            "latitude": p["lat"], "longitude": p["lon"],
            "start_date": m["inicio"], "end_date": cfg["periodo"]["fin"],
            "daily": ",".join(m["variables"]), "models": m["modelo"], "timezone": "UTC",
        }
        url = f"{m['url']}?{urllib.parse.urlencode(params)}"
        datos = _pedir(url)

        df = pd.DataFrame(datos["daily"])
        df["time"] = pd.to_datetime(df["time"])
        df = df.set_index("time")
        # guardamos la celda real que ha usado la API (no la pedida)
        df.attrs = {"lat_celda": datos["latitude"], "lon_celda": datos["longitude"]}
        df.to_parquet(salida)
        print(f"[{nombre}] celda {datos['latitude']}, {datos['longitude']} · {len(df)} días · "
              f"nulos {int(df.isna().sum().sum())} -> {salida}")
