"""Descarga de previsiones meteorológicas REALES archivadas (Open-Meteo Previous Runs API).

Para cada hora H y antelación k (1..7 días), `<var>_previous_dayk` es el valor que se PREVEÍA para H en la
ejecución del modelo de k días antes. Es lo que de verdad se sabía entonces: permite evaluar un modelo que
usa previsión meteorológica sin hacer trampas con la meteorología observada.
"""
import pandas as pd

from gemelo.config import DATA_RAW, cargar_config
from gemelo.ingesta.meteo import _pedir


def _tramos(inicio, fin, meses=3):
    ini = pd.Timestamp(inicio)
    fin = pd.Timestamp(fin)
    while ini <= fin:
        sig = min(ini + pd.DateOffset(months=meses) - pd.Timedelta(days=1), fin)
        yield ini, sig
        ini = sig + pd.Timedelta(days=1)


def descargar_previsiones(sobrescribir: bool = False):
    cfg = cargar_config()
    p = cfg["prevision"]
    destino = DATA_RAW / "meteo"
    destino.mkdir(parents=True, exist_ok=True)
    variables = [f"{v}_previous_day{k}" for v in p["variables_horarias"] for k in range(1, p["dias_max"] + 1)]

    for nombre, punto in cfg["meteo"]["puntos"].items():
        salida = destino / f"prevision_{nombre}.parquet"
        if salida.exists() and not sobrescribir:
            print(f"[{nombre}] ya existe {salida.name}, se omite")
            continue
        celda = punto.get("celda", punto)
        partes = []
        # Hasta fin + dias_max: el último día base necesita la previsión de sus 7 días siguientes
        # (emitida ese mismo día o antes, así que no es información del futuro).
        fin_prev = pd.Timestamp(cfg["periodo"]["fin"]) + pd.Timedelta(days=p["dias_max"])
        for ini, fin in _tramos(p["inicio"], fin_prev):
            params = (f"latitude={celda['lat']}&longitude={celda['lon']}&start_date={ini.date()}&end_date={fin.date()}"
                      f"&hourly={','.join(variables)}&timezone=UTC")
            datos = _pedir(f"{p['url']}?{params}")
            df = pd.DataFrame(datos["hourly"])
            df["time"] = pd.to_datetime(df["time"])
            partes.append(df.set_index("time"))
            print(f"[{nombre}] {ini.date()} → {fin.date()}: {len(df)} horas", flush=True)
        df = pd.concat(partes).sort_index()
        df.attrs = {"lat_celda": datos["latitude"], "lon_celda": datos["longitude"]}
        df.to_parquet(salida)
        nulos = df.isna().mean().mul(100).round(1)
        print(f"[{nombre}] celda {datos['latitude']:.3f}, {datos['longitude']:.3f} · {len(df)} horas · "
              f"% nulos máx {nulos.max()} ({nulos.idxmax()}) -> {salida}")
