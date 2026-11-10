"""Cliente HTTP de la API del gemelo. El dashboard solo habla con la API (no carga modelos)."""
import os

import httpx
import pandas as pd

URL_POR_DEFECTO = os.environ.get("GEMELO_API_URL", "http://127.0.0.1:8000")
# URL de la API vista desde el NAVEGADOR del usuario (para enlaces como el visor 4D). En Docker, el dashboard
# habla con la API por la red interna (http://api:8000), que el navegador no puede resolver.
URL_PUBLICA = os.environ.get("GEMELO_API_URL_PUBLICA", URL_POR_DEFECTO)


class ErrorAPI(RuntimeError):
    pass


class ClienteGemelo:
    def __init__(self, url: str = URL_POR_DEFECTO, timeout: float = 60.0):
        self.url = url.rstrip("/")
        self.http = httpx.Client(base_url=self.url, timeout=timeout)

    def _pedir(self, metodo: str, ruta: str, **kw):
        try:
            r = self.http.request(metodo, ruta, **kw)
        except httpx.HTTPError as e:
            raise ErrorAPI(f"No se puede conectar con la API en {self.url} ({e.__class__.__name__}). "
                           f"¿Está arrancada? `uvicorn gemelo.servicio.api:app`") from e
        if r.status_code >= 400:
            detalle = r.json().get("detail", r.text) if "json" in r.headers.get("content-type", "") else r.text
            raise ErrorAPI(f"{r.status_code}: {detalle}")
        return r.json()

    # lectura
    def salud(self):
        return self._pedir("GET", "/salud")

    def modelos(self):
        return self._pedir("GET", "/modelos")

    def estado(self, fecha=None):
        return self._pedir("GET", "/estado", params={"fecha": fecha} if fecha else None)

    def series(self, zona, desde, hasta) -> pd.DataFrame:
        d = self._pedir("GET", "/series", params={"zona": zona, "desde": desde, "hasta": hasta})["datos"]
        return pd.DataFrame(d).assign(fecha=lambda x: pd.to_datetime(x["fecha"])).set_index("fecha")

    def campanas(self) -> pd.DataFrame:
        d = self._pedir("GET", "/hipoxia/campanas")["datos"]
        return pd.DataFrame(d).assign(fecha=lambda x: pd.to_datetime(x["fecha"])).set_index("fecha")

    def monitor(self):
        return self._pedir("GET", "/monitor")

    def drivers(self):
        return self._pedir("GET", "/scenario/drivers")

    # modelos
    def predecir(self, zona, horizontes, fecha=None):
        return self._pedir("POST", "/predict", json={"zona": zona, "horizontes": horizontes, "fecha": fecha})["predicciones"]

    def explicar(self, zona, h, fecha=None, top=12):
        return self._pedir("POST", "/explain", json={"zona": zona, "h": h, "fecha": fecha, "top": top})

    def hipoxia(self, fecha=None, dias_hasta_siguiente=7):
        return self._pedir("POST", "/hipoxia/predict", json={"fecha": fecha, "dias_hasta_siguiente": dias_hasta_siguiente})

    def escenario(self, **peticion):
        return self._pedir("POST", "/scenario", json=peticion)
