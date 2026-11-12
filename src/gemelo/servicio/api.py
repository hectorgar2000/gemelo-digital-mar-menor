"""API HTTP del gemelo digital del Mar Menor (MLaaS).

Arrancar en local:
    uvicorn gemelo.servicio.api:app --reload
Documentación interactiva: http://127.0.0.1:8000/docs
"""
from contextlib import asynccontextmanager
from datetime import date
from typing import Literal

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from gemelo.servicio.escenarios import (DRIVERS, METEO_FUTURO, NO_DISPONIBLES, NOTAS, PERIODOS, MotorEscenarios,
                                        Perturbacion)
from gemelo.servicio.motor import Gemelo, ErrorGemelo

ESTADO = {}


@asynccontextmanager
async def ciclo_de_vida(app: FastAPI):
    ESTADO["gemelo"] = Gemelo()  # modelos y datos se cargan UNA vez al arrancar
    ESTADO["escenarios"] = MotorEscenarios(ESTADO["gemelo"])
    yield
    ESTADO.clear()


app = FastAPI(
    title="Gemelo digital del Mar Menor",
    description="Predicción de temperatura del mar y olas de calor marinas (laguna y costa) y riesgo de "
                "hipoxia, con intervalos de incertidumbre y explicaciones SHAP.",
    version="0.10.0",
    lifespan=ciclo_de_vida,
)


# Visor 4D (CesiumJS): página estática servida por la propia API (mismo origen que /cubo)
app.mount("/visor", StaticFiles(directory=Path(__file__).parent / "visor", html=True), name="visor")


def gemelo() -> Gemelo:
    return ESTADO["gemelo"]


def _ejecutar(funcion, *args, **kwargs):
    try:
        return funcion(*args, **kwargs)
    except ErrorGemelo as e:
        raise HTTPException(status_code=422, detail=str(e)) from e


# ------------------------------------------------------------------ esquemas
class PeticionPrediccion(BaseModel):
    zona: Literal["laguna", "costa"] = Field(description="Subzona del gemelo")
    horizontes: list[int] = Field(default=[1, 3, 7, 30], description="Días hacia delante (1, 3, 7 y/o 30)")
    fecha: date | None = Field(default=None, description="Día base. Vacío = último día con datos")

    model_config = {"json_schema_extra": {"examples": [{"zona": "laguna", "horizontes": [7], "fecha": "2024-08-01"}]}}


class PeticionExplicacion(BaseModel):
    zona: Literal["laguna", "costa"]
    h: int = Field(description="Horizonte en días (1, 3, 7 o 30)")
    fecha: date | None = None
    top: int = Field(default=10, ge=1, le=50, description="Nº de variables a devolver")

    model_config = {"json_schema_extra": {"examples": [{"zona": "laguna", "h": 7, "fecha": "2024-08-01", "top": 8}]}}


class PeticionHipoxia(BaseModel):
    fecha: date | None = Field(default=None, description="Fecha de una campaña de muestreo. Vacío = la última")
    dias_hasta_siguiente: int = Field(default=7, ge=3, le=14, description="Días hasta la siguiente campaña")

    model_config = {"json_schema_extra": {"examples": [{"fecha": "2021-07-27", "dias_hasta_siguiente": 7}]}}


class PerturbacionIn(BaseModel):
    driver: Literal[tuple(DRIVERS)] = Field(description="Qué se perturba (ver GET /scenario/drivers)")
    valor: float = Field(description="Suma (°C) o factor (×), según el driver")
    incertidumbre: float = Field(default=0.0, ge=0, description="Desviación típica del valor (Monte Carlo)")
    dias: int = Field(default=30, ge=1, le=365,
                      description="Días a los que se aplica: los últimos (pasado) o los próximos (futuro, máx. 7)")
    periodo: Literal["pasado", "futuro", "ambos"] | None = Field(
        default=None, description="Vacío = 'futuro' para aire/lluvia/viento/radiación (perturba la previsión) y "
                                  "'pasado' para el resto. El agua y la clorofila solo admiten 'pasado'.")


class HipoxiaEscenario(BaseModel):
    fecha_campana: date | None = Field(default=None, description="Campaña base. Vacío = la última")
    dias_hasta_siguiente: int = Field(default=7, ge=3, le=14)


class PeticionEscenario(BaseModel):
    zona: Literal["laguna", "costa"]
    horizontes: list[int] = Field(default=[1, 7, 30])
    fecha: date | None = Field(default=None, description="Día base. Vacío = último día con datos")
    perturbaciones: list[PerturbacionIn] = Field(default=[], max_length=6)
    hipoxia: HipoxiaEscenario | None = Field(default=None, description="Si se indica, simula también la hipoxia")
    n_muestras: int = Field(default=30, ge=5, le=200, description="Muestras de la incertidumbre de la perturbación")
    semilla: int = 42

    model_config = {"json_schema_extra": {"examples": [{
        "zona": "laguna", "horizontes": [7, 30], "fecha": "2024-06-15",
        "perturbaciones": [{"driver": "temp_agua", "valor": 1.5, "incertidumbre": 0.3, "dias": 30}],
        "hipoxia": {"fecha_campana": "2024-06-10"}}]}}


# ------------------------------------------------------------------ endpoints
@app.get("/salud", summary="Estado del servicio")
def salud():
    g = gemelo()
    return {"estado": "ok", "modelos_sst": len(g.modelos), "ultima_fecha": str(g.ultima_fecha().date()),
            "ultima_campana": str(g.campanas.max().date())}


@app.get("/modelos", summary="Modelos en servicio y sus métricas")
def modelos():
    m = gemelo().manifiesto
    return {
        "exportado": m["creado"], "git_commit": m["git_commit"],
        "sst": {k: {"zona": v["zona"], "h": v["h"], "version_registro": v["version_registro"],
                    "usa_prevision_meteo": v.get("usa_prevision", False), "metricas": v["metricas"],
                    "datos_hasta": v.get("datos_hasta"), "receta": v.get("receta", "clasica"),
                    "con_parte_lineal": bool(v.get("lineal")),
                    "respaldo": ({"version_registro": v["respaldo"]["version_registro"],
                                  "metricas": v["respaldo"]["metricas"]} if v.get("respaldo") else None)}
                for k, v in m["sst"].items()},
        "hipoxia": {k: m["hipoxia"][k] for k in ("version_registro", "umbral_alerta", "metricas_loyo")},
    }


@app.get("/cubo/fechas", summary="Campañas del cubo 4D in situ, con volumen en hipoxia y variables disponibles")
def cubo_fechas():
    return _ejecutar(gemelo().cubo_fechas)


@app.get("/cubo", summary="Puntos 3D (lon, lat, profundidad, valor) de una campaña in situ para el visor 4D")
def cubo(fecha: date, variable: Literal["o2", "temp", "sal", "chl", "turb"] = "o2"):
    return _ejecutar(gemelo().cubo, fecha, variable)


@app.get("/monitor", summary="Último informe del monitor de deriva (estado de cada modelo)")
def monitor():
    from gemelo.servicio.motor import DIR_DATOS
    ruta = DIR_DATOS / "monitor.json"
    if not ruta.exists():
        raise HTTPException(404, "Aún no hay informe del monitor: python scripts/monitorizar.py")
    import json
    return json.loads(ruta.read_text(encoding="utf-8"))


def _tabla(df) -> list[dict]:
    out = df.round(3).reset_index()
    out.iloc[:, 0] = out.iloc[:, 0].dt.strftime("%Y-%m-%d")
    return out.rename(columns={out.columns[0]: "fecha"}).to_dict(orient="records")


@app.get("/estado", summary="Estado del gemelo un día: SST, anomalía, ola de calor en curso y última campaña")
def estado(fecha: date | None = None):
    return _ejecutar(gemelo().estado, fecha)


@app.get("/series", summary="Serie diaria de SST observada con climatología y umbral de ola de calor")
def series(zona: Literal["laguna", "costa"], desde: date | None = None, hasta: date | None = None):
    return {"zona": zona, "datos": _tabla(_ejecutar(gemelo().serie_sst, zona, desde, hasta))}


@app.get("/hipoxia/campanas", summary="Historial de campañas in situ (oxígeno y temperatura de fondo)")
def campanas():
    return {"datos": _tabla(gemelo().historial_campanas())}


@app.post("/predict", summary="Predicción de SST y ola de calor con intervalo del 90 %")
def predict(p: PeticionPrediccion):
    g = gemelo()
    return {"predicciones": [_ejecutar(g.predecir, p.zona, h, p.fecha) for h in p.horizontes]}


@app.post("/explain", summary="Explicación SHAP de una predicción (por grupos y variables)")
def explain(p: PeticionExplicacion):
    return _ejecutar(gemelo().explicar, p.zona, p.h, p.fecha, p.top)


@app.post("/hipoxia/predict", summary="Riesgo de hipoxia en la próxima campaña, con explicación")
def hipoxia(p: PeticionHipoxia):
    return _ejecutar(gemelo().predecir_hipoxia, p.fecha, p.dias_hasta_siguiente)


@app.get("/scenario/drivers", summary="Qué se puede perturbar en un escenario (y qué no)")
def drivers():
    return {
        "drivers": {k: {"tipo": v["tipo"], "unidad": v["unidad"], "descripcion": v["descripcion"], "nota": NOTAS[k],
                        "periodos": list(PERIODOS) if k in METEO_FUTURO else ["pasado"]}
                    for k, v in DRIVERS.items()},
        "no_disponibles": NO_DISPONIBLES,
    }


@app.post("/scenario", summary="Escenario what-if frente al escenario base, con incertidumbre (Monte Carlo)")
def scenario(p: PeticionEscenario):
    me: MotorEscenarios = ESTADO["escenarios"]
    try:
        perts = [Perturbacion(x.driver, x.valor, x.incertidumbre, x.dias, x.periodo) for x in p.perturbaciones]
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    for x in perts:
        if DRIVERS[x.driver]["tipo"] == "factor" and x.valor < 0:
            raise HTTPException(422, f"El driver '{x.driver}' es un factor (×): el valor no puede ser negativo.")
    out = {"sst": _ejecutar(me.simular_sst, p.zona, p.horizontes, p.fecha, perts, p.n_muestras, p.semilla)}
    if p.hipoxia is not None:
        out["hipoxia"] = _ejecutar(me.simular_hipoxia, p.hipoxia.fecha_campana, perts, p.n_muestras, p.semilla,
                                   p.hipoxia.dias_hasta_siguiente)
    return out
