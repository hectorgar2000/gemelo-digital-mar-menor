"""API HTTP del gemelo digital del Mar Menor (MLaaS).

Arrancar en local:
    uvicorn gemelo.servicio.api:app --reload
Documentación interactiva: http://127.0.0.1:8000/docs
"""
from contextlib import asynccontextmanager
from datetime import date
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from gemelo.servicio.escenarios import DRIVERS, NO_DISPONIBLES, NOTAS, MotorEscenarios, Perturbacion
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
    version="0.5.0",
    lifespan=ciclo_de_vida,
)


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
    dias: int = Field(default=30, ge=1, le=365, description="Últimos días a los que se aplica")


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
                    "metricas_test": v["metricas_test"]} for k, v in m["sst"].items()},
        "hipoxia": {k: m["hipoxia"][k] for k in ("version_registro", "umbral_alerta", "metricas_loyo")},
    }


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
        "drivers": {k: {"tipo": v["tipo"], "unidad": v["unidad"], "descripcion": v["descripcion"], "nota": NOTAS[k]}
                    for k, v in DRIVERS.items()},
        "no_disponibles": NO_DISPONIBLES,
    }


@app.post("/scenario", summary="Escenario what-if frente al escenario base, con incertidumbre (Monte Carlo)")
def scenario(p: PeticionEscenario):
    me: MotorEscenarios = ESTADO["escenarios"]
    perts = [Perturbacion(x.driver, x.valor, x.incertidumbre, x.dias) for x in p.perturbaciones]
    for x in perts:
        if DRIVERS[x.driver]["tipo"] == "factor" and x.valor < 0:
            raise HTTPException(422, f"El driver '{x.driver}' es un factor (×): el valor no puede ser negativo.")
    out = {"sst": _ejecutar(me.simular_sst, p.zona, p.horizontes, p.fecha, perts, p.n_muestras, p.semilla)}
    if p.hipoxia is not None:
        out["hipoxia"] = _ejecutar(me.simular_hipoxia, p.hipoxia.fecha_campana, perts, p.n_muestras, p.semilla,
                                   p.hipoxia.dias_hasta_siguiente)
    return out
