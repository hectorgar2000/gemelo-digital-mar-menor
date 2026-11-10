"""Tests de integración de la API. Necesitan los modelos exportados (models/servicio) y los datos
procesados; si no existen (p. ej. en un clon limpio), se saltan con un aviso."""
import pytest

from gemelo.servicio.motor import DIR_DATOS, DIR_MODELOS

pytestmark = pytest.mark.skipif(
    not (DIR_MODELOS / "servicio.json").exists() or not (DIR_DATOS / "dataset_sst.parquet").exists(),
    reason="Faltan models/servicio o data/processed: ejecuta el pipeline, entrena y exporta.",
)


@pytest.fixture(scope="module")
def cliente():
    from fastapi.testclient import TestClient

    from gemelo.servicio.api import app
    with TestClient(app) as c:  # `with` ejecuta el ciclo de vida (carga de modelos)
        yield c


def test_salud(cliente):
    r = cliente.get("/salud")
    assert r.status_code == 200
    assert r.json()["modelos_sst"] == 8


def test_predict_reconstruye_sst_y_da_intervalo(cliente):
    r = cliente.post("/predict", json={"zona": "laguna", "horizontes": [1, 7], "fecha": "2024-08-01"})
    assert r.status_code == 200
    preds = r.json()["predicciones"]
    assert [p["h"] for p in preds] == [1, 7]
    for p in preds:
        lo, hi = p["intervalo_90"]
        assert lo < p["sst_prevista"] < hi
        assert p["metodo_intervalo"] in ("conformal_estacional", "conformal_movil_365d")
        assert p["sst_real"] is not None and p["visto_en_entrenamiento"] is False
        assert 0 <= p["prob_superar_umbral"] <= 1


def test_predict_elige_modelo_con_o_sin_prevision(cliente):
    """Con previsión archivada (2024-02+) usa v2 para h=1/3/7; antes de 2024, el respaldo v1. h=30 siempre v1."""
    con = cliente.post("/predict", json={"zona": "costa", "horizontes": [7, 30], "fecha": "2025-03-01"}).json()["predicciones"]
    sin = cliente.post("/predict", json={"zona": "costa", "horizontes": [7], "fecha": "2023-03-01"}).json()["predicciones"]
    assert con[0]["usa_prevision_meteo"] and con[0]["modelo"]["variante"] == "con_prevision"
    assert not con[1]["usa_prevision_meteo"]
    assert not sin[0]["usa_prevision_meteo"] and sin[0]["modelo"]["variante"] == "sin_prevision"
    m = cliente.get("/modelos").json()["sst"]
    assert m["sst_costa_h7"]["usa_prevision_meteo"] and m["sst_costa_h7"]["respaldo"] is not None
    assert m["sst_costa_h30"]["respaldo"] is None


def test_predict_sin_fecha_usa_el_ultimo_dia(cliente):
    ultima = cliente.get("/salud").json()["ultima_fecha"]
    p = cliente.post("/predict", json={"zona": "costa", "horizontes": [3]}).json()["predicciones"][0]
    assert p["fecha_base"] == ultima
    assert p["sst_real"] is None  # el futuro aún no se conoce


def test_explain_suma_shap_igual_a_prediccion(cliente):
    r = cliente.post("/explain", json={"zona": "laguna", "h": 7, "fecha": "2024-08-01", "top": 5})
    assert r.status_code == 200
    e = r.json()
    assert len(e["variables"]) == 5
    # los grupos suman TODAS las variables (no solo el top): base + grupos = predicción
    total = e["valor_base"] + sum(g["contribucion"] for g in e["grupos"])
    assert total == pytest.approx(e["prediccion"], abs=1e-2)  # SHAP es aditivo


def test_hipoxia(cliente):
    r = cliente.post("/hipoxia/predict", json={"fecha": "2021-07-27"})
    assert r.status_code == 200
    h = r.json()
    assert 0 <= h["prob_hipoxia_siguiente"] <= 1
    assert h["visto_en_entrenamiento"] is True


@pytest.mark.parametrize("ruta,cuerpo", [
    ("/predict", {"zona": "laguna", "horizontes": [7], "fecha": "1990-01-01"}),   # sin datos
    ("/predict", {"zona": "laguna", "horizontes": [5]}),                          # horizonte inexistente
    ("/explain", {"zona": "costa", "h": 2}),
    ("/hipoxia/predict", {"fecha": "2021-07-28"}),                                # no hay campaña ese día
])
def test_errores_claros(cliente, ruta, cuerpo):
    r = cliente.post(ruta, json=cuerpo)
    assert r.status_code == 422
    assert isinstance(r.json()["detail"], str) and len(r.json()["detail"]) > 10


def test_estado_series_y_campanas(cliente):
    e = cliente.get("/estado", params={"fecha": "2024-08-01"}).json()
    lag = e["zonas"]["laguna"]
    assert lag["ola_calor_en_curso"] == (lag["dias_sobre_umbral"] >= 5)
    assert e["ultima_campana_hipoxia"]["fecha"] <= "2024-08-01"
    s = cliente.get("/series", params={"zona": "laguna", "desde": "2024-07-01", "hasta": "2024-07-31"}).json()
    assert len(s["datos"]) == 31 and {"sst", "climatologia", "umbral_ola_calor", "anomalia"} <= set(s["datos"][0])
    assert cliente.get("/series", params={"zona": "laguna", "desde": "1990-01-01", "hasta": "2024-01-01"}).status_code == 422
    c = cliente.get("/hipoxia/campanas").json()["datos"]
    assert len(c) > 300 and "pct_fondo_hipoxia" in c[0]


def test_zona_invalida_la_rechaza_pydantic(cliente):
    assert cliente.post("/predict", json={"zona": "atlantico"}).status_code == 422
