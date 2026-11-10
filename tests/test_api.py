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
    modelos = cliente.get("/modelos").json()["sst"]
    for p in preds:
        lo, hi = p["intervalo_90"]
        assert lo < p["sst_prevista"] < hi
        assert p["sst_real"] is not None and 0 <= p["prob_superar_umbral"] <= 1
        # coherencia con lo que vio el modelo (cambia tras cada reentrenamiento)
        ficha = modelos[p["modelo"]["nombre"]]
        hasta = ficha["datos_hasta"] if p["modelo"]["variante"] == "con_prevision" or not ficha["respaldo"] else None
        if hasta is not None:
            visto = p["fecha_objetivo"] <= hasta
            assert p["visto_en_entrenamiento"] is visto
            # si el modelo ya vio ese periodo no hay errores fuera de muestra recientes: cuantiles de validación
            if visto:
                assert p["metodo_intervalo"] in ("cuantiles_validacion", "conformal_estacional", "conformal_movil_365d")
            else:
                assert p["metodo_intervalo"] in ("conformal_estacional", "conformal_movil_365d")


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


def test_monitor(cliente):
    r = cliente.get("/monitor")
    if r.status_code == 404:
        pytest.skip("sin informe del monitor (python scripts/monitorizar.py)")
    j = r.json()
    assert j["estado_global"] in ("OK", "AVISO", "DERIVA")
    assert set(j["modelos"]) == {f"sst_{z}_h{h}" for z in ("laguna", "costa") for h in (1, 3, 7, 30)}
    assert j["hipoxia"]["estado"] in ("OK", "SIN_DATOS")


def test_cubo_4d(cliente):
    f = cliente.get("/cubo/fechas")
    if f.status_code == 422:
        pytest.skip("sin cubo 4D (python scripts/construir_dataset.py)")
    meta = f.json()
    assert {"o2", "temp", "sal"} <= set(meta["variables"]) and meta["variables"]["o2"]["unidad"] == "mg/L"
    assert meta["profundidades"][0] == 0.5 and meta["profundidades"][-1] == 5.0
    campana = max(meta["campanas"], key=lambda c: c["pct_volumen_anoxia"])
    c = cliente.get("/cubo", params={"fecha": campana["fecha"], "variable": "o2"}).json()
    assert c["n"] == len(c["lon"]) == len(c["lat"]) == len(c["profundidad"]) == len(c["valor"]) > 1000
    assert all(37.6 < y < 37.85 for y in c["lat"]) and all(-0.9 < x < -0.7 for x in c["lon"])
    # el % de volumen anóxico declarado coincide con los puntos enviados
    anoxia = 100 * sum(v < 2.0 for v in c["valor"]) / c["n"]
    assert anoxia == pytest.approx(c["resumen"]["pct_volumen_anoxia"], abs=0.5)
    fechas = {x["fecha"] for x in meta["campanas"]}
    sin_campana = next(d for d in ("2019-12-25", "2020-01-01", "2021-12-25") if d not in fechas)
    assert cliente.get("/cubo", params={"fecha": sin_campana, "variable": "o2"}).status_code == 422
    assert cliente.get("/cubo", params={"fecha": campana["fecha"], "variable": "nitratos"}).status_code == 422


def test_visor_4d_se_sirve_con_cesium_fijado(cliente):
    r = cliente.get("/visor/")
    assert r.status_code == 200 and "cesium@1.120.0" in r.text and "visor.js" in r.text
    assert cliente.get("/visor/visor.js").status_code == 200


def test_zona_invalida_la_rechaza_pydantic(cliente):
    assert cliente.post("/predict", json={"zona": "atlantico"}).status_code == 422
