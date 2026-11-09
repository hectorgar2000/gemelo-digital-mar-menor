"""Tests del motor de escenarios: coherencia con /predict, neutralidad, método delta y avisos."""
import numpy as np
import pandas as pd
import pytest

from gemelo.servicio import escenarios as E
from gemelo.servicio.motor import DIR_DATOS, DIR_MODELOS

hay_modelos = pytest.mark.skipif(
    not (DIR_MODELOS / "servicio.json").exists() or not (DIR_DATOS / "series_diarias.parquet").exists(),
    reason="Faltan models/servicio o data/processed: ejecuta el pipeline, entrena y exporta.",
)
FECHA = "2024-06-15"


# ------------------------------------------------------------------ unitarios (sin modelos)
def test_perturbar_solo_dentro_de_la_ventana():
    idx = pd.date_range("2024-01-01", periods=60, freq="D")
    t = pd.DataFrame({"sst_laguna": 20.0, "prec_cuenca": 2.0, "logchl_laguna": 0.0}, index=idx)
    fecha = idx[-1]
    perts = [E.Perturbacion("temp_agua", 1.5, dias=10), E.Perturbacion("lluvia", 3.0, dias=10),
             E.Perturbacion("clorofila", 10.0, dias=10)]
    p = E.perturbar_series(t, fecha, perts, [1.5, 3.0, 10.0])
    dentro = p.index > fecha - pd.Timedelta(days=10)
    assert dentro.sum() == 10
    assert (p.loc[dentro, "sst_laguna"] == 21.5).all() and (p.loc[~dentro, "sst_laguna"] == 20.0).all()
    assert (p.loc[dentro, "prec_cuenca"] == 6.0).all() and (p.loc[~dentro, "prec_cuenca"] == 2.0).all()
    assert np.allclose(p.loc[dentro, "logchl_laguna"], 1.0)  # ×10 en clorofila = +1 en log10
    assert (t["sst_laguna"] == 20.0).all()  # no modifica el original


def test_muestreo_reproducible_y_centrado():
    perts = [E.Perturbacion("temp_agua", 1.5, 0.3), E.Perturbacion("lluvia", 2.0, 0.5)]
    a, b = E.muestrear(perts, 200, 7), E.muestrear(perts, 200, 7)
    assert np.array_equal(a, b)
    assert a[:, 0].mean() == pytest.approx(1.5, abs=0.05) and a[:, 0].std() == pytest.approx(0.3, abs=0.05)
    assert (a[:, 1] >= 0).all()
    assert E.muestrear([E.Perturbacion("temp_agua", 1.5)], 200, 7).shape == (1, 1)  # sin incertidumbre: 1 muestra


def test_todos_los_drivers_tienen_nota():
    assert set(E.NOTAS) == set(E.DRIVERS)


# ------------------------------------------------------------------ integración (con modelos)
@pytest.fixture(scope="module")
def motor():
    from gemelo.servicio.motor import Gemelo
    g = Gemelo()
    return g, E.MotorEscenarios(g)


@hay_modelos
def test_sin_perturbacion_coincide_con_predict(motor):
    g, me = motor
    r = me.simular_sst("laguna", [1, 7, 30], FECHA, [])
    for x in r["horizontes"]:
        p = g.predecir("laguna", x["h"], FECHA)
        assert x["base"]["sst_puntual"] == pytest.approx(p["sst_prevista"], abs=1e-3)
        assert x["escenario"]["sst_puntual"] == pytest.approx(p["sst_prevista"], abs=1e-3)
        assert x["fuera_de_rango"] == []


@hay_modelos
@pytest.mark.parametrize("driver,neutro", [("temp_agua", 0.0), ("temp_aire", 0.0), ("lluvia", 1.0), ("viento", 1.0)])
def test_perturbacion_neutra_no_cambia_nada(motor, driver, neutro):
    _, me = motor
    r = me.simular_sst("costa", [7], FECHA, [E.Perturbacion(driver, neutro, 0.0, 30)])
    x = r["horizontes"][0]
    assert x["diferencia_sst"]["p50"] == pytest.approx(0.0, abs=1e-9)
    assert x["escenario"]["sst_puntual"] == pytest.approx(x["base"]["sst_puntual"], abs=1e-9)


@hay_modelos
def test_calentamiento_base_es_desplazamiento_exacto(motor):
    _, me = motor
    r = me.simular_sst("laguna", [7, 30], FECHA, [E.Perturbacion("calentamiento_base", 1.5)])
    for x in r["horizontes"]:
        assert x["diferencia_sst"]["p05"] == pytest.approx(1.5) and x["diferencia_sst"]["p95"] == pytest.approx(1.5)
        assert x["cambio_explicado_por_grupo"] == []          # el modelo no ve el cambio: es el método delta
        assert x["escenario"]["prob_ola_calor"] >= x["base"]["prob_ola_calor"]


@hay_modelos
def test_escenario_extremo_avisa_fuera_de_rango(motor):
    _, me = motor
    x = me.simular_sst("laguna", [7], FECHA, [E.Perturbacion("temp_agua", 8.0)])["horizontes"][0]
    variables = {a["variable"] for a in x["fuera_de_rango"]}
    assert {"anom_laguna_lag0", "sst_laguna"} <= variables


@hay_modelos
def test_cuantiles_ordenados(motor):
    _, me = motor
    x = me.simular_sst("laguna", [7], FECHA, [E.Perturbacion("temp_agua", 1.0, 0.5)])["horizontes"][0]
    for k in ("base", "escenario"):
        assert x[k]["sst_p05"] <= x[k]["sst_p50"] <= x[k]["sst_p95"]
    d = x["diferencia_sst"]
    assert d["p05"] < d["p50"] < d["p95"]  # con incertidumbre, la diferencia tiene dispersión


@hay_modelos
def test_hipoxia_neutra_coincide_con_prediccion(motor):
    g, me = motor
    h = me.simular_hipoxia("2024-06-10", [E.Perturbacion("lluvia", 1.0)])
    assert h["base"]["prob_hipoxia"] == pytest.approx(g.predecir_hipoxia("2024-06-10")["prob_hipoxia_siguiente"], abs=1e-3)
    assert h["escenario"]["prob_hipoxia"]["p50"] == pytest.approx(h["base"]["prob_hipoxia"], abs=1e-9)


@hay_modelos
def test_api_scenario(motor):
    from fastapi.testclient import TestClient

    from gemelo.servicio.api import app
    with TestClient(app) as c:
        d = c.get("/scenario/drivers").json()
        assert "nitratos" in d["no_disponibles"] and "calentamiento_base" in d["drivers"]
        r = c.post("/scenario", json={"zona": "laguna", "horizontes": [7], "fecha": FECHA,
                                      "perturbaciones": [{"driver": "temp_agua", "valor": 1.5, "incertidumbre": 0.3}],
                                      "hipoxia": {"fecha_campana": "2024-06-10"}})
        assert r.status_code == 200
        j = r.json()
        assert j["sst"]["horizontes"][0]["diferencia_sst"]["p50"] > 0 and "hipoxia" in j
        assert "temp_agua" in j["sst"]["notas"]
        assert c.post("/scenario", json={"zona": "laguna", "perturbaciones": [{"driver": "nitratos", "valor": 0.7}]}).status_code == 422
        assert c.post("/scenario", json={"zona": "laguna", "perturbaciones": [{"driver": "lluvia", "valor": -1}]}).status_code == 422
