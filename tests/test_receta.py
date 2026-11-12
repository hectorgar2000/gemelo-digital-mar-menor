"""Tests de la receta de entrenamiento, el modelo combinado (árboles + lineal) y las variables físicas previstas.
Datos SINTÉTICOS."""
import numpy as np
import pandas as pd
import pytest

from gemelo.modelos import receta as RC
from gemelo.modelos.comun import mejora_bootstrap
from gemelo.modelos.conjunto import ModeloSST
from gemelo.procesado import meteo_futura as M
from gemelo.procesado.features import PARES_AIRE_MAR, es_fisica, grupo_variable

PARAMS = {"max_depth": 3, "min_child_weight": 5}


@pytest.fixture(scope="module")
def datos():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2015-01-01", periods=1500, freq="D")
    X = pd.DataFrame(rng.normal(size=(1500, 5)), index=idx, columns=list("abcde"))
    y = pd.Series(2 * X["a"] - X["b"] + 0.5 * np.sin(3 * X["c"]) + rng.normal(0, 0.1, 1500), index=idx)
    return X, y


def test_pesos_recientes():
    idx = pd.date_range("2010-01-01", "2026-01-01", freq="YS")
    w = RC.pesos_recientes(idx, semivida_anios=8)
    assert w[-1] == pytest.approx(1.0) and w[-9] == pytest.approx(0.5, abs=0.01) and (np.diff(w) > 0).all()


def test_recetas(datos):
    X, y = datos
    clasica = RC.entrenar(X, y, PARAMS, 40, "clasica", h=7)
    nueva7 = RC.entrenar(X, y, PARAMS, 40, "fisica_pesos", h=7)
    nueva30 = RC.entrenar(X, y, PARAMS, 40, "fisica_pesos", h=30)
    assert clasica.lineal is None and nueva7.lineal is None
    assert clasica.arboles.n_estimators == 40 and nueva7.arboles.n_estimators == 100     # 2,5 veces más árboles
    assert nueva30.lineal is not None and nueva30.peso_lineal == 0.5                      # a 30 días, media con lineal
    with pytest.raises(ValueError):
        RC.entrenar(X, y, PARAMS, 40, "inventada", h=7)


def test_modelo_combinado_es_la_media_y_extrapola(datos):
    X, y = datos
    m = RC.entrenar(X, y, PARAMS, 40, "fisica_pesos", h=30)
    solo_arboles = ModeloSST(m.arboles)
    li = m.lineal
    lineal = ((X[li["variables"]].values - li["media"]) / li["escala"]) @ np.array(li["coef"]) + li["intercepto"]
    assert np.allclose(m.predict(X), 0.5 * solo_arboles.predict(X) + 0.5 * lineal)
    # fuera del rango visto, los árboles se saturan y la parte lineal sigue respondiendo
    lejos = X.iloc[[0]].assign(a=10.0)
    mas_lejos = X.iloc[[0]].assign(a=20.0)
    assert solo_arboles.predict(mas_lejos)[0] == pytest.approx(solo_arboles.predict(lejos)[0])
    assert m.predict(mas_lejos)[0] > m.predict(lejos)[0] + 1


@pytest.mark.parametrize("h", [7, 30])
def test_contribuciones_son_aditivas(datos, h):
    X, y = datos
    m = RC.entrenar(X, y, PARAMS, 40, "fisica_pesos", h=h)
    valores, base = m.contribuciones(X.iloc[:20])
    assert valores.shape == (20, 5)
    assert np.allclose(base + valores.sum(axis=1), m.predict(X.iloc[:20]), atol=1e-3)


def test_guardar_y_cargar_da_las_mismas_predicciones(datos, tmp_path):
    import json
    X, y = datos
    m = RC.entrenar(X, y, PARAMS, 40, "fisica_pesos", h=30)
    m.arboles.save_model(tmp_path / "m.json")
    lineal = json.loads(json.dumps(m.lineal))                # como viaja en el manifiesto
    m2 = ModeloSST.cargar(tmp_path / "m.json", lineal)
    assert np.allclose(m.predict(X), m2.predict(X), atol=1e-5)
    assert m2.variables == list(X.columns)


def test_fisica_prevista():
    cols = {}
    for p in ("cuenca", "mar"):
        for w in (1, 3, 7):
            cols[f"prev{w}_t2m_anom_{p}"] = [2.0, 0.0]
            cols[f"prev{w}_viento_{p}"] = [10.0, 5.0]
    prev = pd.DataFrame(cols)
    anom = pd.DataFrame({"anom_laguna_lag0": [0.5, 1.0], "anom_costa_lag0": [1.0, -1.0]}, index=[7, 9])  # otro índice
    f = M.fisica_prevista(prev, anom)
    assert len(f.columns) == len(PARES_AIRE_MAR) * 3 * 2
    assert f["prev3_aire_agua_cuenca_laguna"].tolist() == [1.5, -1.0]       # aire previsto - agua de hoy
    assert f["prev3_flujo_cuenca_laguna"].tolist() == [15.0, -5.0]          # × viento previsto
    assert all(es_fisica(c) and grupo_variable(c) == "meteo_prevista" for c in f.columns)
    assert grupo_variable("aire_agua_mar_costa_media7") == "intercambio_aire_mar"


def test_mejora_bootstrap():
    rng = np.random.default_rng(1)
    ref = rng.normal(0, 1.0, 400)
    lo, hi = mejora_bootstrap(ref, 0.8 * ref)
    assert lo == pytest.approx(0.2) and hi == pytest.approx(0.2)            # siempre un 20 % mejor
    lo, hi = mejora_bootstrap(ref, rng.permutation(ref))                     # mismos errores, en otro orden
    assert lo < 0 < hi                                                        # mismo RMSE global: el IC contiene el 0


def test_residuos_guardados_se_combinan_y_manda_el_error_en_vivo():
    from gemelo.servicio.motor import Gemelo
    vivo = pd.Series([9.0], index=pd.to_datetime(["2026-01-02"]))
    info = {"residuos_validacion": {"fechas": ["2026-01-01", "2026-01-02"], "residuos": [1.0, 2.0]}}
    r = Gemelo.con_residuos_guardados(vivo, info)
    assert r.index.is_monotonic_increasing and r.tolist() == [1.0, 9.0]      # el día repetido usa el error en vivo
    assert Gemelo.con_residuos_guardados(vivo, {}).equals(vivo)              # modelos antiguos: sin cambios
