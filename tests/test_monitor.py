"""Tests de las funciones de deriva del monitor. Datos SINTÉTICOS."""
import numpy as np
import pandas as pd
import pytest

from gemelo.config import cargar_config
from gemelo.monitor import metricas as D

CFG_MON = cargar_config()["monitor"]


def test_psi_detecta_desplazamientos():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 5000)
    assert D.psi(ref, rng.normal(0, 1, 2000)) < 0.02          # misma distribución
    assert 0.10 < D.psi(ref, rng.normal(0.5, 1, 2000)) < 0.4  # desplazamiento moderado
    assert D.psi(ref, rng.normal(2.0, 1, 2000)) > 1.0          # desplazamiento fuerte
    assert np.isnan(D.psi(ref[:5], ref))                        # referencia insuficiente


def test_mascara_estacional_compara_mismas_fechas():
    ref = pd.date_range("2000-01-01", "2001-12-31", freq="D")
    ventana = pd.date_range("2025-07-01", "2025-07-31", freq="D")
    m = D.mascara_estacional(ref, ventana, margen=5)
    meses = set(ref[m].month)
    assert meses == {6, 7, 8}                                   # finales de junio a principios de agosto
    # cruza fin de año (circular)
    m2 = D.mascara_estacional(ref, pd.date_range("2025-12-28", "2026-01-03", freq="D"), margen=3)
    assert set(ref[m2].month) == {12, 1}


def test_fraccion_fuera_de_rango():
    X = pd.DataFrame({"a": [0, 5, 11], "b": [1, 1, 1]})
    assert D.fraccion_fuera_de_rango(X, pd.Series({"a": 0, "b": 0}), pd.Series({"a": 10, "b": 2})) == pytest.approx(1 / 3)


def test_intervalos_conformales_son_causales_y_tienen_respaldo():
    idx = pd.date_range("2024-01-01", periods=400, freq="D")
    res = pd.Series(np.where(idx < "2024-07-01", 0.0, 5.0), index=idx)   # cambio brusco en julio
    iv = D.intervalos_conformales(res, h=7, ventana=365, min_residuos=60, q_fijos=(-1.0, 1.0))
    assert iv.loc["2024-01-10", "lo"] == -1.0                             # pocos residuos: cuantiles fijos
    assert iv.loc["2024-06-20", "hi"] == 0.0                              # ventana anterior al cambio
    # el 2024-07-03 aún no puede "ver" el cambio de julio (usa residuos hasta t-7 = 2024-06-26)
    assert iv.loc["2024-07-03", "hi"] == 0.0
    assert iv.loc["2025-01-15", "hi"] == 5.0


CFG_INT = cargar_config()["intervalo"]


def test_cuantiles_conformales_en_cascada_y_estacionales():
    idx = pd.date_range("2020-01-01", "2024-12-31", freq="D")
    # errores pequeños en invierno y grandes en verano
    res = pd.Series(np.where(idx.month.isin([6, 7, 8]), 3.0, 0.5) * np.sign(np.sin(np.arange(len(idx)))), index=idx)
    dias = idx.values.astype("datetime64[D]").astype(np.int64)
    doy = D._doy(idx)
    lo_v, hi_v, met_v, _ = D.cuantiles_conformales(dias, doy, res.values, pd.Timestamp("2024-07-15"), 7, CFG_INT, (-9, 9))
    lo_i, hi_i, met_i, _ = D.cuantiles_conformales(dias, doy, res.values, pd.Timestamp("2024-01-15"), 7, CFG_INT, (-9, 9))
    assert met_v == met_i == "conformal_estacional"
    assert hi_v - lo_v > 4 * (hi_i - lo_i)                      # verano mucho más ancho que invierno
    # sin historia suficiente: respaldo con cuantiles fijos
    _, _, met, _ = D.cuantiles_conformales(dias[:30], doy[:30], res.values[:30], pd.Timestamp("2020-02-15"), 7,
                                           CFG_INT, (-9, 9))
    assert met == "cuantiles_validacion"


def test_interval_score_premia_estrechos_que_aciertan():
    y, pred = np.zeros(4), np.zeros(4)
    assert D.interval_score(y, pred, -1, 1) == pytest.approx(2.0)          # acierta: solo cuenta el ancho
    assert D.interval_score(y + 2, pred, -1, 1) == pytest.approx(2 + 20)   # falla por 1: + (2/0,1)·1


def test_estado():
    assert D.estado(0.10, 0.88, 0.05, 0.0, CFG_MON) == ("OK", [])
    e, m = D.estado(-0.05, 0.88, 0.05, 0.0, CFG_MON)
    assert e == "DERIVA" and "persistencia" in m[0]
    assert D.estado(0.10, 0.70, 0.05, 0.0, CFG_MON)[0] == "DERIVA"
    assert D.estado(0.10, 0.88, 0.40, 0.0, CFG_MON)[0] == "AVISO"
    assert D.estado(0.10, 0.88, 0.05, 0.30, CFG_MON)[0] == "AVISO"
