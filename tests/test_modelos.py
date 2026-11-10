"""Tests de la lógica de modelado (líneas base, episodios y métricas). Datos SINTÉTICOS."""
import numpy as np
import pandas as pd
import pytest

from gemelo.modelos import hipoxia as H
from gemelo.modelos.sst import predicciones_base, rho_amortiguada


def test_rho_recupera_un_ar1():
    """En un AR(1) con phi=0.9, anom(t+h) ≈ phi^h · anom(t): rho debe salir ≈ 0.9^h."""
    rng = np.random.default_rng(0)
    phi, n = 0.9, 50_000
    a = np.zeros(n)
    for i in range(1, n):
        a[i] = phi * a[i - 1] + rng.normal()
    s = pd.Series(a)
    for h in [1, 7]:
        delta = s.shift(-h) - s
        ok = delta.notna()
        assert rho_amortiguada(s[ok], delta[ok]) == pytest.approx(phi ** h, abs=0.02)


def test_lineas_base():
    anom = pd.Series([1.0, -2.0, 0.5])
    b = predicciones_base(anom, rho=0.6)
    assert np.allclose(b["persistencia"], 0)            # la anomalía se mantiene
    assert np.allclose(b["climatologia"], -anom)        # la anomalía desaparece
    assert np.allclose(b["amortiguada"], -0.4 * anom)   # queda el 60 %


def test_ventana_y_purga_del_reentrenamiento():
    from gemelo.modelos import reentrenamiento as R
    v = R.ventana_evaluacion(pd.Timestamp("2026-09-30"), h=7, dias=180)
    assert v.fin == pd.Timestamp("2026-09-23")                        # su objetivo es el último dato conocido
    assert (v.fin - v.ini).days == 179
    idx = pd.date_range("2026-01-01", "2026-09-30", freq="D")
    m = R.mascara_entrenamiento(idx, v.ini, h=7)
    assert (idx[m] + pd.Timedelta(days=7) < v.ini).all()               # ningún objetivo de entrenamiento en la ventana
    assert (idx[~m] + pd.Timedelta(days=7) >= v.ini).all()


def test_regla_de_promocion():
    from gemelo.modelos.reentrenamiento import decidir
    assert decidir(1.0, 0.97, 0.02) == (True, pytest.approx(0.03))
    assert decidir(1.0, 0.99, 0.02)[0] is False
    assert decidir(1.0, 1.10, 0.02)[0] is False                         # empeorar nunca promociona


def test_episodios_y_anticipacion():
    idx = pd.date_range("2021-01-01", periods=10, freq="7D")
    y = pd.Series([0, 1, 1, 0, 0, 1, 0, 1, 1, 1], index=idx)
    ep = H.episodios(y)
    assert ep.dropna().nunique() == 3
    prob = pd.Series([0, 0.9, 0.1, 0, 0, 0.2, 0, 0.7, 0, 0], index=idx, dtype=float)
    m = H.metricas(y, prob)
    assert m["episodios_total"] == 3
    assert m["episodios_detectados"] == 2      # el 1º y el 3º tienen alguna alerta
    assert m["inicios_anticipados"] == 2       # alertas en el primer día del 1º y del 3º
    assert m["falsas_alarmas"] == 0
