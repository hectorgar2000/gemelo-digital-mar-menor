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
