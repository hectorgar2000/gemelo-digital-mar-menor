"""Tests de las variables de meteorología futura (previsión). Datos SINTÉTICOS."""
import numpy as np
import pandas as pd
import pytest

from gemelo.procesado import meteo_futura as M
from gemelo.procesado.features import grupo_variable

IDX = pd.date_range("2024-01-01", periods=60, freq="D")
REF = pd.DataFrame({"clim_t2m_cuenca": np.zeros(365)}, index=range(1, 366))


def diario_sintetico():
    n = len(IDX)
    d = pd.DataFrame(index=IDX)
    for v in ["t2m", "t2max", "prec", "viento", "vientomax", "rad", "viento_u", "viento_v"]:
        d[f"{v}_cuenca"] = np.arange(n, dtype=float)  # valor = nº de día: fácil de comprobar
    return d


def test_ventana_es_t_mas_1_a_t_mas_w():
    d = diario_sintetico()
    f = M.construir(M.fuente_era5(d), ["cuenca"], REF, IDX)
    t = 10
    assert f["prev1_prec_cuenca"].iloc[t] == t + 1                     # solo mañana, nunca hoy
    assert f["prev3_prec_cuenca"].iloc[t] == (t + 1) + (t + 2) + (t + 3)
    assert f["prev7_t2max_cuenca"].iloc[t] == t + 7
    assert f["prev3_viento_cuenca"].iloc[t] == pytest.approx(t + 2)
    assert np.isnan(f["prev7_prec_cuenca"].iloc[-3])                    # falta futuro: NaN, no se inventa


def test_no_depende_de_mas_alla_de_7_dias():
    d = diario_sintetico()
    f1 = M.construir(M.fuente_era5(d), ["cuenca"], REF, IDX)
    d2 = d.copy()
    d2.iloc[30:] += 100                                                  # cambia el futuro lejano
    f2 = M.construir(M.fuente_era5(d2), ["cuenca"], REF, IDX)
    pd.testing.assert_frame_equal(f1.iloc[:30 - 7], f2.iloc[:30 - 7])   # días base con t+7 < 30: iguales


def horaria_sintetica(valor_por_antelacion=lambda k: float(k)):
    h_idx = pd.date_range("2024-01-01", periods=60 * 24, freq="h")
    cols = {}
    for k in range(1, 8):
        cols[f"temperature_2m_previous_day{k}"] = valor_por_antelacion(k)
        cols[f"precipitation_previous_day{k}"] = 0.5
        cols[f"wind_speed_10m_previous_day{k}"] = 10.0
        cols[f"wind_direction_10m_previous_day{k}"] = 90.0   # viento del este
        cols[f"shortwave_radiation_previous_day{k}"] = 1000.0
    return pd.DataFrame(cols, index=h_idx)


def test_agregado_diario_de_la_prevision():
    d = M.diaria_desde_horaria(horaria_sintetica(), 2)
    dia = d.iloc[5]
    assert dia["t2m"] == 2 and dia["prec"] == pytest.approx(12.0)       # 0,5 mm/h × 24 h
    assert dia["rad"] == pytest.approx(86.4)                            # 1000 W/m² × 24 h = 86,4 MJ/m²
    assert dia["viento_u"] == pytest.approx(-10.0) and dia["viento_v"] == pytest.approx(0.0, abs=1e-9)


def test_cada_dia_de_la_ventana_usa_su_antelacion():
    """El día t+k debe salir de la previsión emitida k días antes (previous_dayk), no de otra."""
    prev = {"cuenca": M.prevision_por_antelacion(horaria_sintetica())}   # previsión con antelación k vale k
    f = M.construir(M.fuente_prevision(prev), ["cuenca"], REF, IDX)
    assert f["prev1_t2m_anom_cuenca"].iloc[10] == pytest.approx(1.0)
    assert f["prev3_t2m_anom_cuenca"].iloc[10] == pytest.approx((1 + 2 + 3) / 3)
    assert f["prev7_t2max_cuenca"].iloc[10] == pytest.approx(7.0)


def test_correccion_de_sesgo():
    horaria = horaria_sintetica(lambda k: 5.0)                           # previsión siempre 5 °C
    prev = M.prevision_por_antelacion(horaria)
    era5 = diario_sintetico() * 0
    era5["t2m_cuenca"] = 3.0                                             # observado 3 °C -> sesgo +2
    s = M.sesgo_prevision(prev, era5, "cuenca", "2024-01-01", "2024-01-31")
    assert s.loc[1, "t2m"] == pytest.approx(2.0)
    corr = M.corregir(prev, s)
    assert corr[3]["t2m"].iloc[40] == pytest.approx(3.0)


def test_grupo_meteo_prevista():
    assert grupo_variable("prev3_prec_cuenca") == "meteo_prevista"
    assert grupo_variable("prec_cuenca_suma7") == "lluvia"               # no confundir con 'prev'
