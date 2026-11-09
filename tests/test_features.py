"""Tests del procesado. Usan series SINTÉTICAS generadas aquí (no son datos reales):
solo sirven para comprobar propiedades del código, sobre todo que no hay fuga de futuro."""
import numpy as np
import pandas as pd
import pytest

from gemelo.config import cargar_config
from gemelo.procesado import features as F


@pytest.fixture(scope="module")
def cfg():
    return cargar_config()


def diario_sintetico(cfg, semilla=0):
    """Tabla con las mismas columnas que data/interim/diario.parquet, valores inventados."""
    rng = np.random.default_rng(semilla)
    idx = pd.date_range("1982-01-01", cfg["periodo"]["fin"], freq="D", name="fecha")
    estac = 5 * np.sin(2 * np.pi * (idx.day_of_year - 120) / 365)
    cols = {}
    for z in F.ZONAS:
        cols[f"sst_{z}"] = 19 + estac + rng.normal(0, 0.5, len(idx)).cumsum() * 0.05
        cols[f"logchl_{z}"] = rng.normal(0, 0.2, len(idx))
    for c in ["thetao_costa", "bottomT_costa", "mld_costa"]:
        cols[c] = rng.normal(15, 1, len(idx))
    for p in cfg["meteo"]["puntos"]:
        for v in ["t2m", "t2max", "prec", "viento", "vientomax", "rad", "viento_u", "viento_v"]:
            cols[f"{v}_{p}"] = rng.normal(10, 3, len(idx))
    return pd.DataFrame(cols, index=idx)


@pytest.fixture(scope="module")
def base(cfg):
    d = diario_sintetico(cfg)
    return d, F.construir_dataset_sst(d, cfg)


def columnas_x(df):
    return [c for c in df.columns if not c.startswith("y_") and c != "particion"]


@pytest.mark.parametrize("corte", ["2020-06-15", "2023-03-01"])
def test_variables_no_miran_al_futuro(cfg, base, corte):
    """Si cambio los datos DESPUÉS de un corte, las variables hasta el corte no cambian.

    El corte va en validación/test a propósito: la climatología de anomalías se calcula con
    todo el periodo de entrenamiento (práctica estándar), así que un corte dentro de train sí
    movería las anomalías. Lo que debe ser imposible es que validación o test 'contaminen'."""
    d, ds = base
    corte = pd.Timestamp(corte)
    d2 = d.copy()
    d2.loc[d2.index > corte] += 7.0  # perturbación grande del futuro
    ds2 = F.construir_dataset_sst(d2, cfg)
    pd.testing.assert_frame_equal(ds.loc[:corte, columnas_x(ds)], ds2.loc[:corte, columnas_x(ds2)])


def test_climatologia_solo_usa_datos_anteriores_a_validacion(cfg):
    d = diario_sintetico(cfg)
    s = d["sst_laguna"]
    fin = str(pd.Timestamp(cfg["particion"]["validacion"][0]) - pd.Timedelta(days=1))
    c1 = F.climatologia(s, "1982-01-01", fin)
    s2 = s.copy()
    s2[cfg["particion"]["validacion"][0]:] += 10
    c2 = F.climatologia(s2, "1982-01-01", fin)
    pd.testing.assert_series_equal(c1, c2)


@pytest.mark.parametrize("h", [1, 3, 7, 30])
def test_objetivo_alineado_con_el_horizonte(base, h):
    _, ds = base
    for z in F.ZONAS:
        y = ds[f"y_sst_{z}_h{h}"]
        actual = ds[f"sst_{z}"]
        t = ds.index[100]
        assert y.loc[t] == pytest.approx(actual.loc[t + pd.Timedelta(days=h)])


def test_particion_temporal_ordenada(base):
    _, ds = base
    fechas = {p: ds.index[ds.particion == p] for p in ["train", "validacion", "test"]}
    assert all(len(f) for f in fechas.values())
    assert fechas["train"].max() < fechas["validacion"].min()
    assert fechas["validacion"].max() < fechas["test"].min()


@pytest.mark.parametrize("h", [1, 7, 30])
def test_purga_de_frontera(cfg, base, h):
    _, ds = base
    part = cfg["particion"]
    for nombre in ["train", "validacion"]:
        filas = F.filas_de_particion(ds, nombre, h, part)
        assert (filas.index + pd.Timedelta(days=h)).max() <= pd.Timestamp(part[nombre][1])
        assert (filas.particion == nombre).all()


def test_dias_desde_es_causal():
    idx = pd.date_range("2000-01-01", periods=6, freq="D")
    r = F.dias_desde(pd.Series([False, True, False, False, True, False], index=idx), tope=99)
    assert r.tolist() == [99, 0, 1, 2, 0, 1]


def test_mhw_exige_cinco_dias_seguidos():
    idx = pd.date_range("2000-01-01", periods=20, freq="D")
    umbral = pd.Series(20.0, index=range(1, 366))
    valores = [21] * 4 + [19] + [21] * 5 + [19] * 10  # racha de 4 (no) y de 5 (sí)
    mhw = F.flag_mhw(pd.Series(valores, index=idx, dtype=float), umbral, min_dias=5)
    assert not mhw.iloc[:4].any()
    assert mhw.iloc[5:10].all()
    assert not mhw.iloc[10:].any()


def test_racha_es_causal():
    idx = pd.date_range("2000-01-01", periods=6, freq="D")
    umbral = pd.Series(20.0, index=range(1, 366))
    r = F.racha_sobre_umbral(pd.Series([21, 21, 19, 21, 21, 21], index=idx, dtype=float), umbral)
    assert r.tolist() == [1, 2, 0, 1, 2, 3]
