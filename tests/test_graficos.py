"""Tests de los gráficos del dashboard (funciones puras) y del contorno de la laguna. Datos SINTÉTICOS."""
import pandas as pd
import pytest

from gemelo.dashboard import graficos as G
from gemelo.procesado.cubo import contorno_laguna


def test_contorno_laguna_es_un_poligono_cerrado_que_envuelve_los_puntos():
    # "laguna" triangular: filas de latitud cada vez más estrechas
    puntos = [(lon, lat) for lat, ancho in [(37.0, 4), (37.1, 3), (37.2, 1)] for lon in [-0.8 + 0.01 * i for i in range(ancho)]]
    cubo = pd.DataFrame([{"lon": lon, "lat": lat, "profundidad": p} for lon, lat in puntos for p in (0.5, 1.0)])
    pol = contorno_laguna(cubo)
    assert pol[0] == pol[-1] and len(pol) == 2 * 3 + 1                 # cerrado: 3 filas × (oeste + este)
    lons = [p[0] for p in pol]
    assert min(lons) < -0.8 and max(lons) > -0.77                       # llega al borde de la celda, no a su centro


def _estado(ola=True, contorno=True):
    zona = {"sst": 26.1, "anomalia": 2.4, "dias_sobre_umbral": 10, "ola_calor_en_curso": ola,
            "caja": {"lat": [37.6, 37.8], "lon": [-0.85, -0.74]}}
    return {"zonas": {"laguna": {**zona, "contorno": [[-0.8, 37.6], [-0.75, 37.7], [-0.8, 37.8], [-0.8, 37.6]] if contorno else None},
                      "costa": {**zona, "ola_calor_en_curso": False, "anomalia": -1.0, "contorno": None}},
            "puntos_meteo": {"cuenca": {"lat": 37.75, "lon": -1.0}}}


def test_mapa_usa_el_contorno_real_y_colorea_por_estado():
    fig = G.mapa(_estado(), "laguna")
    laguna, costa = fig.data[0], fig.data[2]
    assert list(laguna.lon) == [-0.8, -0.75, -0.8, -0.8]               # contorno real, no el recuadro
    assert len(costa.lon) == 5                                           # sin contorno: recuadro
    assert laguna.line.color == G.ROJO and costa.line.color == G.AZUL   # ola de calor / más fría de lo normal
    assert laguna.line.width > costa.line.width                          # la zona activa se resalta


def test_color_zona():
    base = {"ola_calor_en_curso": False, "anomalia": 0.0}
    assert G.color_zona(base) == G.GRIS
    assert G.color_zona({**base, "anomalia": 0.8}) == G.NARANJA
    assert G.color_zona({**base, "anomalia": 0.8, "ola_calor_en_curso": True}) == G.ROJO


def test_serie_sombrea_las_olas_de_calor_y_separa_lo_ocurrido_despues():
    idx = pd.date_range("2025-06-01", periods=30, freq="D")
    sst = [20.0] * 5 + [26.0] * 8 + [20.0] * 3 + [26.0] * 2 + [20.0] * 12   # una racha de 8 días y otra de 2
    serie = pd.DataFrame({"sst": sst, "climatologia": 21.0, "umbral_ola_calor": 23.0}, index=idx)
    fig = G.serie_con_prediccion(serie, [], "laguna", fecha_base="2025-06-20")
    sombreados = [s for s in fig.layout.shapes if s.type == "rect"]
    assert len(sombreados) == 1                                           # solo la racha de 5 días o más
    nombres = [t.name for t in fig.data]
    assert "Temperatura observada" in nombres and "Lo que ocurrió después" in nombres


def test_cascada_cuadra_y_agrupa_el_resto():
    grupos = [{"grupo": f"g{i}", "contribucion": c} for i, c in enumerate([0.5, -0.3, 0.2, -0.1, 0.05, 0.04, 0.03, 0.02, 0.01])]
    pred = 0.1 + sum(g["contribucion"] for g in grupos)
    w = G.cascada(grupos, 0.1, pred, max_grupos=4).data[0]
    assert list(w.x)[0] == "Punto de partida" and list(w.x)[-1] == "Predicción" and "Otros (5)" in list(w.x)
    assert w.y[0] + sum(w.y[1:-1]) == pytest.approx(w.y[-1])              # base + contribuciones = predicción


def test_aguja_cambia_de_color_con_la_alerta():
    assert G.aguja(0.2).data[0].gauge.bar.color == G.VERDE
    assert G.aguja(0.8).data[0].gauge.bar.color == G.ROJO
