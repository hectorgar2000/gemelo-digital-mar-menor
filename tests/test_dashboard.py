"""Tests del dashboard con Streamlit AppTest contra una API real arrancada en un hilo (puerto libre).
Se saltan si faltan los modelos exportados o los datos procesados."""
import socket
import threading
import time

import pytest

from gemelo.config import RAIZ
from gemelo.servicio.motor import DIR_DATOS, DIR_MODELOS

pytestmark = pytest.mark.skipif(
    not (DIR_MODELOS / "servicio.json").exists() or not (DIR_DATOS / "series_diarias.parquet").exists(),
    reason="Faltan models/servicio o data/processed.",
)
APP = str(RAIZ / "src" / "gemelo" / "dashboard" / "app.py")  # AppTest resuelve rutas relativas al test


@pytest.fixture(scope="module")
def api_url():
    import uvicorn

    from gemelo.servicio.api import app
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    puerto = s.getsockname()[1]
    s.close()
    servidor = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=puerto, log_level="warning"))
    hilo = threading.Thread(target=servidor.run, daemon=True)
    hilo.start()
    for _ in range(120):
        if servidor.started:
            break
        time.sleep(0.25)
    yield f"http://127.0.0.1:{puerto}"
    servidor.should_exit = True
    hilo.join(timeout=10)


def abrir(api_url):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    at.text_input[0].set_value(api_url).run()
    return at


def test_carga_sin_errores_y_muestra_estado(api_url):
    at = abrir(api_url)
    assert not at.exception, at.exception
    etiquetas = [m.label for m in at.metric]
    assert "Temperatura del agua" in etiquetas and "Prob. de ola de calor a 7 días" in etiquetas
    assert len(at.error) == 0


def test_cambiar_zona_y_dia(api_url):
    at = abrir(api_url)
    at.radio[0].set_value("costa").run()
    import datetime
    at.date_input[0].set_value(datetime.date(2024, 8, 1)).run()
    assert not at.exception, at.exception
    assert at.header[0].value.endswith("01/08/2024")
    assert len(at.error) == 0
    # día del pasado fuera de entrenamiento: la tabla de predicciones incluye el valor real
    tabla = at.dataframe[0].value
    assert tabla["Real"].notna().any()


def test_escenario_calentamiento_base(api_url):
    at = abrir(api_url)
    boton = next(b for b in at.button if b.label == "Simular escenario")
    assert boton.disabled  # sin cambios no se puede simular
    at.slider(key="v_calentamiento_base").set_value(1.5).run()
    boton = next(b for b in at.button if b.label == "Simular escenario")
    assert not boton.disabled
    boton.click().run()
    assert not at.exception, at.exception
    r = at.session_state["escenario"]["respuesta"]
    h7 = next(h for h in r["sst"]["horizontes"] if h["h"] == 7)
    assert h7["diferencia_sst"]["p50"] == pytest.approx(1.5, abs=1e-6)
    assert len(at.error) == 0


def test_avisa_si_la_campana_de_hipoxia_es_antigua(api_url):
    at = abrir(api_url)  # último día (2026): la última campaña es de 2024
    assert any("días** antes del día base" in w.value for w in at.warning)


def test_url_incorrecta_muestra_error_claro(api_url):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    at.text_input[0].set_value("http://127.0.0.1:9").run()  # puerto sin servicio
    assert not at.exception
    assert at.error and "No se puede conectar" in at.error[0].value
