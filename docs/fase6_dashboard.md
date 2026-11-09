# Fase 6 — Dashboard del gemelo

*2026-11-09*

## Arrancar

```bash
docker compose up --build        # API http://localhost:8000/docs · dashboard http://localhost:8501
docker compose down              # parar
```

En local, sin Docker, se necesitan dos terminales:

```bash
uvicorn gemelo.servicio.api:app
streamlit run src/gemelo/dashboard/app.py
```

## Arquitectura

```
Navegador ──► Dashboard (Streamlit, cliente HTTP) ──► API (FastAPI + motor del gemelo) ──► modelos exportados + datos
```

El dashboard **no carga modelos**: todo lo pide a la API (`gemelo.dashboard.cliente`). Así:
- Lo que ve el usuario es exactamente lo que sirve el MLaaS.
- Si se promociona un modelo, el dashboard lo usa sin tocarlo.
- La imagen del dashboard no lleva XGBoost ni SHAP.

Para eso se añadieron a la API tres endpoints de lectura:
- `GET /estado`: foto del día, ola de calor en curso, última campaña y geometría.
- `GET /series`: SST con climatología y umbral.
- `GET /hipoxia/campanas`: historial de campañas.

## Pestañas

| Pestaña | Qué muestra |
|---|---|
| 🌡️ Estado y predicción | Indicadores del día, **mapa** de subzonas coloreadas por anomalía, celdas ERA5 reales, serie de SST con climatología y umbral, y predicción a 1/3/7/30 días con **intervalo del 90 %**. En días del pasado muestra **lo que pasó** y avisa si el día se usó para entrenar. |
| 🧪 Escenarios | Controles para agua (episodio o clima de fondo), aire, lluvia, viento, radiación y clorofila, cada uno con incertidumbre y duración. Muestra base frente a escenario, probabilidad de ola de calor, **aviso fuera de rango**, notas de interpretación, **qué explica el cambio** (SHAP) e hipoxia. |
| 🔍 Por qué (XAI) | SHAP de la predicción por grupo y por variable, con guía de lectura. |
| 🫧 Hipoxia | Historial de campañas, probabilidad en la siguiente, alerta temprana y contribuciones de la logística. |
| ⚙️ Modelos | Versiones en `@produccion` y métricas de test. |

## Decisiones de interfaz
- **Avisos honestos**:
  - El día cae en el periodo de entrenamiento.
  - Variables fuera de rango.
  - **Campaña de hipoxia antigua**: el 30-09-2026 la última campaña in situ es del 18-07-2024, a 804 días.
- Escenarios **sin `st.form`**: los campos de incertidumbre aparecen al mover un control. La simulación (unos 2 s) solo se lanza con el botón y el resultado se conserva en la sesión.
- Caché de 10 minutos en las llamadas a la API (`st.cache_data`).

## Tests
`tests/test_dashboard.py` usa **Streamlit AppTest** contra una API real arrancada en un hilo:
- Carga sin errores.
- Cambio de zona y de día (con el valor real).
- Escenario completo (calentamiento de fondo de +1,5 °C → diferencia de +1,5 °C exacta).
- Aviso de campaña antigua.
- Error claro si la API no responde.

Total del proyecto: **49 tests**.

## Docker
| Imagen | Tamaño | Contenido |
|---|---|---|
| `gemelo-mar-menor:0.6.0` | 948 MB | API + modelos (montados) |
| `gemelo-mar-menor-dashboard:0.6.0` | 889 MB | Streamlit + Plotly + pandas/pyarrow |

El dashboard arranca cuando la API está *healthy* (`depends_on: condition: service_healthy`). Se verificó que el dashboard del contenedor muestra los datos servidos por el contenedor de la API.
