# Gemelo digital marino — Mar Menor

Réplica virtual del **Mar Menor y su costa adyacente** alimentada con datos oceanográficos reales (Copernicus Marine y la UPCT), sobre la que unos modelos de IA:

1. **Describen** el estado actual del agua (temperatura, clorofila, oxígeno).
2. **Predicen** la temperatura superficial y el riesgo de **ola de calor marina** a 1, 3, 7 y 30 días.
3. **Simulan escenarios** *what-if* (p. ej. +1,5 °C) con incertidumbre.
4. **Explican** sus predicciones con XAI (SHAP).

Módulo secundario: riesgo de **hipoxia** en la laguna a partir de los datos in situ de la UPCT.

Los modelos se sirven como **MLaaS**: entrenados, versionados en MLflow y expuestos con FastAPI.

Alcance detallado: [docs/alcance.md](docs/alcance.md).

## Estructura

```
config/zona.yaml     zona, periodos, umbrales y dataset_id (fuente única de verdad)
data/raw/            descargas originales (fuera de git)
data/interim/        datos intermedios (fuera de git)
data/processed/      dataset de entrenamiento en Zarr/Parquet (fuera de git)
docs/                documentación del proyecto
models/              modelos entrenados (fuera de git)
notebooks/           exploración
scripts/             puntos de entrada ejecutables (descarga, entrenamiento…)
src/gemelo/          paquete Python: ingesta, procesado, modelos, xai
tests/               tests
```

## Puesta en marcha (Windows)

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[datos,ml,api,dev]"
python -m ipykernel install --sys-prefix --name gemelo --display-name "Python (gemelo)"
copernicusmarine login
python scripts/descargar_copernicus.py
python scripts/descargar_upct.py
python scripts/descargar_meteo.py
python scripts/construir_dataset.py
pytest
python scripts/entrenar_sst.py
python scripts/entrenar_hipoxia.py
mlflow ui --backend-store-uri sqlite:///mlflow.db
python scripts/registrar_modelos.py     # Model Registry, alias @produccion
python scripts/exportar_modelos.py      # -> models/servicio (artefacto de la API)
uvicorn gemelo.servicio.api:app         # http://127.0.0.1:8000/docs
streamlit run src/gemelo/dashboard/app.py   # dashboard (necesita la API arrancada)
python scripts/ciclo_mantenimiento.py   # monitor de deriva -> reentrenamiento si hace falta -> exportar
docker compose up --build               # API (:8000) + dashboard (:8501) en contenedores
```

Los notebooks usan el kernel **Python (gemelo)**.

Las credenciales de Copernicus se guardan en `%USERPROFILE%\.copernicusmarine`, nunca en el repositorio.

## Roadmap

| Fase | Entregable | Estado |
|---|---|---|
| 0 · Planteamiento | Repo y documento de alcance | ✅ |
| 1 · Ingesta y exploración | Datos en `data/raw/` y notebook de exploración ([resumen](docs/fase1_resumen.md)) | ✅ |
| 2 · Procesado | Pipeline `raw → processed` ([resumen y revisión](docs/fase2_resumen.md)) | ✅ |
| 3 · Primer modelo + XAI | XGBoost que bate a la línea base, con SHAP ([informe](docs/fase3_informe.md)) | ✅ |
| 4 · MLaaS mínimo | API `/predict` y `/explain` con Docker ([detalle](docs/fase4_mlaas.md)) | ✅ |
| 5 · Escenarios | `/scenario` con incertidumbre ([detalle](docs/fase5_escenarios.md)) | ✅ |
| 6 · Dashboard | Streamlit de punta a punta ([detalle](docs/fase6_dashboard.md)) | ✅ |
| 7 · Evolución | Previsión meteo como entrada ✅ ([detalle](docs/fase7_prevision.md)) · deriva y reentrenamiento ✅ ([detalle](docs/fase7_deriva.md)) · 3D/4D | ⏳ |

## Fuentes de datos
- [Copernicus Marine Service](https://marine.copernicus.eu) — SST, clorofila, física y biogeoquímica del Mediterráneo.
- [Servidor de Datos Científicos del Mar Menor (UPCT)](https://marmenor.upct.es) — oxígeno disuelto in situ.
