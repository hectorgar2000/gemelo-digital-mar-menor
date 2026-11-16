# CLAUDE.md — Gemelo digital marino del Mar Menor

## Contexto del usuario
- Desarrollador en IA explicable (XAI), formación DAM. Domina programación general; está **aprendiendo** ML, gemelos digitales, GIS multidimensional y MLaaS.
- **Responder en español.** Explicar brevemente cada decisión técnica (sin teoría de más). Código claro antes que ingenioso.
- Windows 11, Python (3.13 instalado), venv en `.venv`. Equipo: 8 GB RAM, sin GPU NVIDIA → recortes pequeños y `dask`. Para entrenar modelos grandes, avisar y usar Colab.

## Reglas de trabajo
- **No inventar datos ni resultados.** Si algo falla o no se puede descargar, decirlo. Datos sintéticos solo avisando y marcándolos como tales.
- **Verificar** productos y `dataset_id` de Copernicus con `copernicusmarine describe` antes de usarlos. Los verificados están en `config/zona.yaml`.
- **Credenciales**: el usuario hace `copernicusmarine login`. Nunca escribirlas en código ni en ficheros del repo.
- Pasos pequeños y verificables: ejecutar y enseñar el resultado antes de avanzar.
- **No pasar de fase sin confirmación** del usuario (roadmap en `README.md`).
- No introducir herramientas de "más adelante" (Docker, PostGIS, TimescaleDB, Airflow, MQTT, CesiumJS, pytorch) hasta la fase que las necesite.
- Ante varias opciones razonables: recomendar una y justificar en 2-3 líneas.
- Partición de datos siempre **temporal**, nunca aleatoria.
- `data/`, `models/`, `mlruns/` y secretos fuera de git.

## Convenciones de código
- Paquete en `src/gemelo/`; scripts ejecutables en `scripts/`; exploración en `notebooks/`.
- Toda la configuración de zona, periodos, umbrales y datasets se lee de `config/zona.yaml`. Nada hardcodeado.
- `xarray` es la estructura central para los cubos (lat, lon, profundidad, tiempo).
- Nombres de variables y comentarios en español; nombres de variables de los datasets tal cual vienen (p. ej. `analysed_sst`, `thetao`).

## Hechos ya comprobados (2026-11-09)
- Los modelos de Copernicus a 1/24° (PHY y BGC) tratan **la laguna como tierra**: solo valen para la costa (mar abierto al este de ~0,73 W).
- SST satélite L4 (0,05°) y clorofila L4 (1 km) **sí** cubren la laguna.
- O₂ in situ: UPCT THREDDS `L2/Oxigeno.nc`, semanal, mar 2017 → jul 2024, 10 profundidades de 0,5 a 5 m, mg/L. Hay un registro con fecha errónea (1970) que hay que filtrar. El mismo THREDDS tiene `Temperatura`, `Salinidad`, `Clorofila`, `Turbidez`… con la misma rejilla.
- La SST de satélite en la laguna tiene sesgo estacional frente a la in situ (+1,9 °C en invierno, −2,3 °C en mayo; RMSE 1,7 °C).
- Meteorología: ERA5 vía Open-Meteo (sin cuenta). Viento en km/h. Las peticiones largas pueden devolver 429: el código reintenta.
- Datos más recientes (2026-11-16, ver `docs/propuestas_mejora.md` B1/B2): SST NRT 1 km (`SST_MED_SST_L4_NRT_OBSERVATIONS_010_004_c_V2`) llega a ayer, pero en la laguna difiere del REP (RMSE 0,74 °C incluso corregido): no usarlo ahí sin reentrenar. El REP ya llega a ~1 mes de retraso. **Datos actualizados a 2026-10-16** (`periodo.fin`): re-descargar no cambió ningún día anterior ni la climatología; modelos sin reentrenar (`datos_hasta` 2026-09-30, cooldown). `fisica_hflux`/`mflux` terminan en 2025-07-31 (no se usan como variables). Copia del estado anterior en el scratchpad de la sesión del 2026-11-16. Oxígeno: la UPCT termina en 2024-07-18 en todos sus catálogos; el IEO-CSIC (BELICH, boyas + CTD) tiene datos hasta 2026 pero sin descarga pública.
- Deriva: el test (2022–2026) es ~1 °C más cálido que train y tiene un 38 % de días de ola de calor (5 % en train).

## Decisiones para la Fase 3 (acordadas con el usuario)
- Regresión de SST: objetivo `y_delta_{zona}_h{h}` (cambio de anomalía); reconstruir SST = clim + anom_hoy + delta.
- Olas de calor: derivarlas de la SST predicha + umbral, no con un clasificador aparte.
- Hiperparámetros con validación de origen móvil; línea base obligatoria: persistencia amortiguada.
- SHAP agrupado con `features.grupo_variable`; adjuntar `data/processed/manifest.json` a cada run de MLflow.
- Hipoxia: validación dejando un año fuera, métricas por episodio, batir la persistencia (recall 0,73).

## Estado tras la Fase 3 (ver docs/fase3_informe.md)
- MLflow: `sqlite:///mlflow.db` + `mlartifacts/` (fuera de git). Experimentos `sst_mar_menor` y `hipoxia_mar_menor`. Usar `MLFLOW_DISABLE_AGENT_HINT=1`.
- SST: XGBoost (delta) bate a la persistencia amortiguada en los 8 casos; en la costa a 3–7 días empata con la persistencia simple. Intervalo conformal del 90 % → cobertura en test del 78–89 % (deriva).
- Hipoxia: la logística anticipa 4/9 inicios (28 falsas alarmas); XGBoost no supera a la persistencia. Para la API: logística como alerta temprana.

## Estado tras la Fase 4 (ver docs/fase4_mlaas.md)
- Dependencias por grupos: `.[datos,ml,api,dashboard,dev]` (+ `notebooks` aparte: JupyterLab falla en Windows con rutas largas). La API (`.[api]`) no lleva MLflow ni stack geoespacial.
- Flujo: entrenar → `registrar_modelos.py` (@produccion) → `exportar_modelos.py` (models/servicio) → API/Docker.
- `gemelo.servicio.motor.Gemelo` es el motor reutilizable (escenarios en la Fase 5 y dashboard en la Fase 6): `predecir(zona, h, fecha, X=...)` acepta una fila modificada.
- Intervalo: conformal con ventana móvil de 365 días (cobertura en test del 86–89 %).
- `GEMELO_RAIZ` fija la raíz del proyecto (necesario en Docker). Los parquet de `referencias_sst` los comparten el entrenamiento y la API.

## Estado tras la Fase 5 (ver docs/fase5_escenarios.md)
- `gemelo.servicio.escenarios.MotorEscenarios`: perturba las series diarias (`data/processed/series_diarias.parquet`) y recalcula las variables con `construir_dataset_sst(tramo, cfg, ref)`. No se perturban nunca las variables derivadas directamente.
- `temp_agua` (episodio) ≠ `calentamiento_base` (método delta). Los forzamientos meteorológicos son del pasado: hay poco efecto por diseño.
- Hipoxia en servicio: `hipoxia_logistica` **v2** (sin viento). Para promocionar una versión: entrenar → `registrar_modelos.py --modelos hipoxia` → `exportar_modelos.py`.

## Estado tras la Fase 6 (ver docs/fase6_dashboard.md)
- Dashboard `src/gemelo/dashboard/` (app, cliente HTTP, gráficos): solo consume la API, nunca carga modelos.
- Arranque local: `.claude/launch.json` (api :8000, dashboard :8501). Docker: `docker compose up --build`.
- Streamlit no recarga módulos importados: reiniciar el servidor tras editar `graficos.py` o `cliente.py`.
- Usar `width="stretch"` (no `use_container_width`, obsoleto).
- **No editar ficheros con Get-Content/Set-Content de PowerShell 5.1**: corrompe las tildes. Usar Edit/Write o Python.

## Fase 7 · bloque 1: previsión meteorológica (ver docs/fase7_prevision.md)
- Previsiones reales archivadas (Open-Meteo Previous Runs, antelación 1–7 d) completas desde 2024-02. Config: `prevision` en `config/zona.yaml`.
- `gemelo.procesado.meteo_futura`: variables `prev{w}_*` (w = 1/3/7) desde ERA5 (entrenamiento, "previsión perfecta") o desde la previsión real con sesgo corregido (feb–jul 2024). Miran al futuro A PROPÓSITO: van en tablas aparte (`meteo_futura_*.parquet`), nunca en `dataset_sst`.
- Resultado: −17/−26 % de error a 7 días y F1 de olas de calor de 0,59 a 0,76 (h=1–7); sin efecto a 30 días.
- **Integrado (v0.7.0):** `sst_*_h{1,3,7}` v2 `@produccion` con previsión y v1 `@sin_prevision` como respaldo (h30 sigue en v1). Usar siempre `Gemelo.elegir(zona, h, fecha)` y `Gemelo.matriz(fechas, variables)`; nunca `ds` directamente para las variables de un modelo.
- Escenarios: `Perturbacion.periodo` (`futuro` por defecto en la meteorología). Los árboles se saturan fuera de rango: hay que leer siempre los avisos.
- Flujo de promoción: `entrenar_sst_prevision.py` → `registrar_modelos.py --modelos sst_prevision` → `exportar_modelos.py`.

## Fase 7 · bloque 2: deriva y reentrenamiento (ver docs/fase7_deriva.md) · v0.8.0
- `gemelo.monitor`: `metricas.py` (funciones puras) y `deriva.py` (`Monitor`). Umbrales PSI calibrados por modelo (`monitorizar.py --calibrar`); el 0,25 de manual no sirve con anomalías persistentes.
- Intervalo de la API: `monitor.metricas.cuantiles_conformales` (cascada estacional 3 años → 365 d → validación), compartido con el monitor. Config `intervalo`.
- Reentrenamiento: `scripts/reentrenar.py` (campeón vs. retador, ventana de 180 d, mejora ≥ 2 %, alias `@anterior`). Ciclo completo: `scripts/ciclo_mantenimiento.py`.
- **`datos_hasta` por modelo**: usar `Gemelo.datos_hasta(info)` para todo lo que dependa de "qué vio el modelo" (`visto_en_entrenamiento`, residuos del intervalo, rendimiento del monitor, rangos). No usar `fin_entrenamiento_sst` para modelos reentrenados.
- Tras un reentrenamiento, los tests no pueden asumir que 2024–2026 está fuera de muestra: comprobar contra `datos_hasta`.
- Corrección del sesgo de la previsión: se mantiene `fija`. La estacional mejora la previsión pero no la SST (evaluado 2026-11-10). El aviso de sesgo del monitor es informativo. Cualquier cambio de corrección se valida por su efecto en la SST con modelos fuera de muestra, no por el error de la previsión.

## Fase 7 · bloque 3: visor 3D/4D (ver docs/fase7_visor4d.md) · v0.9.0
- `gemelo.procesado.cubo` -> `data/processed/cubo_insitu.parquet` y `cubo_resumen.parquet` (paso 6/7 del pipeline). No importar xarray a nivel de módulo: la API lo importa por `UNIDADES`.
- API: `GET /cubo/fechas`, `GET /cubo` (carga perezosa) y visor estático en `/visor` (`src/gemelo/servicio/visor/`, incluido como `package-data`).
- CesiumJS 1.120.0 por CDN, sin Cesium ion. En Docker el enlace del dashboard usa `GEMELO_API_URL_PUBLICA`.
- Las capturas del visor con WebGL pueden salir en negro o sin mapa si se toman antes de componer o de cargar las teselas: comprobar el estado (`visor.scene.globe.tilesLoaded`) antes de dar algo por roto.

## Mejora de modelos (ver docs/mejora_modelos.md) · v0.10.0
- Protocolo para cualquier mejora futura: desarrollo con `scripts/experimentos_mejora.py` (origen móvil 2014–2021, nada de 2022+) → confirmación única y promoción con `scripts/promocionar_receta.py` (regla: mejora ≥ 2 % e IC90 por bloques > 0).
- Receta `fisica_pesos` (`gemelo.modelos.receta`): variables físicas (`aire_agua_*`, `flujo_*`, `prev{w}_aire_agua_*`, `prev{w}_flujo_*`; `features.es_fisica`), lr 0,02 con 2,5× árboles, pesos recientes (semivida 8 años); a 30 días, media 50/50 con ridge. En producción: h=3/7/30 (laguna v4/v3/v3, costa v4/v4/v3); h=1 sigue en `clasica` (v3).
- `gemelo.modelos.conjunto.ModeloSST` envuelve TODOS los modelos de SST (árboles + lineal opcional): usar `m.predict(X)` y `m.contribuciones(X)` (SHAP aditivo); no usar `shap.TreeExplainer` directamente sobre los modelos servidos.
- Las variables físicas previstas se calculan con `meteo_futura.con_fisica(prev, anom)` (pipeline y escenarios).
- Cada versión guarda `residuos_validacion` (errores fuera de muestra con fecha) → `Gemelo.con_residuos_guardados`; sin ellos la P(ola de calor) degenera. Tras promocionar o reentrenar, comprobar que `metodo_intervalo` empieza por `conformal`.
- Descartado con datos: huber, más regularización, híbrido lineal+residuo, ridge solo (salvo en la media a 30 días).
- **Pendiente:** cobertura del intervalo al 90 % nominal; saturación de árboles a 1–7 días; hipoxia (sin datos nuevos); reentrenar respaldos.

## Mejora visual (ver docs/mejora_visual.md) · v0.11.0
- Dashboard: resumen en una frase (`resumen_estado` + `aviso_estado`, HTML propio; NO usar `st.error` para estados del mar), tarjetas con borde, mapa con `contorno_laguna.json` (servido en `GET /estado`), olas de calor sombreadas, cascada XAI (`graficos.cascada`), `PRESETS` de escenarios, aguja de hipoxia y pestaña «Visor 4D» con `st.iframe`.
- `dashboard/graficos.py` son funciones puras con tests en `tests/test_graficos.py`. Tema en `.streamlit/config.toml` (se copia en `Dockerfile.dashboard`).
- Mapa siempre claro: `st.context.theme` no es fiable. Mapas y visor (WebGL) pueden salir en blanco en capturas: comprobar con `map.loaded()` / `triggerRepaint()` antes de darlos por rotos.
- Al arrancar API y dashboard a la vez, esperar a que la API responda (`/salud`) antes de abrir el dashboard: cachea el error de conexión 10 minutos.
- **Pendiente:** verificar Docker 0.11.0; resto de pendientes del bloque anterior (intervalo al 90 %, saturación a 1–7 días, hipoxia, respaldos).

## Anti-fuga (obligatorio)
- Las variables del día t solo usan datos ≤ t. Climatologías y umbrales de las variables, solo con datos anteriores a validación.
- Al entrenar o evaluar a horizonte h, usar `features.filas_de_particion()` (purga de frontera).
- Cualquier variable nueva queda cubierta por `tests/test_features.py::test_variables_no_miran_al_futuro`: ejecutar `pytest` siempre.
- Con xarray, hacer `.load()` antes de indexar con listas de posiciones sobre NetCDF comprimidos (si no, tarda minutos).
