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
- O₂ in situ: UPCT THREDDS `L2/Oxigeno.nc`, semanal, mar 2017 → jul 2024, 10 profundidades de 0,5 a 5 m, mg/L. Hay un registro con fecha errónea (1970) que hay que filtrar.
