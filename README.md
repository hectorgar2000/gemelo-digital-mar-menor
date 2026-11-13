# Gemelo digital marino — Mar Menor

Réplica virtual del **Mar Menor y su costa** alimentada con datos reales (Copernicus Marine, ERA5 y previsiones de Open-Meteo, y las campañas in situ de la UPCT). Sus modelos de IA:

1. **Describen** el estado del agua: temperatura, ola de calor en curso y oxígeno.
2. **Predicen** la temperatura superficial y las **olas de calor marinas** a 1, 3, 7 y 30 días, con intervalo de incertidumbre, y el **riesgo de hipoxia**.
3. **Simulan escenarios** "qué pasaría si" con incertidumbre (Monte Carlo).
4. **Explican** cada predicción con SHAP.

Todo se sirve como **MLaaS**:
- Modelos versionados en MLflow.
- API FastAPI en Docker.
- Dashboard Streamlit.
- Visor 4D con CesiumJS.
- Ciclo de mantenimiento que vigila la deriva y reentrena.

📄 **[Informe final](docs/informe_final.md)**: resultados, decisiones, limitaciones y trabajo futuro.

## Qué hay
| Pieza | Dónde | Qué hace |
|---|---|---|
| API | `http://localhost:8000/docs` | `/predict`, `/explain`, `/scenario`, `/hipoxia/predict`, `/estado`, `/series`, `/monitor`, `/cubo`… |
| Dashboard | `http://localhost:8501` | Estado y predicción, escenarios, XAI, hipoxia, visor 4D integrado y salud de los modelos |
| Visor 4D | `http://localhost:8000/visor/` | Cubos in situ (O₂, T, S…) en 3D a lo largo de 380 campañas |
| Experimentos | `mlflow ui --backend-store-uri sqlite:///mlflow.db` | Ejecuciones, métricas, SHAP y Model Registry |

## Arranque rápido (si ya tienes `data/processed` y `models/servicio`)
```bash
docker compose up --build
```
O sin Docker, en dos terminales:
```bash
uvicorn gemelo.servicio.api:app
streamlit run src/gemelo/dashboard/app.py
```

## Reproducir desde cero
Probado en Windows 11 con Python 3.13. Las imágenes Docker usan Python 3.13 sobre Linux. `requirements.lock.txt` fija las versiones exactas.

`data/`, `models/`, `mlflow.db` y `mlartifacts/` **no están en git**: se regeneran con estos pasos, en este orden.

**1. Entorno**
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -c requirements.lock.txt -e ".[datos,ml,api,dashboard,dev]"
```
Para abrir los notebooks de exploración, instala además el grupo `notebooks` y registra el kernel:
```bash
pip install -c requirements.lock.txt -e ".[notebooks]"
python -m ipykernel install --sys-prefix --name gemelo --display-name "Python (gemelo)"
```
> **Windows y rutas largas:** JupyterLab trae ficheros con rutas muy profundas. Si el proyecto está en una carpeta con ruta larga, la instalación de `notebooks` falla por el límite de 260 caracteres. Clónalo en una ruta corta (p. ej. `C:\gemelo`) o [activa las rutas largas de Windows](https://pip.pypa.io/warnings/enable-long-paths). El resto de grupos no se ve afectado.

**2. Credenciales de Copernicus Marine** (cuenta gratuita en marine.copernicus.eu). Se guardan en `%USERPROFILE%\.copernicusmarine`, nunca en el repositorio.
```bash
copernicusmarine login
```

**3. Descargas** (≈ 20–30 min en total; la UPCT es la más lenta). Van a `data/raw/`.
```bash
python scripts/descargar_copernicus.py
python scripts/descargar_upct.py
python scripts/descargar_meteo.py
python scripts/descargar_previsiones.py
```

**4. Procesado** (≈ 3 min): series por subzona, variables, objetivos, meteorología futura, cubo 4D y manifiesto. Va a `data/processed/`.
```bash
python scripts/construir_dataset.py
```

**5. Entrenamiento** (≈ 25 min). Queda registrado en MLflow.
```bash
python scripts/entrenar_sst.py
python scripts/entrenar_hipoxia.py
python scripts/entrenar_sst_prevision.py
```

**6. Registro, exportación y monitor**
```bash
python scripts/registrar_modelos.py                          # @produccion (SST e hipoxia)
python scripts/registrar_modelos.py --modelos sst_prevision  # v2 con previsión; v1 -> @sin_prevision
python scripts/exportar_modelos.py                           # -> models/servicio (lo que carga la API)
python scripts/promocionar_receta.py --promover              # receta mejorada (física + pesos) donde gana >= 2 %
python scripts/monitorizar.py --calibrar                     # umbrales del monitor + data/processed/monitor.json
```

**7. Comprobar y arrancar**
```bash
pytest
docker compose up --build
```

**Mantenimiento** (se puede programar; ver el propio script):
```bash
python scripts/ciclo_mantenimiento.py
```

> **Nota sobre resultados:** los números de versión dependen del orden de ejecución (el ciclo de mantenimiento y la promoción de la receta crean versiones nuevas). Si se descarga en otra fecha, los reanálisis de Copernicus pueden traer más días y las cifras pueden variar ligeramente respecto a los informes.

**Tests:** 92 en total, incluidos los anti-fuga, la API, los escenarios, el monitor y el dashboard. En un clon sin `data/` ni `models/` pasan los que no dependen de ellos y el resto se **salta** con un aviso que indica qué ejecutar. Con todo generado, pasan los 92.

## Estructura
```
config/zona.yaml     zona, periodos, umbrales, dataset_id, monitor, intervalo… (fuente única de verdad)
src/gemelo/          paquete Python
  ingesta/           descargas (Copernicus, UPCT, ERA5, previsiones)
  procesado/         series, variables, meteorología futura, in situ, cubo 4D
  modelos/           SST, hipoxia, XAI, reentrenamiento
  monitor/           deriva (PSI estacional, rendimiento, cobertura) e intervalos conformales
  servicio/          motor del gemelo, escenarios, API FastAPI y visor 4D (visor/)
  dashboard/         app Streamlit (solo consume la API)
scripts/             puntos de entrada (descargar, construir, entrenar, registrar, exportar, monitorizar…)
tests/               pytest
notebooks/           exploración (Fase 1)
docs/                informe final, alcance y documento de cada fase
Dockerfile, Dockerfile.dashboard, compose.yaml
```

## Roadmap
| Fase | Entregable | Estado |
|---|---|---|
| 0 · Planteamiento | Repo y [documento de alcance](docs/alcance.md) | ✅ |
| 1 · Ingesta y exploración | Datos y notebook de exploración ([resumen](docs/fase1_resumen.md)) | ✅ |
| 2 · Procesado | Pipeline `raw → processed` ([resumen y revisión](docs/fase2_resumen.md)) | ✅ |
| 3 · Primer modelo + XAI | XGBoost que bate a la línea base, con SHAP ([informe](docs/fase3_informe.md)) | ✅ |
| 4 · MLaaS mínimo | Registro, API y Docker ([detalle](docs/fase4_mlaas.md)) | ✅ |
| 5 · Escenarios | `/scenario` con incertidumbre ([detalle](docs/fase5_escenarios.md)) | ✅ |
| 6 · Dashboard | Streamlit de punta a punta ([detalle](docs/fase6_dashboard.md)) | ✅ |
| Mejora de modelos | Variables físicas, pesos recientes y modelo combinado: −4 % de error, −10 % en extremos ([detalle](docs/mejora_modelos.md)) | ✅ |
| Mejora visual | Resumen en una frase, contorno real de la laguna, cascada XAI, escenarios de un clic y visor integrado ([detalle](docs/mejora_visual.md)) | ✅ |
| 7 · Evolución | Previsión meteorológica ([detalle](docs/fase7_prevision.md)) · deriva y reentrenamiento ([detalle](docs/fase7_deriva.md)) · visor 3D/4D ([detalle](docs/fase7_visor4d.md)) | ✅ |

## Fuentes de datos
- [Copernicus Marine Service](https://marine.copernicus.eu): SST y clorofila por satélite, y física del Mediterráneo.
- [Open-Meteo](https://open-meteo.com): ERA5 y previsiones archivadas (uso no comercial).
- [Servidor de Datos Científicos del Mar Menor (UPCT)](https://marmenor.upct.es): oxígeno, temperatura, salinidad, clorofila y turbidez in situ.
