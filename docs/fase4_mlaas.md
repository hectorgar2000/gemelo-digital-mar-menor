# Fase 4 — MLaaS mínimo

*2026-11-09*

## Flujo de un modelo, de entrenamiento a servicio

```
entrenar_*.py ──► MLflow Tracking (runs, métricas, SHAP, manifiesto de datos)
                     │
registrar_modelos.py ─► MLflow Model Registry: sst_{zona}_h{h}, hipoxia_logistica  @produccion
                     │
exportar_modelos.py ─► models/servicio/  (XGBoost JSON + logística + servicio.json)   ← artefacto portable
                     │
                 API FastAPI (gemelo.servicio.api) / Docker  ◄── data/processed (estado del gemelo)
```

**Por qué un paso de exportación:** el Registry guarda rutas absolutas de Windows, que dentro de un contenedor Linux no existen. El artefacto exportado no depende de MLflow ni de rutas, y es lo único que necesita la API. Para "promocionar" un modelo nuevo basta con registrar, mover el alias `@produccion` y volver a exportar. La API no cambia.

## Endpoints (`uvicorn gemelo.servicio.api:app` → http://127.0.0.1:8000/docs)

| Método | Ruta | Qué hace |
|---|---|---|
| GET | `/salud` | Estado del servicio, último día con datos y última campaña |
| GET | `/modelos` | Modelos en servicio, versión del registro y métricas de test |
| POST | `/predict` | SST prevista a 1/3/7/30 días con **intervalo del 90 %**, umbral y **probabilidad de ola de calor** |
| POST | `/explain` | SHAP de una predicción: contribución por **grupo** y por variable |
| POST | `/hipoxia/predict` | Probabilidad de hipoxia en la próxima campaña, alerta temprana y contribuciones de la logística |

La API recibe una **fecha** y no un vector de 81 variables: el gemelo conoce el estado del mar y monta él mismo las variables. Sin fecha, usa el último día disponible.

**Honestidad en el backtesting:** cada respuesta incluye `sst_real` (si ya se conoce) y `visto_en_entrenamiento`. Los modelos finales se entrenaron hasta 2021, así que una "predicción" de 2019 no es una prueba real.

## Intervalo de incertidumbre: conformal con ventana móvil

En la Fase 3, el intervalo calibrado con validación cubría en test solo entre el 78 % y el 89 %, por la deriva hacia un mar más cálido. La API calcula el intervalo con los **errores de los últimos 365 días cuyo resultado ya se conocía** (causal) y así se adapta a la deriva.

Cobertura en test 2023–2026 (objetivo: 90 %), según `scripts/evaluar_intervalos.py`:

| Zona | h | Cuantiles de validación fijos | **Ventana móvil 365 d** | Ancho medio (°C) fijo → móvil |
|---|---|---|---|---|
| Costa | 1 | 87,6 % | **88,6 %** | 0,96 → 1,03 |
| Costa | 3 | 83,4 % | **88,9 %** | 1,36 → 1,57 |
| Costa | 7 | 81,1 % | **87,9 %** | 1,83 → 2,04 |
| Costa | 30 | 78,7 % | **87,8 %** | 2,81 → 3,75 |
| Laguna | 1 | 88,8 % | **88,9 %** | 1,44 → 1,47 |
| Laguna | 3 | 87,0 % | **88,3 %** | 2,16 → 2,30 |
| Laguna | 7 | 83,5 % | **86,5 %** | 2,55 → 2,83 |
| Laguna | 30 | 82,4 % | **89,2 %** | 3,41 → 4,12 |

El intervalo es algo más ancho, porque refleja la incertidumbre real del mar actual, y la cobertura queda cerca del 90 % en todos los casos. Con los mismos residuos se calcula `prob_superar_umbral`: la probabilidad de ola de calor.

## Docker

```bash
python scripts/exportar_modelos.py          # genera models/servicio
docker compose up --build                   # http://localhost:8000/docs
```

- La imagen solo lleva el código y las dependencias de la API, con versiones fijadas por `requirements.lock.txt`. No incluye MLflow, cartopy ni xarray.
- Los modelos y los datos se **montan** como volúmenes de solo lectura: se actualizan sin reconstruir la imagen.
- Usuario sin privilegios y `HEALTHCHECK` sobre `/salud`.

## Tests
`pytest` → 30 tests:
- Pipeline y anti-fuga.
- Líneas base.
- Integración de la API: reconstrucción de la SST, intervalo, aditividad de SHAP y errores 422 claros.
