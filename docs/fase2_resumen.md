# Fase 2 — Dataset de entrenamiento y revisión

*2026-11-09*

## Pipeline reproducible
```bash
python scripts/descargar_copernicus.py     # data/raw/copernicus
python scripts/descargar_upct.py           # data/raw/upct
python scripts/descargar_meteo.py          # data/raw/meteo (ERA5 vía Open-Meteo)
python scripts/construir_dataset.py        # data/interim -> data/processed (~15 s)
pytest                                     # 14 tests, incluidos los anti-fuga
```

| Salida | Contenido |
|---|---|
| `data/interim/diario.parquet` | 16.344 días × 23 series por subzona (SST, log-clorofila, física de costa, ERA5 cuenca/mar) |
| `data/interim/oxigeno_campanas.parquet` | 380 campañas UPCT: O₂ en superficie y fondo, % del área en hipoxia/anoxia |
| `data/processed/dataset_sst.parquet` | 10.500 días (1998–2026) × 81 variables + objetivos `y_anom_*`, `y_sst_*`, `y_mhw_*` para h = 1, 3, 7, 30 |
| `data/processed/dataset_hipoxia.parquet` | 359 pares de campañas (3–14 días) × 46 variables; objetivo `y_hipoxia` (> 10 % del fondo < 4 mg/L) |

**Decisiones**
- **Rejilla común = media por subzona.** Cada fuente tiene su resolución (de 1 km a 0,25°), y todas se reducen a laguna/costa con paso diario.
- **Anomalías** respecto a la climatología calculada **solo con datos anteriores a validación**. Así se quita la estacionalidad y no entra información de validación ni de test.
- **Dos umbrales p90:** el estándar de Hobday (1991–2020) define el **objetivo** de ola de calor. Un p90 calculado sin datos de validación alimenta la **variable** `racha_p90`. El test anti-fuga detectó que usar el estándar también para la variable filtraba 2019–2020.
- **Relleno de huecos causal** (`ffill`, como mucho 3 días): interpolar usaría el dato de mañana.
- Viento de ERA5 en **km/h**, descompuesto en componentes u/v. Lluvia acumulada a 1–90 días y días desde la última lluvia fuerte (≥ 30 mm).

## Revisión: lo que dicen los números

| # | Hallazgo | Evidencia |
|---|---|---|
| 1 | **La SST de satélite en la laguna está sesgada estacionalmente** | Frente a la temperatura in situ de la UPCT (380 campañas): RMSE 1,7 °C, r = 0,97. Sale **+1,9 °C en invierno y −2,3 °C en mayo**, y subestima los máximos (in situ 29–31 °C, satélite 27–30 °C). |
| 2 | **Deriva fuerte entre particiones** | Anomalía media de SST: train +0,2 °C, validación +0,6 °C, test +1,2 °C. Días de ola de calor: **5 % en train, 15 % en validación (laguna), 38 % en test**. |
| 3 | **La persistencia amortiguada es la línea base que hay que batir** | RMSE de la anomalía en validación (laguna): h=1 0,44 · h=3 0,68 · h=7 0,84 · h=30 1,10 (climatología: 1,27). |
| 4 | **Fuga en la frontera** de particiones | Los últimos *h* días de train tienen el objetivo en validación (1–30 filas). Ya existe `filas_de_particion()` para purgarlos. |
| 5 | **Variables muy redundantes** | 28 pares con \|r\| > 0,95 (SST laguna, SST costa, temperatura del modelo, lags y medias…). SHAP repartirá la importancia entre ellas. |
| 6 | **La hipoxia tiene muy pocos episodios** | 30 positivos, pero solo **9 episodios** independientes (11 positivos en 2021). La persistencia ya logra recall 0,73 y precisión 0,76. |

## Mejoras ya aplicadas
- Fuga del umbral p90 en `racha_p90` (detectada por un test).
- Relleno de huecos causal.
- Purga de frontera disponible (`filas_de_particion`) y testeada.
- Eliminada `estratif_costa` (redundante, r = 0,99 con la SST, y con un nombre engañoso).
- Nuevas variables de lluvia a 60/90 días y días desde la última lluvia fuerte.
- Paso del oxígeno: de 6 min a 8 s (cargar en memoria antes de indexar).

## Mejoras propuestas (por prioridad)

**Para la Fase 3 (entrenamiento)**
1. **Predecir el cambio y no el nivel:** objetivo = `anom(t+h) − anom(t)`. Los árboles (XGBoost) no extrapolan: con el test 1 °C más cálido, predecir el nivel tiende a "recortar" las anomalías altas. Al predecir el cambio, el modelo parte de la persistencia y solo aprende la corrección.
2. **Olas de calor derivadas de la regresión:** predecir la SST y aplicarle el umbral, en lugar de entrenar un clasificador aparte con un 5 % de positivos que en test pasan a ser el 38 %.
3. **Validación cruzada con origen móvil** (expanding window, varios cortes anuales) para elegir hiperparámetros, en lugar de un único bloque 2019–2021 que es atípico (+15 % de ola de calor en la laguna y solo 3 % en la costa).
4. **Comparar siempre con la persistencia amortiguada** y reportar el *skill* por horizonte. A 30 días el objetivo realista es acercarse a la climatología ajustada por tendencia.
5. **SHAP por grupos** (temperatura propia, costa, meteorología, clorofila, calendario) o poda de variables redundantes, para que la explicación no se diluya.

**Para el módulo de hipoxia**
6. **Usar las variables in situ de la UPCT** (temperatura, salinidad, clorofila y turbidez, ya disponibles en el mismo THREDDS y en las mismas campañas) en vez de la SST de satélite, que está sesgada en la laguna (hallazgo 1).
7. **Evaluar por episodio** y dejar fuera un año completo en la validación cruzada. Hay que batir la persistencia (recall 0,73) y reportar el Brier score.

**Para el gemelo operativo (Fases 4–5)**
8. Los productos actuales son **reanálisis con retraso** (la SST llega a oct. 2026 y la física a sep. 2026). Para el "estado actual" harán falta los productos NRT equivalentes, cuyos `dataset_id` hay que verificar.
9. Experimento opcional: usar la **meteorología futura de ERA5** como "previsión perfecta", que da una cota superior de lo que aportaría una previsión meteorológica real. Debe ir siempre etiquetado como tal.
