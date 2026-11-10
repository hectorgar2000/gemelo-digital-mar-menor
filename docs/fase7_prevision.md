# Fase 7 · Bloque 1 — Previsión meteorológica como entrada

*2026-11-10 · `python scripts/descargar_previsiones.py` y `python scripts/entrenar_sst_prevision.py` · MLflow: experimento `sst_prevision`*

## Por qué
Dos hallazgos anteriores apuntaban a lo mismo:
- **Fase 3:** el peor fallo explicado (15-10-2023, −5,5 °C en una semana) lo causó un temporal **posterior** al día de la predicción.
- **Fase 5:** los escenarios de aire y viento apenas tenían efecto porque el modelo solo veía el **pasado**.

## Datos: previsiones reales archivadas
- Open-Meteo *Previous Runs*: para cada hora, el valor previsto **1…7 días antes**.
- Comprobado: el archivo está completo (temperatura, viento, dirección, lluvia y radiación) **desde febrero de 2024**; antes solo hay temperatura.
- Descargado para las dos celdas ERA5 (cuenca y mar): 23.352 horas por punto, 0 % de nulos.

## Diseño honesto
| Periodo | Uso |
|---|---|
| 1998–2021 | Entrenamiento con la meteorología **que ocurrió** (ERA5, "previsión perfecta"): no hay previsiones archivadas tan antiguas |
| feb–jul 2024 | Calibración del sesgo previsión − ERA5 por variable y antelación (excluido de la evaluación) |
| **ago 2024 – sep 2026** | **Evaluación con las previsiones realmente emitidas** cada día (~790 días) |

- Variables `prev{w}_*` (w = 1, 3, 7 días; el modelo a 30 días usa los 7 primeros): anomalía de temperatura del aire, máxima, lluvia, viento medio y máximo, radiación y componentes u/v, para cuenca y mar.
- Nunca usan la temperatura del agua futura.
- Tests (`tests/test_meteo_futura.py`):
  - La ventana es t+1…t+w.
  - El día t+k sale de la previsión emitida k días antes.
  - Nada más allá de t+7.
  - Unidades y corrección de sesgo.
- La mejora sobre el modelo actual lleva un **IC del 90 % por bootstrap de bloques de 30 días**, porque los errores diarios están autocorrelacionados.

Sesgo medido de la previsión frente a ERA5 (antelación 1 / 7 días):
- Cuenca: temperatura −1,0 / −0,6 °C; viento −1,3 / +1,8 km/h.
- Mar: diferencias pequeñas.

Se corrige antes de usar la previsión.

## Resultados (ventana honesta ago-2024 → sep-2026, RMSE del cambio de anomalía en °C)

| Zona | h | Persistencia | A (actual) | **B con previsión real** | B con previsión perfecta | Mejora B real vs A [IC 90 %] | F1 ola de calor A → B |
|---|---|---|---|---|---|---|---|
| Laguna | 1 | 0,463 | 0,442 | **0,426** | 0,425 | +3,7 % [+2,1, +5,1] | 0,87 → 0,87 |
| Laguna | 3 | 0,791 | 0,735 | **0,684** | 0,673 | +6,9 % [+4,4, +8,9] | 0,78 → 0,81 |
| Laguna | 7 | 0,946 | 0,911 | **0,751** | 0,738 | **+17,5 %** [+14,0, +20,3] | **0,59 → 0,76** |
| Laguna | 30 | 1,281 | 1,188 | 1,189 | 1,187 | −0,1 % [−1,3, +0,9] | 0,38 → 0,35 |
| Costa | 1 | 0,338 | 0,324 | **0,304** | 0,302 | +6,2 % [+4,3, +8,4] | 0,86 → 0,87 |
| Costa | 3 | 0,525 | 0,531 | **0,432** | 0,416 | **+18,7 %** [+14,3, +22,6] | 0,74 → 0,80 |
| Costa | 7 | 0,669 | 0,707 | **0,523** | 0,487 | **+26,0 %** [+20,0, +30,1] | **0,57 → 0,75** |
| Costa | 30 | 1,024 | 1,045 | 1,042 | 1,034 | +0,3 % [−0,9, +1,4] | 0,36 → 0,35 |

## Lectura
1. **La previsión meteorológica es la mejora más grande del proyecto** a 3–7 días: entre un 17 % y un 26 % menos de error a 7 días. Todos los intervalos de confianza de 1–7 días están por encima de cero.
2. **La costa deja de empatar con la persistencia.** En la Fase 3, a 3–7 días el modelo no la superaba; ahora la supera con claridad (0,52 frente a 0,67 °C a 7 días).
3. **Olas de calor a 7 días:** el F1 sube de 0,59 a 0,76 en la laguna y de 0,57 a 0,75 en la costa.
4. **La previsión real rinde casi como la perfecta:** B real se queda a 0,01–0,04 °C de la cota superior. A 1–7 días, las previsiones de hoy son lo bastante buenas como para que su error apenas importe.
5. **A 30 días no aporta nada** (todos los IC contienen el 0). Conocer la primera semana no basta para predecir el mes: el modelo a 30 días se queda como está.
6. **SHAP:** la meteorología prevista pasa a explicar el 24–38 % de la predicción de 1 a 7 días, frente al ~10 % a 30 días.

## Limitaciones
- Se evalúa en ~2 años (790 días), con IC por bootstrap de bloques. Es suficiente para estas mejoras, que son grandes, pero no para afinar diferencias pequeñas.
- El modelo aprendió con previsión **perfecta** y se usa con previsión **real**. Funciona porque a 1–7 días el error de previsión es pequeño, pero los intervalos del 90 % deben recalibrarse con errores de la previsión real (el conformal móvil de la API ya lo hace).
- El modelo de previsión de Open-Meteo puede cambiar con el tiempo: hay que vigilar el sesgo, que es una tarea natural del bloque de monitorización de la deriva.
