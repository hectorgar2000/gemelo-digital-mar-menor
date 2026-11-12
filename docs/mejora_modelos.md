# Mejora de los modelos de SST

*2026-11-12 · v0.10.0 · `scripts/experimentos_mejora.py` y `scripts/promocionar_receta.py` · MLflow: experimento `mejora_receta`*

## Cómo se ha buscado la mejora
El periodo de test (2022–2026) ya se había mirado muchas veces, así que cualquier ajuste elegido con él sería sospechoso. El protocolo fue:

1. **Desarrollo sin tocar el test:** origen móvil con 8 años fuera de muestra (2014–2021). Cada variante se compara con la configuración en producción, con un IC del 90 % por bootstrap de bloques de 30 días.
2. **Confirmación, una sola vez:** la receta elegida y la actual se entrenan hasta 2021 y se evalúan con las **previsiones realmente emitidas** de agosto de 2024 a septiembre de 2026 (30 días: 2022–2026).
3. **Promoción con la regla de siempre:** mejora ≥ 2 % y, además, IC90 > 0.

## Qué se probó (desarrollo, media de los 8 modelos)
| Variante | Mejora del error | En días extremos | Veredicto |
|---|---|---|---|
| Aprendizaje lento (lr 0,02; 2,5× árboles) | +0,2 % | +0,8 % | Ayuda poco, pero nunca empeora |
| Más peso a años recientes (semivida 8 años) | +0,4 % | +0,6 % | Ayuda en la laguna a 3–7 días |
| Más regularización | −0,3 % | +0,1 % | Descartada (empeora a 30 días) |
| Pérdida robusta (Huber) | −0,2 % | +0,7 % | Descartada |
| Solo lineal (ridge) | −2,4 % | +2,3 % | Descartada, salvo a 30 días (+2,7 % en la laguna) |
| Híbrido: lineal + árboles sobre el residuo | −0,7 % | +2,6 % | Descartado (empeora la costa) |
| **Variables físicas** (aire − agua, × viento) | +0,6 % | **+3,6 %** | Adoptada |
| **Lento + pesos + físicas** | **+1,5 %** | +2,7 % | **Adoptada para 1, 3 y 7 días** |
| **Media de árboles y lineal** | +1,0 % (30 d: **+2,5 %**) | +3,8 % | **Adoptada para 30 días** |

Los ajustes del algoritmo por sí solos dan poco: los modelos ya estaban cerca de su techo. Lo que más aporta es **información física explícita**. El intercambio de calor depende de la diferencia entre la temperatura del aire y la del agua, y del viento. Antes el modelo tenía las dos anomalías por separado y debía deducir la diferencia.

## La receta nueva (`gemelo.modelos.receta`, receta `fisica_pesos`)
- **Variables nuevas** (grupo SHAP `intercambio_aire_mar`, y `meteo_prevista` para las de previsión):
  - `aire_agua_{punto}_{zona}`: temperatura del aire menos la del agua, hoy y en media de 7 días.
  - `flujo_{punto}_{zona}`: esa diferencia por el viento.
  - `prev{w}_aire_agua_*` y `prev{w}_flujo_*`: lo mismo con el aire y el viento **previstos** frente al agua de hoy.
- **Entrenamiento:** aprendizaje más lento con 2,5 veces más árboles, y más peso a los años recientes (la mitad cada 8 años hacia atrás).
- **A 30 días:** media al 50 % con un modelo lineal (`ModeloSST`, en `gemelo.modelos.conjunto`). La parte lineal se guarda como JSON legible, no como pickle.

## Confirmación con previsiones reales
Modelos entrenados hasta 2021; RMSE del cambio de anomalía (°C).

| Modelo | Receta actual | Receta nueva | Mejora [IC 90 %] | Días extremos | Decisión |
|---|---|---|---|---|---|
| Laguna 1 d | 0,426 | 0,423 | +0,7 % [−0,8, +2,0] | +3,2 % | se mantiene |
| Laguna 3 d | 0,677 | **0,636** | **+6,0 %** [+3,2, +8,3] | +10,3 % | → v4 |
| Laguna 7 d | 0,754 | **0,720** | **+4,5 %** [+0,4, +8,2] | +17,3 % | → v3 |
| Laguna 30 d | 1,306 | **1,240** | **+5,1 %** [+3,0, +6,5] | +8,4 % | → v3 |
| Costa 1 d | 0,302 | 0,297 | +1,6 % [−0,1, +3,0] | +0,2 % | se mantiene |
| Costa 3 d | 0,430 | **0,406** | **+5,5 %** [+3,6, +7,4] | +13,0 % | → v4 |
| Costa 7 d | 0,528 | **0,507** | **+3,9 %** [+0,3, +7,7] | +10,0 % | → v4 |
| Costa 30 d | 1,214 | **1,155** | **+4,8 %** [+2,8, +6,8] | +19,0 % | → v3 |

- Media: **−4,0 % de error y −10,2 % en los días extremos** (el 10 % con mayor anomalía). Ningún modelo empeora.
- La confirmación sale mejor que el desarrollo (+4,0 % frente a +1,5 %): los años 2024–2026 son más extremos, que es donde la receta más ayuda.
- Los seis modelos promocionados se reajustaron con todos los datos hasta septiembre de 2026. Los anteriores quedan como `@anterior`.

## Otras mejoras de esta tanda
1. **La probabilidad de ola de calor había quedado degradada, y está corregida.**
   - Desde el reentrenamiento (v0.8.0), los modelos reajustados con todos los datos no tenían errores fuera de muestra para el intervalo. Caían al respaldo de cuantiles fijos, y la probabilidad de ola de calor salía siempre 0, 0,5 o 1.
   - Ahora cada modelo guarda **sus errores fuera de muestra con fecha** (`residuos_validacion`): los de su evaluación, con su misma receta. El intervalo estacional los usa hasta que lleguen datos nuevos; cuando lleguen, mandan los errores en vivo.
   - Si no hubiera ni unos ni otros, el último recurso interpola los cuantiles guardados en vez de usar dos números.
2. **Reentrenamiento más exigente:** además de mejorar ≥ 2 %, la mejora debe distinguirse del ruido (IC90 por bloques > 0). El retador usa la receta de su campeón.
3. **Los escenarios responden más al aire:** con las variables físicas, "aire +3 °C la próxima semana" mueve la previsión a 7 días **+0,8 °C** en un día normal, frente a +0,09 °C antes.

## Limitaciones
- **Los árboles siguen saturándose** a 1–7 días: aire +3 °C y +6 °C dan el mismo resultado en un día normal. El híbrido que lo evitaba empeoraba la costa y no se adoptó. A 30 días la parte lineal sí extrapola.
- Los modelos en producción son reajustes con todos los datos; la evidencia viene de sus equivalentes entrenados hasta 2021.
- Los modelos de 1 día y los respaldos sin previsión siguen con la receta clásica.
- **No abordado:** llevar la cobertura del intervalo al 90 % nominal, y la hipoxia (sin datos nuevos no hay con qué mejorarla ni validarla).
- Docker 0.10.0 no se ha verificado en esta sesión (Docker Desktop estaba parado). Sí pasan los 86 tests, que cubren la API y el dashboard.
