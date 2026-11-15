# Estudio de propuestas de mejora (v0.11.0 · 2026-11-15)

Estudio hecho al cierre del proyecto, con los KPIs fuera de muestra de `scripts/kpis_informe.py`
(`docs/figuras/informe/kpis_sst.csv`). Las propuestas están ordenadas por relación impacto/esfuerzo.
**Ninguna está implementada**: son candidatas para una fase 8.

## 1. Punto de partida: qué funciona y qué no

Modelos entrenados con su receta de producción solo hasta 2021 y evaluados fuera de muestra (h ≤ 7: ago-2024 →
sep-2026 con previsiones reales; h = 30: 2022 → 2026).

| Modelo | RMSE (°C) | vs. persistencia | Acierto ≤ 1 °C | Sesgo (°C) | F1 ola de calor | F1 persistencia |
|---|---|---|---|---|---|---|
| Laguna · 1 día | 0,43 | −8 % | 96 % | −0,01 | 0,87 | 0,87 |
| Laguna · 3 días | 0,64 | −20 % | 89 % | 0,00 | 0,83 | 0,80 |
| Laguna · 7 días | 0,72 | −24 % | 85 % | −0,02 | 0,76 | 0,72 |
| Laguna · 30 días | 1,24 | −14 % | 62 % | **−0,43** | **0,39** | 0,54 |
| Costa · 1 día | 0,30 | −11 % | 99 % | −0,01 | 0,87 | 0,85 |
| Costa · 3 días | 0,41 | −23 % | 98 % | −0,04 | 0,80 | 0,77 |
| Costa · 7 días | 0,51 | −24 % | 94 % | −0,07 | 0,75 | 0,68 |
| Costa · 30 días | 1,16 | −11 % | 68 % | **−0,46** | **0,28** | 0,52 |

- **Fortaleza:** de 1 a 7 días el modelo bate a la persistencia en todo (error y olas de calor) y no tiene sesgo.
- **Hallazgo nuevo:** a 30 días el modelo gana en RMSE pero **pierde en olas de calor**. Tiene un sesgo frío de casi
  medio grado (aprendió con un mar más fresco y devuelve la anomalía hacia lo normal), así que solo detecta el
  28 % (laguna) y el 18 % (costa) de los días de ola.
- **Hipoxia:** sin datos in situ desde julio de 2024; 52 % de falsas alarmas.
- **Intervalo:** cobertura del 87 % frente al 90 % nominal.

## 2. Propuestas

### A. Modelos

**A1 · Corregir el sesgo a 30 días** — impacto alto, esfuerzo bajo. *Prioridad 1.*
Restar a la predicción la media de los errores ya conocidos (los de orígenes ≤ t − 30) en una ventana móvil.
Comprobación exploratoria con una ventana de 365 días (causal, sin mirar al futuro):

| Zona | RMSE | Sesgo | Recall de ola | F1 de ola |
|---|---|---|---|---|
| Laguna | 1,240 → 1,203 | −0,43 → −0,08 | 0,28 → 0,48 | 0,39 → 0,53 |
| Costa | 1,155 → 1,103 | −0,46 → −0,07 | 0,18 → 0,47 | 0,28 → 0,49 |

Con ventanas de 90 y 180 días el F1 sube parecido (0,50–0,63) pero el RMSE empeora: la de 365 es la única que mejora
las dos cosas. **Aviso:** la ventana se ha mirado sobre el periodo de prueba, así que estas cifras son una estimación
optimista. Hay que elegirla con origen móvil 2014–2021 (`experimentos_mejora.py`) y confirmarla una sola vez. Además,
los modelos en producción están reajustados hasta 2026 y su sesgo actual será menor que el medido aquí.

**A2 · Intervalo al 90 % real (conformal adaptativo)** — impacto medio, esfuerzo bajo. *Prioridad 2.*
Ajustar en línea el nivel del intervalo según la cobertura reciente (si cubre menos del 90 %, se ensancha). Es el
remedio estándar cuando hay deriva. Criterio de éxito: cobertura 89–91 % sin empeorar la puntuación de intervalo (3,21).

**A3 · Ola de calor como racha, no como día suelto** — impacto medio, esfuerzo medio.
Hoy se compara la SST de un día con el umbral, pero la definición exige ≥ 5 días seguidos. Usar juntas las predicciones
de 1, 3 y 7 días (y la racha ya acumulada, `racha_p90_*`) para estimar la probabilidad de racha. Debería reducir las
falsas alarmas de 1–7 días (62–109 por modelo).

**A4 · Extrapolación en escenarios a 1–7 días** — impacto medio (en escenarios), esfuerzo medio.
Los árboles se saturan fuera de rango. Dos opciones a comparar: (a) restricciones monótonas de XGBoost en las variables
de temperatura del aire prevista; (b) extender la media con el modelo lineal a 1–7 días solo si no empeora el RMSE
(a 30 días ya funciona). Evaluar con el protocolo: no vale mejorar los escenarios a costa de la predicción.

**A5 · Reentrenar los modelos de respaldo** — impacto bajo, esfuerzo bajo.
Los `@sin_prevision` siguen en v1 (receta clásica, datos hasta 2021). Reajustarlos con `fisica_pesos` y datos hasta 2026.

**A6 · Previsión estacional o por conjuntos para 14–30 días** — impacto potencialmente alto, esfuerzo alto, incierto.
A 30 días no hay previsión meteorológica. Verificar si hay previsiones estacionales o por conjuntos archivadas con
historial suficiente (habría que comprobarlo antes, igual que se hizo con Previous Runs) y añadir un horizonte de 14 días.

### B. Datos

**B1 · Funcionamiento casi en tiempo real** — impacto alto, esfuerzo medio. *Prioridad 3.*
El gemelo se detiene el 30-09-2026 porque usa reanálisis. Verificar con `copernicusmarine describe` los productos NRT
de SST y clorofila, medir su diferencia con el reanálisis en el periodo común y, si es pequeña, alimentar con ellos el
último tramo. Con la previsión en vivo de Open-Meteo, el gemelo daría una predicción nueva cada día.

**B2 · Datos de oxígeno recientes** — impacto alto, esfuerzo incierto (depende de terceros).
Sin campañas nuevas el modelo de hipoxia no se puede vigilar, reentrenar ni usar hoy. Preguntar a la UPCT por la
continuidad del servidor y buscar series de boyas o sondas fijas. Sin esto, cualquier mejora del modelo de hipoxia es
teórica: hay 9 episodios para aprender.

**B3 · Calibrar la SST de satélite de la laguna con la in situ** — impacto medio, esfuerzo medio.
El sesgo estacional (+1,9 °C en invierno, −2,3 °C en mayo) está medido. Una corrección por día del año, ajustada solo
con campañas anteriores a la validación, permitiría dar la temperatura «de termómetro» además de la de satélite.

**B4 · Aportes de la cuenca (caudales, nutrientes)** — impacto alto para hipoxia, esfuerzo alto.
Nitratos, fosfatos y vertidos figuran hoy como «no disponibles» en los escenarios. Son las causas reales de la
eutrofización; habría que localizar una fuente pública con serie larga antes de plantearlo.

### C. Servicio y operación

**C1 · Integración continua** — impacto medio, esfuerzo bajo. *Prioridad 4.*
GitHub Actions con `ruff` y `pytest` en cada push. Las pruebas que necesitan datos ya se saltan solas, así que funciona
sin subir datos. Añadir un conjunto mínimo de datos de prueba permitiría ejecutar también las de la API.

**C2 · Seguridad de la API** — impacto alto si se despliega, esfuerzo bajo.
Clave de API en cabecera, límite de peticiones y CORS restringido. `/scenario` con 200 muestras es el endpoint caro.

**C3 · Registro de las predicciones servidas** — impacto medio, esfuerzo bajo.
Guardar cada predicción con su fecha y versión de modelo. El monitor mediría el rendimiento real en servicio en vez de
reconstruirlo, y quedaría un historial auditable.

**C4 · Despliegue en la nube** — impacto medio, esfuerzo medio.
Render o Fly.io con las dos imágenes. Requiere C2, datos y modelos dentro de la imagen y ~1 GB de RAM. Antes, verificar
las imágenes Docker 0.11.0 (pendiente).

**C5 · Programar el mantenimiento** — impacto bajo, esfuerzo bajo.
Tarea semanal con `ciclo_mantenimiento.py` y aviso cuando el estado global no sea OK. Solo tiene sentido con B1.

### D. Producto y explicabilidad

**D1 · Pestaña de verificación** — impacto medio, esfuerzo bajo.
Mostrar en el dashboard lo que se predijo frente a lo que ocurrió y los KPIs de este estudio (acierto ≤ 1 °C, aciertos
y falsas alarmas de ola). Es lo que más confianza da a un usuario no técnico.

**D2 · Explicación en una frase** — impacto medio, esfuerzo bajo.
Traducir los tres grupos SHAP principales a texto («se espera que baje 0,8 °C sobre todo porque se prevé viento fuerte»),
con plantillas, igual que el resumen de estado.

**D3 · Explicar los escenarios** — impacto medio, esfuerzo bajo.
Mostrar la diferencia de contribuciones SHAP entre el caso base y el escenario: qué variables mueven el resultado.

**D4 · Capas de satélite y predicción en el visor 4D** — impacto bajo-medio, esfuerzo medio.
Hoy el visor solo muestra los cubos in situ (hasta 2024). Añadir la SST de satélite y la predicha como capa de superficie.

## 3. Hoja de ruta sugerida

| Tanda | Propuestas | Resultado esperado |
|---|---|---|
| 1 · Rápidas (1–2 sesiones) | A1, A2, A5, C1, D1 | Olas de calor a 30 días utilizables, intervalo calibrado, CI en verde |
| 2 · Operación (2–4 sesiones) | B1, C2, C3, C5, C4 | Gemelo que se actualiza a diario y se puede publicar |
| 3 · Fondo (abierto) | A3, A4, A6, B2, B3, B4, D2–D4 | Mejor ciencia: rachas, extrapolación, hipoxia con datos nuevos |

Todas las propuestas de modelos (A) deben pasar por el protocolo de `docs/mejora_modelos.md`: desarrollo con origen móvil
2014–2021, una sola confirmación fuera de muestra y promoción solo con mejora ≥ 2 % e IC90 > 0.
