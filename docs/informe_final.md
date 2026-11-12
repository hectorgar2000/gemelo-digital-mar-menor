# Gemelo digital del Mar Menor — Informe final

*2026-11-12 · v0.9.0 · Detalle de cada fase en los documentos enlazados.*

## 1. Qué es
Una réplica virtual del **Mar Menor y su costa** alimentada con datos reales. Sus modelos de IA:

1. **Describen** el estado del agua: temperatura, ola de calor en curso y oxígeno.
2. **Predicen** la temperatura superficial y las olas de calor marinas a 1, 3, 7 y 30 días, con intervalo de incertidumbre, y el riesgo de hipoxia en la siguiente campaña de muestreo.
3. **Simulan escenarios** "qué pasaría si" con incertidumbre (Monte Carlo).
4. **Explican** cada predicción con SHAP.

Todo se sirve como **MLaaS**: modelos versionados en MLflow, API FastAPI en Docker, dashboard Streamlit, visor 4D CesiumJS y un ciclo de mantenimiento que vigila la deriva y reentrena.

```
FUENTES        Copernicus Marine (SST y clorofila por satélite, física de la costa) · ERA5 y previsiones (Open-Meteo)
               · UPCT (O₂, T, S, clorofila y turbidez in situ, 3D)
INGESTA        scripts/descargar_*.py  ──►  data/raw
PROCESADO      scripts/construir_dataset.py ──► data/processed (series, variables, objetivos, previsión, cubo 4D, manifiesto)
MODELOS        entrenar_*.py ──► MLflow (experimentos) ──► registrar (@produccion / @anterior / @sin_prevision) ──► exportar
SERVICIO       API: /predict /explain /scenario /hipoxia/predict /estado /series /monitor /cubo  +  /visor (4D)
CONSUMO        Dashboard Streamlit (5 pestañas) · visor CesiumJS · terceros por API
MANTENIMIENTO  ciclo_mantenimiento.py: monitor de deriva ──► reentrenamiento campeón/retador ──► exportar
```

## 2. Datos
| Fuente | Qué aporta | Periodo |
|---|---|---|
| SST de satélite L4 (0,05°) | Temperatura de laguna y costa: **objetivo** | 1982 → sep 2026 |
| Clorofila de satélite L4 (1 km) | Entrada | 1997 → sep 2026 |
| Modelo físico de Copernicus (1/24°) | Física de la costa (la laguna es tierra para el modelo) | 1998 → sep 2026 |
| ERA5 (Open-Meteo) | Aire, lluvia, viento y radiación en la cuenca y el mar | 1982 → sep 2026 |
| Previsiones archivadas (Open-Meteo) | Meteorología prevista a 1–7 días, **la emitida de verdad** | feb 2024 → oct 2026 |
| UPCT in situ (10 profundidades) | O₂ (objetivo de hipoxia), T, S, clorofila y turbidez; cubo 4D | mar 2017 → jul 2024 (380 campañas) |

## 3. Resultados principales
| Pregunta | Resultado | Doc |
|---|---|---|
| ¿Bate el modelo de SST a las líneas base? | Sí, a la persistencia amortiguada en los **8 casos** (laguna +4–10 % frente a la mejor línea base). Antes de añadir la previsión, la costa empataba con la persistencia a 3–7 días. | [Fase 3](fase3_informe.md) |
| ¿Cuánto aporta la previsión meteorológica? | **−17,5 % (laguna) y −26 % (costa) de error a 7 días** con las previsiones realmente emitidas. F1 de ola de calor a 7 días de 0,59/0,57 a **0,76/0,75**. Nada a 30 días. | [Fase 7 · previsión](fase7_prevision.md) |
| ¿Se puede anticipar la hipoxia? | En parte: la logística **anticipa 4 de 9 inicios de episodio** (la persistencia, 1), con precisión de 0,48. Con 30 eventos, XGBoost no supera a la persistencia. | [Fase 3](fase3_informe.md) |
| ¿Cómo de fiable es la incertidumbre? | El intervalo conformal estacional cubre el **87 %** (objetivo: 90 %) y mejora el interval score un 8 %. | [Fase 7 · deriva](fase7_deriva.md) |
| ¿Qué pasa con +1,5 °C? | Como episodio cálido, se disipa en parte. Como cambio de clima de fondo, la probabilidad de ola de calor a 7 días pasa del **24 % al 81 %**. | [Fase 5](fase5_escenarios.md) |
| ¿Envejecen los modelos? | Sí. El de 30 días dejó de batir a la persistencia en los veranos de 2023–2026. Reentrenado: **+14 %**, y vuelve a batirla. | [Fase 7 · deriva](fase7_deriva.md) |
| ¿Qué aporta ver el agua en 3D? | El 10-08-2023 la **mitad del volumen** de la laguna estaba en hipoxia, aunque solo el 23 % del fondo. En la DANA de 2019, el 26 % del volumen estaba anóxico. | [Fase 7 · visor](fase7_visor4d.md) |

**Modelos en servicio:**
- SST a 1 y 3 días: v3 (con previsión, reentrenados hasta septiembre de 2026).
- SST a 7 días: costa v3; laguna v2 (su retador mejoró solo un 1,99 %, por debajo del umbral).
- SST a 30 días: v2 (reentrenados, sin previsión).
- Respaldos sin previsión: v1.
- Hipoxia: logística v2 (sin viento).

## 4. Hallazgos y lecciones (XAI y MLaaS)
1. **Explicar no basta: hay que actuar sobre lo que la explicación revela.**
   - SHAP mostró en la Fase 3 que la hipoxia "subía con el viento", al revés que la física.
   - Los escenarios de la Fase 5 lo convirtieron en un error visible ("menos viento, menos riesgo").
   - Se corrigió con una v2 sin viento, que no empeora las métricas.
2. **Los fallos explicados dicen qué mejorar.** El peor fallo (octubre de 2023, −5,5 °C) lo causó un temporal posterior a la predicción. De ahí salió la mayor mejora del proyecto: la previsión meteorológica como entrada.
3. **Los árboles no extrapolan.** En el récord de julio de 2025, aire +3 °C y +6 °C daban la misma predicción. Los avisos fuera de rango y el reentrenamiento con datos recientes son imprescindibles en un mar que se calienta.
4. **Un umbral de manual puede no servir.** El PSI de 0,25 marcaba deriva todos los meses: las anomalías persisten semanas. Calibrado con años normales, el umbral real es de 4–7.
5. **Lo que parece deriva puede ser mala calibración.** La cobertura del intervalo caía cada verano por mezclar errores de verano y de invierno, no por cambio climático.
6. **Se evalúa sobre el objetivo, no sobre un intermediario.** Una previsión mejor corregida (−19 % de error) no mejoraba la SST, así que no se cambió.
7. **Honestidad por diseño**, siempre con su test:
   - Particiones temporales con purga de frontera.
   - Climatologías sin datos de validación.
   - Previsiones reales para evaluar (no las de "previsión perfecta").
   - `datos_hasta` por modelo: un modelo recién reentrenado aparece como "aún no evaluable", no con aciertos ficticios.

## 5. Qué no se hizo (y por qué)
| Propuesto en el plan | Decisión | Motivo |
|---|---|---|
| ConvLSTM / Temporal Fusion Transformer | No | La laguna tiene 6 celdas de SST de satélite: no hay campo espacial que modelar. Requiere GPU, y XGBoost con previsión ya bate a las líneas base. |
| Modelos híbridos con física | No | Ningún modelo físico de Copernicus resuelve la laguna (para todos es tierra). |
| Tiempo real con MQTT/Kafka | No | No hay sensores en vivo: los datos de la UPCT acaban en 2024 y el satélite llega con retraso. |
| PostGIS / TimescaleDB | No | Con unos pocos Parquet y una API, una base de datos no aporta nada a esta escala. |
| Airflow | No | Es una sola máquina con 8 GB de RAM y un flujo lineal: basta `ciclo_mantenimiento.py` con el Programador de tareas. |
| Nitratos y vertidos en los escenarios | No | No son entradas de ningún modelo, porque no hay datos. La API lo dice en `/scenario/drivers`. |

## 6. Limitaciones conocidas
- **No es un gemelo en tiempo real.** Actualizar al día de hoy exige descargar datos a mano y ampliar `periodo.fin`; los productos NRT de Copernicus no están verificados.
- **La hipoxia no se puede vigilar ni usar con datos actuales:** no hay campañas in situ desde julio de 2024.
- La SST de satélite **suaviza la laguna**: 6 celdas, con un sesgo estacional de ±2 °C frente a la medida in situ.
- **Los intervalos cubren el 87 %**, algo por debajo del 90 % nominal.
- **Los reentrenos se deciden con una sola ventana de 180 días:** las mejoras del 2–4 % podrían ser ruido.
- **Los respaldos sin previsión (v1) no se reentrenan.**
- **La API no tiene autenticación:** está pensada para uso local.

## 7. Trabajo futuro (por orden de valor)
1. **Operación en tiempo real:**
   - Verificar los productos NRT de Copernicus (SST y física).
   - Automatizar la descarga diaria y ampliar `periodo.fin`.
   - Encadenarlo en `ciclo_mantenimiento.py`.
2. **Hipoxia:** conseguir datos in situ posteriores a 2024 (UPCT o IEO-CSIC, proyecto BELICH) y reentrenar y vigilar el modelo.
3. **Decisiones de reentrenamiento más robustas:** varias ventanas o bootstrap por bloques antes de promocionar; reentrenar también los respaldos.
4. **Intervalos:** alcanzar el 90 % nominal, por ejemplo con conformal adaptativo.
5. **Despliegue fuera de local:** autenticación, HTTPS y un servidor o nube.

## 8. Cómo reproducirlo
Ver el [README](../README.md): instalación, orden de los scripts y arranque.

**Comprobado el 2026-11-12 con un clon limpio**, instalado desde cero en un entorno aislado (sin paquetes del usuario):
- **Instalación:** `pip install -c requirements.lock.txt -e ".[datos,ml,api,dashboard,dev]"`, 6,7 min, sin errores y sin dependencias rotas (`pip check`).
- **Sin `data/` ni `models/`:** pasan 44 tests y se saltan 32 con un aviso claro ("ejecuta el pipeline, entrena y exporta"). Ningún fallo.
- **Con todo generado:** pasan los 76.

La comprobación destapó un problema real, ya corregido. JupyterLab no se instalaba en Windows si el proyecto estaba en una carpeta con ruta larga, por el límite de 260 caracteres. Ahora va en un grupo aparte (`notebooks`) y el README explica cómo evitarlo.
