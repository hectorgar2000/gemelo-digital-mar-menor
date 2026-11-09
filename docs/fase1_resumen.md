# Fase 1 — Qué dicen los datos

*2026-11-09 · Detalle y gráficos en [notebooks/01_exploracion.ipynb](../notebooks/01_exploracion.ipynb)*

## Datos descargados (`data/raw/`, fuera de git)
| Fichero | Contenido | Periodo | Tamaño |
|---|---|---|---|
| `copernicus/sst.nc` | SST satélite L4, 0,05° | 1982-01-01 → 2026-09-30 | 6 MB |
| `copernicus/clorofila.nc` | Clorofila satélite L4, 1 km | 1998-01-01 → 2026-09-30 | 69 MB |
| `copernicus/fisica_*.nc` | Modelo: temperatura (sup. y fondo), capa de mezcla, flujo de calor, viento en superficie | 1998 → 2026-09 · **flujos hasta 2025-07-31** | 31 MB |
| `upct/oxigeno_l2.nc` | O₂ in situ interpolado, 10 niveles 0,5–5 m | 2017-03-22 → 2024-07-18 | 119 MB |

Reproducible con `python scripts/descargar_copernicus.py` y `python scripts/descargar_upct.py`.

## Hallazgos

**Calidad y huecos**
- Los productos L4 y el modelo no tienen **ningún día faltante**. Los NaN son celdas de tierra dentro del recuadro.
- Los flujos de calor y viento del modelo **terminan en julio de 2025**. Habrá que sustituirlos (ERA5) o recortar el periodo.
- UPCT: 387 campañas válidas (una con fecha errónea de 1970, descartada), **cada 7 días** de mediana, con un hueco máximo de 35 días. Hay valores de O₂ exactamente 0,0 mg/L que probablemente sean artefactos de la interpolación: se revisarán en la Fase 2.

**Cobertura de la laguna**
- SST: **6 celdas** de laguna frente a 36 de costa. Clorofila: **91 celdas**. Modelo: **0** (confirmado).
- Caveat: en verano la laguna sale solo **+0,4 a +0,7 °C** más cálida que la costa (en invierno, −0,4 °C). El patrón es físicamente coherente para una laguna somera, pero a 5 km el satélite probablemente **suaviza** el contraste real. Es una limitación a tener en cuenta en las conclusiones.

**Temperatura (SST)**
- Tendencia 1982–2025: **+0,31 °C/década en la laguna** y +0,27 °C/década en la costa.
- Máximo anual medio: 28,1 °C en la laguna y 27,3 °C en la costa.
- Autocorrelación de la anomalía: 0,93 a 1 día, 0,83 a 3 días, 0,73 a 7 días y 0,44 a 30 días. **La persistencia será una línea base muy dura** a 1–3 días. A 30 días queda poca señal, como se esperaba.
- **Olas de calor marinas**: han pasado del 3,5 % de los días (1982–2000) al **20 %** (2015–2025) en la costa. Hubo 166–174 días en 2023 y 150–160 en 2025.

**Clorofila**
- La laguna tiene unos **10 veces más clorofila** que la costa (3–4,5 mg/m³ de media anual). Los máximos de 2016–2018 coinciden con la crisis de eutrofización conocida.

**Modelo frente a satélite (costa)**
- SST del modelo frente a la del satélite: sesgo −0,17 °C, RMSE 0,52 °C, r = 0,995. El modelo es fiable como fuente de variables para la costa.

**Oxígeno e hipoxia**
- Episodios con más del 5 % del fondo por debajo de 4 mg/L: **sep. 2017, sep.–oct. 2019 (hasta el 72 % del área, con anoxia), jul.–sep. 2021 (hasta el 61 %), ago. 2023 y verano de 2024 al alza**.
- Correlación SST laguna – O₂ de fondo: **−0,77**. Gran parte de esta relación es estacional (el agua caliente disuelve menos oxígeno), así que no hay que leerla como causa directa.
- Prueba con ERA5 (Open-Meteo, sin cuenta): la DANA del 12–13 de septiembre de 2019 dejó **84 y 90 mm** de lluvia. Dos semanas después, el 26 de septiembre, llegó la anoxia con el 70 % del área afectada. **La lluvia es una variable clave** que Copernicus Marine no aporta.

## Implicaciones para la Fase 2
1. **Añadir meteorología ERA5** (temperatura del aire, lluvia, viento, radiación) desde la API de Open-Meteo. No necesita cuenta y sustituye a los flujos que terminan en 2025.
2. **Deriva temporal**: el test (2022–2026) es mucho más cálido que el train y tiene muchas más olas de calor. Hay que trabajar con anomalías respecto a la climatología, no con valores absolutos, y vigilar la deriva.
3. **Hipoxia**: el objetivo candidato es "% del área de fondo < 4 mg/L supera el X %" en la siguiente campaña. Hay pocos episodios positivos (unos 6 eventos), así que habrá que usar validación cruzada por años.
4. Revisar los ceros de O₂ y el pico de superficie del 2019-09-19 (11,5 mg/L).
