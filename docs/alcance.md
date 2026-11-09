# Documento de alcance — Gemelo digital marino del Mar Menor

*Fase 0 · 2026-11-09*

## Objetivo
Construir un gemelo digital del **Mar Menor y su costa mediterránea adyacente** que describa, prediga, simule escenarios y explique (XAI) el estado térmico del agua y, como módulo secundario, el riesgo de hipoxia en la laguna.

## Zona
- Recuadro de trabajo: **37,50–38,00 N · 0,95–0,50 W**.
- Dos subzonas: **laguna** (dentro de La Manga) y **costa** (mar abierto al este de La Manga). Se separarán con las máscaras de tierra/mar de cada producto.

## Preguntas
| Módulo | Pregunta | Horizontes |
|---|---|---|
| **Principal** | ¿Qué temperatura superficial (SST) tendrá el agua y habrá **ola de calor marina** (MHW)? | 1, 3, 7 días · 30 días *(experimental)* |
| **Secundario** | ¿Habrá **hipoxia** (O₂ < 4 mg/L) o anoxia (< 2 mg/L) en el fondo de la laguna? | próxima campaña (~7 días) |

Ola de calor marina: SST > percentil 90 de la climatología diaria 1991–2020 durante ≥ 5 días seguidos (definición de Hobday et al., 2016).

## Datos (verificados en el catálogo, nov. 2026)
| Fuente | Dataset | Uso | Resolución · periodo |
|---|---|---|---|
| SST satélite L4 | `cmems_SST_MED_SST_L4_REP_OBSERVATIONS_010_021` | **Objetivo** principal y entrada | 0,05°, diaria · 1982 → oct 2026 |
| Clorofila satélite L4 | `cmems_obs-oc_med_bgc-plankton_my_l4-gapfree-multi-1km_P1D` | Entrada | 1 km, diaria · 1997 → oct 2026 |
| Física (modelo) | `cmems_mod_med_phy-temp/-mld/-hflux/-mflux_my_4.2km_P1D-m` | Entrada (solo costa) | 1/24°, diaria · 1987 → sep 2026 |
| Biogeoquímica (modelo) | `cmems_mod_med_bgc-*_my_4.2km_P1D-m` | Entrada (solo costa) | 1/24°, diaria · 1999 → sep 2026 |
| O₂ in situ (UPCT) | THREDDS `L2/Oxigeno.nc` | **Objetivo** secundario | 200×200 pts, 10 prof. (0,5–5 m), semanal · mar 2017 → jul 2024 |

**Limitaciones conocidas**
- Los modelos de Copernicus tratan la laguna como **tierra**: solo sirven para la costa.
- La SST de satélite cubre la laguna con pocas celdas (0,05° ≈ 5 km).
- El O₂ de la UPCT tiene ~388 muestras semanales, pocos episodios de hipoxia (2019, 2021) y no se actualiza desde jul 2024.
- Meteorología (ERA5: viento, temperatura del aire, lluvia) mejoraría los dos módulos, pero requiere una cuenta aparte en Copernicus Climate. Se decide en la Fase 1.

## Resolución y partición
- Paso temporal **diario** (módulo principal), **semanal** (hipoxia).
- Primer modelo sobre **series medias por subzona**; la predicción por celda llega después.
- Partición temporal (módulo principal): **train 1998–2018 · validación 2019–2021 · test 2022–2026**.
- Hipoxia: validación cruzada temporal por años (dejar un año fuera), por la escasez de datos.

## Métricas de éxito
- **SST**: RMSE y MAE por horizonte. Éxito = *skill score* > 0 (1 − RMSE_modelo / RMSE_base) frente a la mejor línea base (persistencia o climatología) a 1, 3 y 7 días.
- **MHW**: recall y F1 de días en evento sobre test.
- **Hipoxia**: recall, precisión y Brier score; se prioriza el recall (una alerta perdida es peor que una falsa).
- **XAI**: explicaciones SHAP coherentes con la física conocida (inercia térmica, estación, flujos de calor).

## Fuera de alcance (por ahora)
Tiempo real, PostGIS/TimescaleDB, Airflow, MQTT, CesiumJS, modelos profundos: Fase 7.
