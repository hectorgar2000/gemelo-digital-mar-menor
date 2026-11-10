# Fase 7 · Bloque 3 — Visor 3D/4D con CesiumJS

*2026-11-10 · v0.9.0 · `http://localhost:8000/visor/` (o el botón "Abrir visor 4D" del dashboard)*

## Qué muestra
Los **cubos in situ de la UPCT** son los únicos datos realmente 4D del gemelo: latitud, longitud, 10 profundidades (0,5–5 m) y tiempo, con 380 campañas semanales entre 2017 y 2024. El visor los pinta como bloques de agua apilados en capas, con la superficie arriba, sobre un mapa de OpenStreetMap.

- **Variables:** oxígeno disuelto, temperatura, salinidad, clorofila y turbidez. El oxígeno usa una escala de color absoluta (0–12 mg/L, rojo = hipoxia); el resto, los percentiles 2–98 de toda la serie, para que los colores sean comparables entre campañas.
- **Tiempo:**
  - Deslizador, botones y reproducción automática, con precarga de la campaña siguiente.
  - Mini-serie del **% del volumen con O₂ < 4 mg/L** en cada campaña; se puede hacer clic para saltar.
  - Al abrir, el visor va a la campaña con más volumen en hipoxia.
- **Filtros:**
  - "Solo agua con O₂ < 4 mg/L", que deja ver la forma 3D del agua hipóxica.
  - Rango de profundidad (por ejemplo, solo 3–5 m para ver el fondo).
  - Exageración vertical (×100–×2000).

## Lo que se ve en 3D y en 2D no
| Campaña | % del **fondo** < 4 mg/L (Fase 2) | % del **volumen** < 4 mg/L | % del volumen < 2 mg/L |
|---|---|---|---|
| 2023-08-10 | 23 % | **51 %** | 1,6 % |
| 2021-08-19 | 61 % | 50 % | 3,5 % |
| 2019-09-26 (DANA) | 70 % | 40 % | **25,8 %** |

- **En agosto de 2023 el agua pobre en oxígeno subió por la columna:** la mitad de la laguna estaba en hipoxia aunque solo afectaba al 23 % del fondo. Mirando solo el fondo se infravaloraba el episodio.
- **En la DANA de 2019 el problema era la anoxia del fondo** (el 26 % del volumen por debajo de 2 mg/L), visible con el filtro de 3–5 m.

## Decisiones técnicas
| Decisión | Motivo |
|---|---|
| **Página estática servida por la API** (`/visor`, con `StaticFiles`) | Un solo servicio y mismo origen que `/cubo`, sin CORS. Funciona igual en local y en Docker. Los ficheros viajan como `package-data` dentro del paquete. |
| **CesiumJS 1.120.0 desde jsDelivr, sin Cesium ion** | Sin cuenta ni token: mapa base de OpenStreetMap y elipsoide sin relieve, que para una laguna a nivel del mar no hace falta. |
| **Cubo reducido** a bloques de 4×4 (~330 × 450 m) con ≥ 50 % de agua | ~8.000 puntos por campaña: fluido en el navegador. La regla del 50 % evita medias engañosas en la orilla. |
| **`GET /cubo` en columnas** (`lon`, `lat`, `profundidad`, `valor`) | El JSON pesa unas 4 veces menos que una lista de objetos. |
| **Carga perezosa del cubo** en la API | No ralentiza el arranque. Con el cubo cargado, la API usa ~400 MB de RAM. |
| **`cubo.py` no importa xarray al cargarse** | La imagen de la API no lleva xarray (Fase 4) y solo necesita la tabla de unidades. |
| **Capas apiladas sobre el mapa** (superficie arriba) en vez de bajo tierra | Cesium oculta lo que queda bajo el globo. Es una convención visual y se indica en el panel. |
| **URL pública aparte** (`GEMELO_API_URL_PUBLICA`) | En Docker el dashboard habla con `http://api:8000`, que el navegador del usuario no resuelve. |

## Pipeline y API
- `scripts/construir_dataset.py`, paso 6/7: genera `data/processed/cubo_insitu.parquet` (3,06 M filas, 66 MB) y `cubo_resumen.parquet` (indicadores 3D por campaña).
- `GET /cubo/fechas`: campañas con el % de volumen en hipoxia/anoxia, variables, unidades, rangos de color y profundidades.
- `GET /cubo?fecha=…&variable=…`: puntos 3D de una campaña. Devuelve un error 422 claro si no hubo campaña ese día.

## Verificación
- Tests (76 en total):
  - Estructura y coordenadas del cubo.
  - El % de anoxia declarado coincide con los puntos enviados.
  - Errores 422.
  - El visor se sirve con la versión de Cesium fijada.
- **Navegador:** carga sin token, la cámara oblicua deja ver las capas, y funcionan el filtro de hipoxia y el de profundidad (DANA de 2019, 3–5 m).
- **Docker 0.9.0:** el visor y `/cubo` se sirven desde el contenedor, y el enlace del dashboard apunta a `http://localhost:8000/visor/`.

## Limitaciones
- Los datos in situ acaban en julio de 2024: el visor es **histórico**, no muestra el estado actual.
- Son mapas **interpolados** por la UPCT a partir de perfiles en estaciones, no medidas en cada bloque.
- La profundidad máxima de la rejilla es 5 m: no se ven las zonas algo más profundas de la laguna (~6–7 m).
- Para montar el estado en tiempo real o las predicciones en 3D haría falta un modelo con resolución vertical, que no existe en este proyecto.
