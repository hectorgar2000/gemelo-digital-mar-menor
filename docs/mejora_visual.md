# Mejora visual del dashboard

*2026-11-13 · v0.11.0 · `src/gemelo/dashboard/` y `src/gemelo/servicio/visor/`*

Objetivo: que alguien que abre el dashboard entienda **en cinco segundos** cómo está el mar y qué se espera, y que cada pestaña cuente una cosa con un gráfico pensado para ella.

## Qué ha cambiado
| Antes | Ahora | Por qué |
|---|---|---|
| Cuatro cifras sueltas arriba | **Resumen en una frase** con franja de color (rojo: ola de calor · naranja: más cálida de lo normal · verde: normal) y cuatro tarjetas con borde | Lo primero que se lee es la conclusión, no los números |
| Mapa con dos rectángulos | **Contorno real de la laguna**, con la temperatura escrita sobre cada zona y la zona activa resaltada | Un rectángulo sobre tierra no se reconoce como el Mar Menor |
| Serie con la predicción | Además: **olas de calor sombreadas**, línea del día base y, en días del pasado, **lo que ocurrió después** | Se ve de un vistazo cuánto del verano ha sido ola de calor y si la predicción acertó |
| Dos gráficos de barras para explicar | **Cascada** (del punto de partida a la predicción, grupo a grupo) + las 12 variables | La cascada muestra cómo se suman las contribuciones; las barras solo las listaban |
| Siete deslizadores para un escenario | **Escenarios de un clic** (aire muy cálido, temporal de viento, semana en calma, clima +1,5 °C, reiniciar) | Los casos típicos no exigen saber qué deslizador mover |
| Probabilidad de hipoxia como cifra | **Indicador de aguja** con la zona de alerta marcada | Se ve si se ha cruzado el umbral, no solo el número |
| Visor 4D en otra ventana | **Pestaña "Visor 4D"** dentro del dashboard (y enlace a pantalla completa) | Todo el gemelo en un solo sitio |
| URL de la API a la vista | Dentro de "Conexión", plegado | Es un ajuste técnico, no algo que el usuario deba ver |
| Tablas con `None` y muchos decimales | Fechas, °C y probabilidad con formato (barra de progreso) | Legibilidad |
| Rojo por defecto de Streamlit | Azul marino como color principal (`.streamlit/config.toml`) | Coherencia con el tema del proyecto |

## Decisiones técnicas
- **Contorno de la laguna** (`procesado.cubo.contorno_laguna`): sale de la rejilla in situ de la UPCT. Para cada fila de latitud se toma el punto de agua más al oeste y el más al este. Se guarda en `data/processed/contorno_laguna.json` (paso 6/7 del pipeline) y la API lo sirve en `GET /estado`. Si no existe, el mapa dibuja el recuadro de siempre.
- **El resumen no usa `st.error`**: el estado del mar no es un error de la aplicación. Además, los tests comprueban que no haya errores en pantalla. Es un aviso propio en HTML con tres niveles.
- **Mapa siempre claro**: Streamlit no informa de forma fiable del tema activo (devolvía "oscuro" con la página en claro), y el mapa claro se lee bien sobre los dos temas.
- **Visor integrado con `st.iframe`**: apunta a `GEMELO_API_URL_PUBLICA/visor/`. El visor redibuja su mini-serie cuando cambia de tamaño (`ResizeObserver`), porque dentro de una pestaña oculta el lienzo mide 0 al cargar.
- **Los gráficos son funciones puras** (`dashboard/graficos.py`): datos → figura, sin Streamlit. Por eso tienen tests propios (`tests/test_graficos.py`).

## Verificación
- En el navegador: las seis pestañas, un escenario de un clic ("Temporal de viento": la probabilidad de ola de calor a 7 días pasa del 90 % al 2 %) y el visor integrado.
- 92 tests. Los nuevos comprueban el contorno, los colores del mapa, el sombreado de olas de calor, la aditividad de la cascada y la aguja.
- **No verificado:** las imágenes Docker 0.10.0 y 0.11.0 (Docker Desktop estaba parado).

## Limitaciones
- El contorno es aproximado (una rejilla de ~330 × 450 m) y no dibuja islas.
- La costa sigue siendo un recuadro: es una zona de mar abierto sin forma propia.
- Las capturas automáticas de mapas y del visor (WebGL) a veces salen en blanco aunque estén bien; hay que comprobar el estado antes de darlos por rotos.
