"""Gráficos Plotly del dashboard. Funciones puras (datos -> figura): se pueden probar sin Streamlit.

Estilo: el sistema visual del visor 4D (rediseño de Claude Design). Fondo oscuro, IBM Plex y la paleta del visor,
que es la de sus rampas de datos (rojo-azul): rojo = calor/hipoxia, azul = frío/observado, salmón = predicción.
"""
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

# Paleta (las mismas variables CSS que visor.css y .streamlit/config.toml)
AZUL, NARANJA, ROJO, GRIS, VERDE = "#6aaed6", "#f4a582", "#d6604d", "#a9b0bb", "#7fc8a9"
TEXTO, TEXTO_SUAVE, BORDE, FONDO_CAMPO = "#e8e8e8", "#a9b0bb", "#39414f", "#1c2230"
REJILLA = "rgba(169,176,187,0.12)"
FUENTE, FUENTE_MONO = "IBM Plex Sans, system-ui, sans-serif", "IBM Plex Mono, ui-monospace, monospace"


def _rgba(color: str, a: float) -> str:
    """'#rrggbb' -> 'rgba(r,g,b,a)' (rellenos y sombreados translúcidos)."""
    r, g, b = (int(color[i:i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r},{g},{b},{a})"


_eje = dict(gridcolor=REJILLA, zerolinecolor=BORDE, linecolor=BORDE, tickcolor=BORDE,
            tickfont=dict(family=FUENTE_MONO, size=11, color=TEXTO_SUAVE), title=dict(font=dict(size=12, color=TEXTO_SUAVE)))
pio.templates["gemelo"] = go.layout.Template(layout=dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family=FUENTE, size=12.5, color=TEXTO),
    title=dict(font=dict(family=FUENTE, size=15, color="#ffffff"), x=0, xanchor="left"),
    colorway=[AZUL, NARANJA, ROJO, GRIS, VERDE],
    xaxis=_eje, yaxis=_eje,
    legend=dict(font=dict(size=11.5, color=TEXTO_SUAVE), bgcolor="rgba(0,0,0,0)"),
    hoverlabel=dict(bgcolor=FONDO_CAMPO, bordercolor=BORDE, font=dict(family=FUENTE, size=12, color=TEXTO)),
))
NOMBRES_GRUPO = {
    "temp_laguna": "Temperatura de la laguna", "temp_costa": "Temperatura de la costa",
    "aire_y_radiacion": "Aire y radiación (pasado)", "viento": "Viento (pasado)", "lluvia": "Lluvia",
    "calendario": "Época del año", "estratificacion": "Estratificación / mezcla",
    "clorofila_satelite": "Clorofila (satélite)", "oxigeno_in_situ": "Oxígeno in situ", "agua_in_situ": "Agua in situ",
    "muestreo": "Muestreo", "meteo_prevista": "Meteorología prevista", "intercambio_aire_mar": "Intercambio aire-mar",
}


def nombre_grupo(g: str) -> str:
    return NOMBRES_GRUPO.get(g, g)


def _estilo(fig, titulo, alto=380):
    fig.update_layout(template="gemelo", title=titulo, height=alto, margin=dict(l=10, r=10, t=50, b=10),
                      legend=dict(orientation="h", yanchor="top", y=-0.12, x=0), hovermode="x unified")
    return fig


def color_zona(z: dict) -> str:
    """Rojo = ola de calor en curso · naranja = más cálida de lo normal · azul = más fría · gris = normal."""
    if z["ola_calor_en_curso"]:
        return ROJO
    return NARANJA if z["anomalia"] > 0.5 else (AZUL if z["anomalia"] < -0.5 else GRIS)


# ------------------------------------------------------------------ mapa
def mapa(estado: dict, zona_activa: str | None = None, oscuro: bool = True) -> go.Figure:
    """Laguna con su contorno real (si la API lo da) y costa, coloreadas por anomalía; celdas ERA5.

    Mapa base oscuro por defecto: el dashboard tiene tema oscuro fijo (no depende del sistema), así que ya no
    hace falta detectar el tema activo. `oscuro=False` da el mapa claro."""
    fig = go.Figure()
    for zona, z in estado["zonas"].items():
        if z.get("contorno"):
            lon, lat = [p[0] for p in z["contorno"]], [p[1] for p in z["contorno"]]
        else:
            (la0, la1), (lo0, lo1) = z["caja"]["lat"], z["caja"]["lon"]
            lat, lon = [la0, la0, la1, la1, la0], [lo0, lo1, lo1, lo0, lo0]
        color = color_zona(z)
        activa = zona == zona_activa
        fig.add_trace(go.Scattermap(
            lat=lat, lon=lon, mode="lines", fill="toself", fillcolor=_rgba(color, 0.45 if activa else 0.2),
            line=dict(color=color, width=3 if activa else 1.5), name=zona.capitalize(),
            hovertext=f"<b>{zona.capitalize()}</b><br>{z['sst']} °C · anomalía {z['anomalia']:+.2f} °C"
                      + (f"<br>Ola de calor: {z['dias_sobre_umbral']} días" if z["ola_calor_en_curso"] else ""),
            hoverinfo="text"))
        # etiqueta con la temperatura en el centro de la zona
        fig.add_trace(go.Scattermap(lat=[sum(lat) / len(lat)], lon=[sum(lon) / len(lon)], mode="text",
                                    text=[f"{z['sst']:.1f} °C"],
                                    textfont=dict(family=FUENTE, size=15, color="#ffffff" if oscuro else "#0e1117"),
                                    hoverinfo="skip"))
    pts = estado["puntos_meteo"]
    fig.add_trace(go.Scattermap(lat=[p["lat"] for p in pts.values()], lon=[p["lon"] for p in pts.values()],
                                mode="markers", marker=dict(size=8, color=TEXTO if oscuro else "#0e1117"),
                                hovertext=[f"Meteorología ERA5 · {k}" for k in pts], hoverinfo="text"))
    fig.update_layout(template="gemelo", map=dict(style="carto-darkmatter" if oscuro else "carto-positron",
                                                  center=dict(lat=37.76, lon=-0.70), zoom=8.1),
                      height=430, margin=dict(l=0, r=0, t=0, b=0), showlegend=False)
    return fig


# ------------------------------------------------------------------ serie y predicción
def _rachas(mascara: pd.Series) -> list[tuple]:
    """Tramos [inicio, fin] de días consecutivos con mascara=True."""
    tramos, ini, ant = [], None, None
    for f, v in mascara.items():
        if v and ini is None:
            ini = f
        if not v and ini is not None:
            tramos.append((ini, ant))
            ini = None
        ant = f
    if ini is not None:
        tramos.append((ini, ant))
    return tramos


def serie_con_prediccion(serie: pd.DataFrame, preds: list[dict], zona: str, fecha_base=None) -> go.Figure:
    """SST observada + climatología + umbral, con los días sobre el umbral sombreados; predicción con
    intervalo del 90 %. Si `serie` llega más allá de `fecha_base` (día del pasado), se dibuja lo que ocurrió."""
    fecha_base = pd.Timestamp(fecha_base) if fecha_base is not None else serie.index.max()
    pasado, futuro = serie[serie.index <= fecha_base], serie[serie.index >= fecha_base]
    fig = go.Figure()
    for ini, fin in _rachas(pasado["sst"] > pasado["umbral_ola_calor"]):
        if (fin - ini).days >= 4:   # ola de calor: 5 días o más seguidos sobre el umbral
            fig.add_vrect(x0=ini, x1=fin + pd.Timedelta(days=1), fillcolor=_rgba(ROJO, 0.13), line_width=0)
    fig.add_trace(go.Scatter(x=serie.index, y=serie["umbral_ola_calor"], name="Umbral de ola de calor",
                             line=dict(color=ROJO, dash="dash", width=1)))
    fig.add_trace(go.Scatter(x=serie.index, y=serie["climatologia"], name="Lo normal en estas fechas",
                             line=dict(color=GRIS, dash="dot", width=1)))
    fig.add_trace(go.Scatter(x=pasado.index, y=pasado["sst"], name="Temperatura observada", line=dict(color=AZUL, width=2.2)))
    if len(futuro) > 1:
        fig.add_trace(go.Scatter(x=futuro.index, y=futuro["sst"], name="Lo que ocurrió después",
                                 line=dict(color=AZUL, width=1.5, dash="dot")))
    if preds:
        p = pd.DataFrame(preds)
        x = pd.to_datetime([preds[0]["fecha_base"]] + list(p["fecha_objetivo"]))
        hoy = preds[0]["sst_hoy"]
        lo = [hoy] + [i[0] for i in p["intervalo_90"]]
        hi = [hoy] + [i[1] for i in p["intervalo_90"]]
        fig.add_trace(go.Scatter(x=list(x) + list(x[::-1]), y=hi + lo[::-1], fill="toself", mode="lines",
                                 fillcolor=_rgba(NARANJA, 0.22), line=dict(width=0), name="Intervalo del 90 %",
                                 hoverinfo="skip"))
        fig.add_trace(go.Scatter(
            x=x, y=[hoy] + list(p["sst_prevista"]), name="Predicción", mode="lines+markers",
            line=dict(color=NARANJA, width=2.2), marker=dict(size=8),
            hovertext=[""] + [f"a {h} días · P(ola de calor) {pr:.0%}" for h, pr in zip(p["h"], p["prob_superar_umbral"])]))
    fig.add_vline(x=fecha_base, line=dict(color=GRIS, width=1))
    fig.add_annotation(x=fecha_base, y=1.0, yref="paper", text="DÍA BASE", showarrow=False, yanchor="bottom",
                       font=dict(family=FUENTE_MONO, size=10.5, color=GRIS))
    fig.update_yaxes(title="°C")
    return _estilo(fig, f"Temperatura del agua · {zona} (sombreado: olas de calor)", 430)


# ------------------------------------------------------------------ explicaciones
def cascada(grupos: list[dict], base: float, prediccion: float, unidad: str = "°C", max_grupos: int = 7) -> go.Figure:
    """Cascada: de la referencia del modelo a la predicción, sumando la contribución de cada grupo."""
    g = sorted(grupos, key=lambda x: -abs(x["contribucion"]))
    principales, resto = g[:max_grupos], g[max_grupos:]
    etiquetas = ["Punto de partida"] + [nombre_grupo(x["grupo"]) for x in principales]
    valores = [base] + [x["contribucion"] for x in principales]
    medidas = ["absolute"] + ["relative"] * len(principales)
    if resto:
        etiquetas.append(f"Otros ({len(resto)})")
        valores.append(sum(x["contribucion"] for x in resto))
        medidas.append("relative")
    etiquetas.append("Predicción")
    valores.append(prediccion)
    medidas.append("total")
    fig = go.Figure(go.Waterfall(
        x=etiquetas, y=valores, measure=medidas, text=[f"{v:+.2f}" for v in valores], textposition="outside",
        increasing=dict(marker=dict(color=ROJO)), decreasing=dict(marker=dict(color=AZUL)),
        totals=dict(marker=dict(color=NARANJA)), connector=dict(line=dict(color=GRIS, width=1))))
    fig.update_yaxes(title=f"cambio de anomalía ({unidad})")
    fig.update_layout(showlegend=False, hovermode="x")
    return _estilo(fig, "De dónde sale la predicción", 420)


def barras_grupos(grupos: list[dict], titulo: str, unidad: str = "°C") -> go.Figure:
    g = pd.DataFrame(grupos)
    if g.empty:
        return _estilo(go.Figure(), titulo + " (sin cambios)", 250)
    g = g.iloc[::-1]
    fig = go.Figure(go.Bar(x=g["contribucion"], y=[nombre_grupo(n) for n in g["grupo"]], orientation="h",
                           marker_color=[ROJO if v > 0 else AZUL for v in g["contribucion"]],
                           text=[f"{v:+.3f}" for v in g["contribucion"]], textposition="auto"))
    fig.update_xaxes(title=f"contribución ({unidad})")
    return _estilo(fig, titulo, 60 + 32 * len(g))


def barras_variables(variables: list[dict], titulo: str, unidad: str = "°C") -> go.Figure:
    v = pd.DataFrame(variables).iloc[::-1]
    etiqueta = [f"{r.variable} = {r.valor:g}" for r in v.itertuples()]
    fig = go.Figure(go.Bar(x=v["contribucion"], y=etiqueta, orientation="h",
                           marker_color=[ROJO if c > 0 else AZUL for c in v["contribucion"]]))
    fig.update_xaxes(title=f"contribución ({unidad})")
    return _estilo(fig, titulo, 60 + 28 * len(v))


# ------------------------------------------------------------------ escenarios
def comparar_escenario(horizontes: list[dict]) -> go.Figure:
    """Base vs escenario por horizonte: mediana e intervalo 90 % de la SST, y umbral."""
    fig = go.Figure()
    etiquetas = [f"a {x['h']} d" for x in horizontes]
    for clave, nombre, color, desp in [("base", "Sin cambios", AZUL, -0.12), ("escenario", "Escenario", NARANJA, 0.12)]:
        p50 = [x[clave]["sst_p50"] for x in horizontes]
        fig.add_trace(go.Scatter(
            x=[i + desp for i in range(len(horizontes))], y=p50, mode="markers", name=nombre,
            marker=dict(size=12, color=color),
            error_y=dict(type="data", symmetric=False, array=[x[clave]["sst_p95"] - m for x, m in zip(horizontes, p50)],
                         arrayminus=[m - x[clave]["sst_p05"] for x, m in zip(horizontes, p50)]),
            hovertext=[f"P(ola de calor) {x[clave]['prob_ola_calor']:.0%}" for x in horizontes]))
    fig.add_trace(go.Scatter(x=list(range(len(horizontes))), y=[x["umbral_ola_calor"] for x in horizontes],
                             mode="lines+markers", name="Umbral de ola de calor", line=dict(color=ROJO, dash="dash")))
    fig.update_xaxes(tickvals=list(range(len(horizontes))), ticktext=etiquetas)
    fig.update_yaxes(title="°C")
    return _estilo(fig, "Temperatura prevista (mediana e intervalo del 90 %)")


def prob_ola(horizontes: list[dict]) -> go.Figure:
    etiquetas = [f"a {x['h']} d" for x in horizontes]
    fig = go.Figure([
        go.Bar(x=etiquetas, y=[x["base"]["prob_ola_calor"] for x in horizontes], name="Sin cambios", marker_color=AZUL,
               text=[f"{x['base']['prob_ola_calor']:.0%}" for x in horizontes], textposition="outside"),
        go.Bar(x=etiquetas, y=[x["escenario"]["prob_ola_calor"] for x in horizontes], name="Escenario",
               marker_color=NARANJA, text=[f"{x['escenario']['prob_ola_calor']:.0%}" for x in horizontes],
               textposition="outside")])
    fig.update_yaxes(range=[0, 1.12], tickformat=".0%")
    return _estilo(fig, "Probabilidad de ola de calor marina", 380)


# ------------------------------------------------------------------ hipoxia
def aguja(prob: float, umbral: float = 0.5, titulo: str = "Riesgo de hipoxia") -> go.Figure:
    """Indicador de aguja 0-100 % con la zona de alerta marcada."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=100 * prob,
        number=dict(suffix=" %", valueformat=".0f", font=dict(family=FUENTE_MONO, size=40, color="#ffffff")),
        gauge=dict(axis=dict(range=[0, 100], tickcolor=BORDE, tickfont=dict(family=FUENTE_MONO, size=11, color=TEXTO_SUAVE)),
                   bar=dict(color=ROJO if prob >= umbral else VERDE, thickness=0.35),
                   bgcolor="rgba(0,0,0,0)", borderwidth=1, bordercolor=BORDE,
                   steps=[dict(range=[0, 100 * umbral], color=_rgba(VERDE, 0.14)),
                          dict(range=[100 * umbral, 100], color=_rgba(ROJO, 0.16))],
                   threshold=dict(line=dict(color=ROJO, width=3), value=100 * umbral))))
    fig.update_layout(template="gemelo", title=titulo, height=260, margin=dict(l=30, r=30, t=50, b=10))
    return fig


def historial_hipoxia(campanas: pd.DataFrame, umbral_pct: float, seleccion=None) -> go.Figure:
    # Rojo de hipoxia, como la serie del visor 4D; la campaña elegida, en blanco (el cursor del visor)
    fig = go.Figure(go.Scatter(x=campanas.index, y=campanas["pct_fondo_hipoxia"], mode="lines", fill="tozeroy",
                               fillcolor=_rgba(ROJO, 0.18), line=dict(color=ROJO, width=1.5),
                               name="% del fondo con O₂ < 4 mg/L"))
    fig.add_hline(y=umbral_pct, line=dict(color=TEXTO_SUAVE, dash="dash", width=1),
                  annotation_text=f"episodio (> {umbral_pct:g} %)", annotation_font=dict(color=TEXTO_SUAVE, size=11))
    if seleccion is not None:
        fig.add_vline(x=pd.Timestamp(seleccion), line=dict(color="#ffffff", width=1.5))
    fig.update_yaxes(title="% del fondo en hipoxia")
    fig.update_layout(showlegend=False)
    return _estilo(fig, "Hipoxia en el fondo de la laguna, campaña a campaña (UPCT)", 330)
