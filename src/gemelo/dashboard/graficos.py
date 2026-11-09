"""Gráficos Plotly del dashboard. Funciones puras (datos -> figura): se pueden probar sin Streamlit."""
import pandas as pd
import plotly.graph_objects as go

AZUL, NARANJA, ROJO, GRIS, VERDE = "#2c7fb8", "#f28e2b", "#d62728", "#8c8c8c", "#2ca02c"
NOMBRES_GRUPO = {
    "temp_laguna": "Temperatura laguna", "temp_costa": "Temperatura costa", "aire_y_radiacion": "Aire y radiación",
    "viento": "Viento", "lluvia": "Lluvia", "calendario": "Época del año", "estratificacion": "Estratificación / mezcla",
    "clorofila_satelite": "Clorofila (satélite)", "oxigeno_in_situ": "Oxígeno in situ", "agua_in_situ": "Agua in situ",
    "muestreo": "Muestreo",
}


RGBA = {ROJO: "rgba(214,39,40,{a})", NARANJA: "rgba(242,142,43,{a})", AZUL: "rgba(44,127,184,{a})",
        GRIS: "rgba(140,140,140,{a})"}


def _estilo(fig, titulo, alto=380):
    fig.update_layout(title=titulo, height=alto, margin=dict(l=10, r=10, t=50, b=10),
                      legend=dict(orientation="h", yanchor="top", y=-0.12, x=0), hovermode="x unified")
    return fig


def mapa(estado: dict) -> go.Figure:
    """Subzonas coloreadas por anomalía de SST y puntos meteorológicos."""
    fig = go.Figure()
    for zona, z in estado["zonas"].items():
        (la0, la1), (lo0, lo1) = z["caja"]["lat"], z["caja"]["lon"]
        an = z["anomalia"]
        color = ROJO if z["ola_calor_en_curso"] else (NARANJA if an > 0.5 else (AZUL if an < -0.5 else GRIS))
        fig.add_trace(go.Scattermap(
            lat=[la0, la0, la1, la1, la0], lon=[lo0, lo1, lo1, lo0, lo0], mode="lines", fill="toself",
            fillcolor=RGBA[color].format(a=0.25), line=dict(color=color, width=2), name=zona.capitalize(),
            hovertext=f"{zona}: {z['sst']} °C (anomalía {an:+.2f} °C)"
                      + (" · OLA DE CALOR" if z["ola_calor_en_curso"] else ""), hoverinfo="text"))
    pts = estado["puntos_meteo"]
    fig.add_trace(go.Scattermap(lat=[p["lat"] for p in pts.values()], lon=[p["lon"] for p in pts.values()],
                                mode="markers+text", text=[f"ERA5 {k}" for k in pts], textposition="top right",
                                marker=dict(size=9, color="black"), name="Meteorología", hoverinfo="text"))
    fig.update_layout(map=dict(style="carto-positron", center=dict(lat=37.75, lon=-0.75), zoom=8.2),
                      height=420, margin=dict(l=0, r=0, t=0, b=0), showlegend=False)
    return fig


def serie_con_prediccion(serie: pd.DataFrame, preds: list[dict], zona: str) -> go.Figure:
    """SST observada + climatología + umbral, y predicciones con intervalo del 90 %."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=serie.index, y=serie["umbral_ola_calor"], name="Umbral ola de calor (p90)",
                             line=dict(color=ROJO, dash="dash", width=1)))
    fig.add_trace(go.Scatter(x=serie.index, y=serie["climatologia"], name="Climatología",
                             line=dict(color=GRIS, dash="dot", width=1)))
    fig.add_trace(go.Scatter(x=serie.index, y=serie["sst"], name="SST observada", line=dict(color=AZUL, width=2)))
    if preds:
        p = pd.DataFrame(preds)
        x = pd.to_datetime([preds[0]["fecha_base"]] + list(p["fecha_objetivo"]))
        hoy = preds[0]["sst_hoy"]
        lo = [hoy] + [i[0] for i in p["intervalo_90"]]
        hi = [hoy] + [i[1] for i in p["intervalo_90"]]
        fig.add_trace(go.Scatter(x=list(x) + list(x[::-1]), y=hi + lo[::-1], fill="toself", fillcolor="rgba(242,142,43,0.2)",
                                 mode="lines", line=dict(width=0), name="Intervalo 90 %", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=x, y=[hoy] + list(p["sst_prevista"]), name="Predicción", mode="lines+markers",
                                 line=dict(color=NARANJA, width=2),
                                 text=[""] + [f"h={h} d · P(ola)={pr:.0%}" for h, pr in zip(p["h"], p["prob_superar_umbral"])]))
        reales = p.dropna(subset=["sst_real"])
        if len(reales):
            fig.add_trace(go.Scatter(x=pd.to_datetime(reales["fecha_objetivo"]), y=reales["sst_real"], mode="markers",
                                     name="Lo que pasó", marker=dict(color="black", symbol="x", size=9)))
    fig.update_yaxes(title="°C")
    return _estilo(fig, f"Temperatura del agua · {zona}", 420)


def barras_grupos(grupos: list[dict], titulo: str, unidad: str = "°C") -> go.Figure:
    g = pd.DataFrame(grupos)
    if g.empty:
        return _estilo(go.Figure(), titulo + " (sin cambios)", 250)
    g = g.iloc[::-1]
    fig = go.Figure(go.Bar(x=g["contribucion"], y=[NOMBRES_GRUPO.get(n, n) for n in g["grupo"]], orientation="h",
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


def comparar_escenario(horizontes: list[dict]) -> go.Figure:
    """Base vs escenario por horizonte: mediana e intervalo 90 % de la SST, y umbral."""
    fig = go.Figure()
    etiquetas = [f"h={x['h']} d" for x in horizontes]
    for clave, nombre, color, desp in [("base", "Base", AZUL, -0.12), ("escenario", "Escenario", NARANJA, 0.12)]:
        p50 = [x[clave]["sst_p50"] for x in horizontes]
        fig.add_trace(go.Scatter(
            x=[i + desp for i in range(len(horizontes))], y=p50, mode="markers", name=nombre,
            marker=dict(size=11, color=color),
            error_y=dict(type="data", symmetric=False, array=[x[clave]["sst_p95"] - m for x, m in zip(horizontes, p50)],
                         arrayminus=[m - x[clave]["sst_p05"] for x, m in zip(horizontes, p50)]),
            text=[f"P(ola)={x[clave]['prob_ola_calor']:.0%}" for x in horizontes]))
    fig.add_trace(go.Scatter(x=list(range(len(horizontes))), y=[x["umbral_ola_calor"] for x in horizontes],
                             mode="lines+markers", name="Umbral ola de calor", line=dict(color=ROJO, dash="dash")))
    fig.update_xaxes(tickvals=list(range(len(horizontes))), ticktext=etiquetas)
    fig.update_yaxes(title="SST (°C)")
    return _estilo(fig, "SST prevista: base vs escenario (mediana e intervalo 90 %)")


def prob_ola(horizontes: list[dict]) -> go.Figure:
    etiquetas = [f"h={x['h']} d" for x in horizontes]
    fig = go.Figure([go.Bar(x=etiquetas, y=[x["base"]["prob_ola_calor"] for x in horizontes], name="Base", marker_color=AZUL),
                     go.Bar(x=etiquetas, y=[x["escenario"]["prob_ola_calor"] for x in horizontes], name="Escenario",
                            marker_color=NARANJA)])
    fig.update_yaxes(range=[0, 1], tickformat=".0%")
    return _estilo(fig, "Probabilidad de ola de calor marina", 320)


def historial_hipoxia(campanas: pd.DataFrame, umbral_pct: float, seleccion=None) -> go.Figure:
    fig = go.Figure(go.Scatter(x=campanas.index, y=campanas["pct_fondo_hipoxia"], mode="lines+markers",
                               marker=dict(size=4), line=dict(color=AZUL), name="% fondo < 4 mg/L"))
    fig.add_hline(y=umbral_pct, line=dict(color=ROJO, dash="dash"), annotation_text=f"evento (> {umbral_pct:g} %)")
    if seleccion is not None:
        fig.add_vline(x=pd.Timestamp(seleccion), line=dict(color=NARANJA, width=2))
    fig.update_yaxes(title="% del fondo en hipoxia")
    return _estilo(fig, "Hipoxia en el fondo de la laguna por campaña (UPCT)", 330)
