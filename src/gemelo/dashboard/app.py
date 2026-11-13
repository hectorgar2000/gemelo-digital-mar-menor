"""Dashboard del gemelo digital del Mar Menor (Fase 6).

Arrancar (con la API ya en marcha):
    streamlit run src/gemelo/dashboard/app.py
La URL de la API se toma de GEMELO_API_URL (por defecto http://127.0.0.1:8000).
"""
import re
from datetime import date, timedelta

import pandas as pd
import streamlit as st

from gemelo.dashboard import graficos as G
from gemelo.dashboard.cliente import URL_POR_DEFECTO, URL_PUBLICA, ClienteGemelo, ErrorAPI

st.set_page_config(page_title="Gemelo digital · Mar Menor", page_icon="🌊", layout="wide")
HORIZONTES = [1, 3, 7, 30]


# ------------------------------------------------------------------ acceso a la API (con caché)
@st.cache_resource
def cliente(url: str) -> ClienteGemelo:
    return ClienteGemelo(url)


@st.cache_data(ttl=600, show_spinner=False)
def leer(url: str, metodo: str, *args, **kwargs):
    return getattr(cliente(url), metodo)(*args, **kwargs)


def llamar(metodo, *args, **kwargs):
    """Llama a la API y, si falla, muestra el error en la interfaz en lugar de romper la app."""
    try:
        return leer(st.session_state.url, metodo, *args, **kwargs)
    except ErrorAPI as e:
        st.error(str(e))
        st.stop()


def aviso_campana_antigua(fecha_campana, dia_base, max_dias: int = 14):
    """El riesgo de hipoxia se calcula con la última campaña in situ: si es antigua, no describe el presente."""
    dias = (dia_base - date.fromisoformat(str(fecha_campana))).days
    if dias > max_dias:
        st.warning(f"⚠️ La campaña in situ usada es del {fecha_campana}, **{dias} días** antes del día base: "
                   "describe el estado de la laguna entonces, no el actual. Los datos de la UPCT disponibles "
                   "terminan en julio de 2024.")


def resumen_estado(zona: str, z: dict, p7: dict) -> tuple[str, str]:
    """Una frase en lenguaje llano con lo esencial del día: (nivel, texto)."""
    lugar = "La laguna" if zona == "laguna" else "La costa"
    prob = p7["prob_superar_umbral"]
    futuro = (f"A 7 días se prevén **{p7['sst_prevista']:.1f} °C** "
              f"(entre {p7['intervalo_90'][0]:.1f} y {p7['intervalo_90'][1]:.1f} °C) y una probabilidad del "
              f"**{prob:.0%}** de estar en ola de calor.")
    if z["ola_calor_en_curso"]:
        return "alerta", (f"🔴 **{lugar} está en ola de calor marina** desde hace {z['dias_sobre_umbral']} días: "
                         f"{z['sst']:.1f} °C, {z['anomalia']:+.1f} °C sobre lo normal. {futuro}")
    if prob >= 0.5 or z["anomalia"] > 1.0:
        return "atencion", (f"🟠 **{lugar} está más cálida de lo normal** ({z['sst']:.1f} °C, {z['anomalia']:+.1f} °C), "
                           f"sin ola de calor declarada. {futuro}")
    return "normal", (f"🟢 **{lugar} está en valores normales** ({z['sst']:.1f} °C, {z['anomalia']:+.1f} °C respecto "
                       f"a lo habitual). {futuro}")


COLOR_ESTADO = {"alerta": "214,39,40", "atencion": "242,142,43", "normal": "44,160,44"}


def aviso_estado(nivel: str, texto: str):
    """Franja de color con el resumen. No se usa st.error: el estado del mar no es un error de la aplicación."""
    html = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", texto)   # las negritas de Markdown no se aplican dentro de HTML
    color = COLOR_ESTADO[nivel]
    st.markdown(f'<div style="border-left: 6px solid rgb({color}); background: rgba({color},0.10); '
                f'padding: 12px 16px; border-radius: 6px; margin-bottom: 12px;">{html}</div>', unsafe_allow_html=True)


# Escenarios de un clic: valores de los controles (los no citados vuelven a su valor neutro)
PRESETS = {
    "🌡️ Aire muy cálido": {"temp_aire": 3.0, "viento": 0.7},
    "💨 Temporal de viento": {"viento": 2.0, "temp_aire": -1.0},
    "🍃 Semana en calma": {"viento": 0.5},
    "🌍 Clima +1,5 °C": {"calentamiento_base": 1.5},
}


def aplicar_preset(valores: dict, controles: list):
    for d, _, _, _, _, neutro in controles:
        st.session_state[f"v_{d}"] = float(valores.get(d, neutro))
    st.session_state.pop("escenario", None)


def mostrar_escenario(r: dict):
    hs = r["sst"]["horizontes"]
    avisos = {a["variable"] for h in hs for a in h["fuera_de_rango"]}
    if avisos:
        st.warning(f"⚠️ **Fuera de rango:** {len(avisos)} variables del escenario se salen de lo visto en "
                   f"entrenamiento ({', '.join(sorted(avisos)[:6])}{'…' if len(avisos) > 6 else ''}). "
                   "El modelo extrapola: interpreta el resultado con cautela.")
    for d, nota in r["sst"]["notas"].items():
        st.caption(f"**{d}** — {nota}")
    c1, c2 = st.columns(2)
    c1.plotly_chart(G.comparar_escenario(hs), width="stretch")
    c2.plotly_chart(G.prob_ola(hs), width="stretch")
    st.dataframe(pd.DataFrame([{
        "Horizonte": f"{h['h']} d", "Cambio de SST (mediana)": f"{h['diferencia_sst']['p50']:+.2f} °C",
        "Cambio 90 %": f"{h['diferencia_sst']['p05']:+.2f} a {h['diferencia_sst']['p95']:+.2f} °C",
        "P(ola) base → escenario": f"{h['base']['prob_ola_calor']:.0%} → {h['escenario']['prob_ola_calor']:.0%}",
        "Variables fuera de rango": len(h["fuera_de_rango"])} for h in hs]), hide_index=True, width="stretch")
    h7 = next((h for h in hs if h["h"] == 7), hs[0])
    st.plotly_chart(G.barras_grupos(h7["cambio_explicado_por_grupo"],
                                    f"¿Qué explica el cambio a {h7['h']} días? (SHAP escenario − base)"), width="stretch")
    if "hipoxia" in r:
        hx = r["hipoxia"]
        st.subheader(f"Riesgo de hipoxia · campaña {hx['fecha_campana']}")
        aviso_campana_antigua(hx["fecha_campana"], date.fromisoformat(r["sst"]["fecha_base"]))
        e = hx["escenario"]["prob_hipoxia"]
        c1, c2 = st.columns([1, 2])
        c1.metric("Probabilidad en el escenario", f"{e['p50']:.0%}",
                  f"{e['p50'] - hx['base']['prob_hipoxia']:+.0%} vs. base", delta_color="inverse")
        c1.caption(f"Base {hx['base']['prob_hipoxia']:.0%} · escenario 90 %: {e['p05']:.0%} – {e['p95']:.0%}")
        cambio = [{"grupo": x["variable"], "contribucion": x["contribucion"]} for x in hx["cambio_explicado_log_odds"]]
        c2.plotly_chart(G.barras_grupos(cambio, "Qué cambia en el riesgo (log-odds)", "log-odds"), width="stretch")


# ------------------------------------------------------------------ barra lateral
with st.sidebar:
    st.title("🌊 Gemelo digital")
    st.caption("Mar Menor y costa adyacente")
    with st.expander("⚙️ Conexión"):   # antes de las llamadas: si la API falla, se puede corregir la URL
        st.session_state.url = st.text_input("URL de la API", URL_POR_DEFECTO)
    salud = llamar("salud")
    estado_ult = llamar("estado")
    ini, fin = (date.fromisoformat(f) for f in estado_ult["rango_fechas"])
    zona = st.radio("Zona", ["laguna", "costa"], horizontal=True, format_func=str.capitalize)
    fecha = st.date_input("Día base", value=fin, min_value=ini, max_value=fin,
                          help="Por defecto, el último día con datos. Elige un día pasado para comparar con lo que ocurrió.")
    fecha_iso = fecha.isoformat()
    st.divider()
    st.caption(f"API: {salud['estado']} · {salud['modelos_sst']} modelos de SST · datos hasta {salud['ultima_fecha']}")
    if fecha != fin:
        st.info("Día del pasado: verás también lo que pasó realmente.", icon="🕑")

st.header(f"Mar Menor · {fecha:%d/%m/%Y}")
tab_estado, tab_escenarios, tab_xai, tab_hipoxia, tab_visor, tab_modelos = st.tabs(
    ["🌡️ Estado y predicción", "🧪 Escenarios", "🔍 Por qué (XAI)", "🫧 Hipoxia", "🧊 Visor 4D", "⚙️ Modelos"])

# ------------------------------------------------------------------ 1. estado y predicción
with tab_estado:
    est = llamar("estado", fecha_iso)
    z = est["zonas"][zona]
    preds = llamar("predecir", zona, HORIZONTES, fecha_iso)
    p7 = next(p for p in preds if p["h"] == 7)
    nivel, texto = resumen_estado(zona, z, p7)
    aviso_estado(nivel, texto)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Temperatura del agua", f"{z['sst']:.1f} °C", f"{z['anomalia']:+.2f} °C vs. normal",
              delta_color="inverse", border=True)
    c2.metric("Ola de calor marina", "En curso" if z["ola_calor_en_curso"] else "No",
              f"{z['dias_sobre_umbral']} días sobre el umbral", delta_color="off", border=True,
              help="Se declara con 5 días seguidos por encima del percentil 90 de esas fechas (base 1991–2020).")
    c3.metric("Previsión a 7 días", f"{p7['sst_prevista']:.1f} °C",
              f"{p7['sst_prevista'] - z['sst']:+.1f} °C respecto a hoy", delta_color="inverse", border=True,
              help=f"Intervalo del 90 %: {p7['intervalo_90'][0]:.1f} – {p7['intervalo_90'][1]:.1f} °C. "
                   + ("Usa la previsión meteorológica." if p7["usa_prevision_meteo"] else "Sin previsión meteorológica."))
    c4.metric("Prob. de ola de calor a 7 días", f"{p7['prob_superar_umbral']:.0%}",
              f"umbral {p7['umbral_ola_calor']:.1f} °C", delta_color="off", border=True)

    izq, der = st.columns([2, 5])
    with izq:
        st.plotly_chart(G.mapa(est, zona), width="stretch")
        st.caption("🔴 ola de calor · 🟠 más cálida de lo normal · 🔵 más fría · ⚪ normal. "
                   "Puntos: celdas de la meteorología (ERA5).")
    with der:
        desde = (fecha - timedelta(days=120)).isoformat()
        hasta = min(fecha + timedelta(days=35), fin).isoformat()
        serie = llamar("series", zona, desde, hasta)
        st.plotly_chart(G.serie_con_prediccion(serie, preds, zona, fecha), width="stretch")
    tabla = pd.DataFrame([{"Horizonte": f"{p['h']} d", "Fecha": pd.Timestamp(p["fecha_objetivo"]),
                           "SST prevista": p["sst_prevista"],
                           "Intervalo 90 %": f"{p['intervalo_90'][0]:.1f} – {p['intervalo_90'][1]:.1f} °C",
                           "P(ola de calor)": p["prob_superar_umbral"], "Real": p["sst_real"],
                           "Modelo": f"v{p['modelo']['version_registro']} · "
                                     + ("con previsión" if p["usa_prevision_meteo"] else "sin previsión"),
                           "Visto en entrenamiento": "sí" if p["visto_en_entrenamiento"] else "no"} for p in preds])
    st.dataframe(tabla, hide_index=True, width="stretch", column_config={
        "Fecha": st.column_config.DateColumn(format="DD/MM/YYYY"),
        "SST prevista": st.column_config.NumberColumn(format="%.1f °C"),
        "P(ola de calor)": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
        "Real": st.column_config.NumberColumn(format="%.1f °C", help="Lo que ocurrió (solo en días del pasado)")})
    if any(p["visto_en_entrenamiento"] for p in preds):
        st.warning("Este día cae en el periodo de entrenamiento del modelo final (hasta 2021): comparar con lo real "
                   "no es una prueba honesta de su capacidad.")

# ------------------------------------------------------------------ 2. escenarios
with tab_escenarios:
    st.markdown("Modifica el estado del mar y compara con lo que predice el gemelo sin cambios. "
                "Cada cambio admite una **incertidumbre** (desviación típica) que se simula con Monte Carlo.")
    info = llamar("drivers")
    drv = info["drivers"]
    CONTROLES = [  # driver, etiqueta, mín, máx, paso, neutro
        ("temp_agua", "Episodio de agua más cálida (°C)", -3.0, 3.0, 0.1, 0.0),
        ("calentamiento_base", "Clima de fondo más cálido (°C)", 0.0, 3.0, 0.1, 0.0),
        ("temp_aire", "Aire más cálido (°C)", -4.0, 4.0, 0.5, 0.0),
        ("lluvia", "Lluvia (× veces)", 0.0, 5.0, 0.25, 1.0),
        ("viento", "Viento (× veces)", 0.0, 2.0, 0.1, 1.0),
        ("radiacion", "Radiación solar (× veces)", 0.5, 1.5, 0.05, 1.0),
        ("clorofila", "Clorofila (× veces)", 0.25, 4.0, 0.25, 1.0),
    ]
    st.caption("Escenarios de un clic (después pulsa **Simular escenario**):")
    botones = st.columns(len(PRESETS) + 1)
    for b, (nombre, vals) in zip(botones, PRESETS.items()):
        b.button(nombre, on_click=aplicar_preset, args=(vals, CONTROLES), width="stretch")
    botones[-1].button("↺ Reiniciar", on_click=aplicar_preset, args=({}, CONTROLES), width="stretch")
    # Sin st.form a propósito: así los campos de incertidumbre aparecen al mover un control (en un
    # formulario no se actualizarían hasta enviarlo). La simulación solo se lanza con el botón.
    col_a, col_b = st.columns(2)
    valores = {}
    for i, (d, etiqueta, lo, hi, paso, neutro) in enumerate(CONTROLES):
        with (col_a if i % 2 == 0 else col_b):
            valores[d] = st.slider(etiqueta, lo, hi, neutro, paso, help=drv[d]["descripcion"] + " " + drv[d]["nota"],
                                   key=f"v_{d}")
    activos = [c for c in CONTROLES if valores[c[0]] != c[5]]
    perts = []
    if activos:
        with st.expander("Incertidumbre y duración de cada cambio", expanded=True):
            for d, etiqueta, lo, hi, paso, neutro in activos:
                c0, c1, c2, c3 = st.columns([2, 1, 1, 1])
                c0.markdown(f"**{etiqueta}**: {valores[d]:g}")
                inc = c1.number_input("± incertidumbre", 0.0, abs(hi - lo) / 2, 0.0, paso, key=f"i_{d}")
                periodos = drv[d].get("periodos", ["pasado"])
                periodo = "pasado"
                if len(periodos) > 1:
                    periodo = c3.selectbox("cuándo", periodos, index=periodos.index("futuro"), key=f"p_{d}",
                                           format_func={"pasado": "días pasados", "futuro": "próximos días (previsión)",
                                                        "ambos": "pasados y próximos"}.get,
                                           help="'Próximos días' modifica la previsión meteorológica (máx. 7 días) "
                                                "que usan los modelos de 1, 3 y 7 días.")
                if d == "calentamiento_base":
                    dias = 30
                elif periodo == "futuro":
                    dias = c2.number_input("próximos días", 1, 7, 7, key=f"df_{d}")
                else:
                    dias = c2.number_input("últimos días", 1, 365, 30, key=f"d_{d}")
                perts.append({"driver": d, "valor": valores[d], "incertidumbre": inc, "dias": int(dias),
                              "periodo": periodo})
    con_hipoxia = st.checkbox("Simular también el riesgo de hipoxia (última campaña anterior al día base)", True)
    with st.expander("¿Qué no se puede simular?"):
        for k, v in info["no_disponibles"].items():
            st.markdown(f"- **{k}**: {v}")

    if st.button("Simular escenario", type="primary", disabled=not perts,
                 help=None if perts else "Mueve al menos un control"):
        campana = llamar("estado", fecha_iso)["ultima_campana_hipoxia"]
        peticion = {"zona": zona, "horizontes": HORIZONTES, "fecha": fecha_iso, "perturbaciones": perts}
        if con_hipoxia and campana:
            peticion["hipoxia"] = {"fecha_campana": campana["fecha"]}
        with st.spinner("Simulando (Monte Carlo)…"):
            st.session_state.escenario = {"peticion": peticion, "respuesta": llamar("escenario", **peticion)}

    res = st.session_state.get("escenario")
    if res and res["peticion"]["zona"] == zona and res["peticion"]["fecha"] == fecha_iso:
        r = res["respuesta"]
        if res["peticion"]["perturbaciones"] != perts:
            st.caption("ℹ️ Resultado del último escenario simulado; has cambiado controles desde entonces.")
        mostrar_escenario(r)

# ------------------------------------------------------------------ 3. XAI
with tab_xai:
    h = st.select_slider("Horizonte a explicar", HORIZONTES, value=7, format_func=lambda x: f"{x} días")
    e = llamar("explicar", zona, h, fecha_iso, 12)
    st.markdown(f"El modelo predice que, dentro de {h} días, la anomalía de temperatura habrá cambiado "
                f"**{e['prediccion']:+.2f} °C** respecto a la de hoy. La cascada muestra cómo se llega a esa cifra: "
                "🔴 lo que empuja a calentar, 🔵 lo que empuja a enfriar.")
    c1, c2 = st.columns([3, 2])
    c1.plotly_chart(G.cascada(e["grupos"], e["valor_base"], e["prediccion"]), width="stretch")
    c2.plotly_chart(G.barras_variables(e["variables"], "Las 12 variables que más pesan"), width="stretch")
    st.caption(e["nota"])
    with st.expander("¿Cómo leer esto?"):
        st.markdown(
            "- **Temperatura propia** suele dominar: es la *reversión a la media* (una anomalía alta tiende a bajar).\n"
            "- A **1–3 días** pesan más el aire, la radiación y el viento; a **30 días**, la laguna 'mira' a la costa.\n"
            "- SHAP explica **lo que usa el modelo**, no causas físicas: una variable puede pesar por correlación.")

# ------------------------------------------------------------------ 4. hipoxia
with tab_hipoxia:
    camp = llamar("campanas")
    previas = camp.index[camp.index <= pd.Timestamp(fecha)]
    if not len(previas):
        st.info("No hay campañas de muestreo anteriores a este día (empiezan en 2017).")
    else:
        sel = st.selectbox("Campaña", list(previas[::-1].strftime("%Y-%m-%d")), index=0,
                           help="Por defecto, la última campaña anterior al día base.")
        hx = llamar("hipoxia", sel)
        aviso_campana_antigua(sel, fecha)
        c1, c2, c3 = st.columns([2, 1, 1])
        c1.plotly_chart(G.aguja(hx["prob_hipoxia_siguiente"], titulo="Riesgo de hipoxia en la próxima campaña"),
                        width="stretch")
        c2.metric("Alerta temprana", "Sí" if hx["alerta_temprana"] else "No", border=True,
                  help="Se activa cuando la probabilidad supera el 50 %.")
        c3.metric("Fondo en hipoxia ahora", f"{hx['pct_fondo_hipoxia_actual']:.1f} %",
                  "episodio en curso" if hx["hipoxia_actual"] else "sin episodio", delta_color="off", border=True,
                  help="Porcentaje del fondo de la laguna con menos de 4 mg/L de oxígeno en esta campaña.")
        if hx["hipoxia_real_siguiente"] is not None:
            st.caption(f"Lo que pasó en la siguiente campaña: {'hipoxia' if hx['hipoxia_real_siguiente'] else 'sin hipoxia'}"
                       + (" (campaña usada para entrenar: no es una prueba honesta)" if hx["visto_en_entrenamiento"] else ""))
        st.plotly_chart(G.historial_hipoxia(camp, 10, sel), width="stretch")
        st.plotly_chart(G.barras_variables(hx["contribuciones_log_odds"][:10], "Por qué (contribución al riesgo, log-odds)",
                                           "log-odds"), width="stretch")
        st.caption(hx["nota"])

# ------------------------------------------------------------------ 5. visor 4D
with tab_visor:
    url_visor = f"{URL_PUBLICA.rstrip('/')}/visor/"
    st.caption("Oxígeno, temperatura, salinidad, clorofila y turbidez de las campañas in situ de la UPCT (2017–2024), "
               f"en 3D y a lo largo del tiempo. [Abrir a pantalla completa]({url_visor})")
    st.iframe(url_visor, height=720)

# ------------------------------------------------------------------ 6. modelos
with tab_modelos:
    m = llamar("modelos")
    st.markdown(f"Modelos exportados el **{m['exportado'][:10]}** desde el commit `{m['git_commit']}` "
                "(MLflow Model Registry, alias `@produccion`).")
    filas = []
    for k, v in m["sst"].items():
        me, mr = v["metricas"], (v["respaldo"] or {}).get("metricas", {})
        filas.append({
            "Modelo": k, "Zona": v["zona"], "Horizonte (d)": v["h"], "Versión": v["version_registro"],
            "Previsión meteo": "sí" if v["usa_prevision_meteo"] else "no",
            "Respaldo": f"v{v['respaldo']['version_registro']}" if v["respaldo"] else "—",
            "Datos hasta": v.get("datos_hasta"),
            # métrica de la evaluación con la que se promocionó cada versión
            "Receta": ("física + pesos" + (" + lineal" if v.get("con_parte_lineal") else "")
                       if v.get("receta") == "fisica_pesos" else "clásica"),
            "RMSE (°C)": me.get("rmse_nueva", me.get("rmse_retador", me.get("rmse_B_real", me.get("test_rmse")))),
            "Mejora": (f"{me['mejora']:+.1%} vs receta clásica (fuera de muestra)" if "rmse_nueva" in me
                       else f"{me['mejora']:+.1%} vs versión anterior (reentreno)" if "mejora" in me
                       else f"{me['skill_Breal_vs_A']:+.1%} vs sin previsión" if "skill_Breal_vs_A" in me
                       else f"{me.get('test_skill_vs_amortiguada', 0):+.1%} vs pers. amortiguada"),
        })
    st.dataframe(pd.DataFrame(filas).sort_values(["Zona", "Horizonte (d)"]), hide_index=True, width="stretch")
    st.caption("RMSE del cambio de anomalía en la evaluación con la que se promocionó cada versión: receta nueva, modelos "
               "hasta 2021 evaluados con previsiones reales de 2024–2026 (30 días: 2022–2026); reentrenos, "
               "últimos 180 días frente a la versión anterior; con previsión, ventana honesta ago-2024 → sep-2026; "
               "sin previsión, test 2022–2026. El respaldo se usa los días sin previsión archivada (antes de feb-2024).")
    st.subheader("Salud de los modelos (monitor de deriva)")
    try:
        mon = leer(st.session_state.url, "monitor")
    except ErrorAPI:
        mon = None
    if mon is None:
        st.info("Aún no hay informe del monitor (`python scripts/monitorizar.py`).")
    else:
        icono = {"OK": "🟢", "AVISO": "🟡", "DERIVA": "🔴", "SIN_DATOS": "⚪"}
        st.markdown(f"Control del **{mon['fecha']}** · estado global {icono.get(mon['estado_global'], '')} "
                    f"**{mon['estado_global']}**")
        st.dataframe(pd.DataFrame([{
            "Modelo": n, "Estado": f"{icono[x['estado']]} {x['estado']}", "Versión": f"v{x['version']}",
            "Datos hasta": x["datos_hasta"],
            "Skill vs persistencia": (f"{x['rendimiento']['skill']:+.0%}" if x["rendimiento"].get("skill") is not None
                                      else "aún no evaluable"),
            "Cobertura 90 %": (f"{x['rendimiento']['cobertura']:.0%}" if x["rendimiento"].get("cobertura") is not None
                               else "—"),
            "PSI máx / umbral": f"{x['entradas']['psi_max']:.1f} / {x['entradas']['psi_umbral']:.1f}",
            "Fuera de rango": f"{x['entradas']['fuera_rango']:.0%}",
            "Motivos": "; ".join(x["motivos"])} for n, x in mon["modelos"].items()]), hide_index=True, width="stretch")
        for etiqueta, clave in [("Previsión meteorológica", "prevision"), ("Hipoxia", "hipoxia")]:
            e = mon[clave]
            st.caption(f"{icono.get(e['estado'], '')} **{etiqueta}**: {e['estado']} " + " ".join(e.get("motivos", [])))
        if mon.get("reentrenar"):
            st.caption(f"Propuestos para reentrenar: {', '.join(mon['reentrenar'])}. En espera tras un reentreno "
                       f"reciente: {', '.join(mon.get('en_espera_tras_reentreno', [])) or 'ninguno'}.")
        st.caption("Skill 'aún no evaluable' = modelo reentrenado con los últimos datos: hasta que lleguen datos nuevos "
                   "no hay días fuera de muestra con los que medirlo honestamente.")

    hm = m["hipoxia"]
    st.markdown(f"**Hipoxia:** logística v{hm['version_registro']} · validación por años: recall "
                f"{hm['metricas_loyo'].get('logistica_recall', 0):.2f}, precisión {hm['metricas_loyo'].get('logistica_precision', 0):.2f}, "
                f"inicios de episodio anticipados {int(hm['metricas_loyo'].get('logistica_inicios_anticipados', 0))}/9.")
