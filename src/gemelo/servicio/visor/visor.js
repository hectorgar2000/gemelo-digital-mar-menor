// Visor 4D del Mar Menor: cubos in situ (lon, lat, profundidad, tiempo) servidos por la API (/cubo).
// Cada punto es un bloque de agua; las capas se apilan con la superficie ARRIBA y exageración vertical.
"use strict";

const API = window.location.origin;           // la API sirve también esta página (mismo origen)
const PROF_MAX = 5.0;                         // profundidad máxima de las campañas (m)
const HIPOXIA = 4.0;                          // mg/L

// Rampas de color: [posición 0-1, color]. Oxígeno en valores absolutos (rojo = hipoxia).
const RAMPAS = {
  o2: { absoluta: [0, 12], paradas: [[0, "#67001f"], [1 / 6, "#d6604d"], [2 / 6, "#f4a582"], [0.5, "#fddbc7"], [4 / 6, "#92c5de"], [5 / 6, "#4393c3"], [1, "#053061"]] },
  temp: { paradas: [[0, "#313695"], [0.25, "#74add1"], [0.5, "#ffffbf"], [0.75, "#f46d43"], [1, "#a50026"]] },
  defecto: { paradas: [[0, "#440154"], [0.25, "#3b528b"], [0.5, "#21918c"], [0.75, "#5ec962"], [1, "#fde725"]] },
};
const NOMBRES = { o2: "Oxígeno disuelto", temp: "Temperatura", sal: "Salinidad", chl: "Clorofila", turb: "Turbidez" };

const estado = { campanas: [], variables: {}, idx: 0, variable: "o2", exag: 600, cache: new Map(), reproduciendo: null, datos: null };
const $ = (id) => document.getElementById(id);

// ------------------------------------------------------------------ Cesium (sin ion)
Cesium.Ion.defaultAccessToken = "";
const visor = new Cesium.Viewer("mapa", {
  baseLayer: new Cesium.ImageryLayer(new Cesium.OpenStreetMapImageryProvider({ url: "https://tile.openstreetmap.org/" })),
  baseLayerPicker: false, geocoder: false, timeline: false, animation: false, sceneModePicker: false,
  navigationHelpButton: false, fullscreenButton: false, infoBox: false, selectionIndicator: false,
});
visor.scene.globe.depthTestAgainstTerrain = false;
const puntos = visor.scene.primitives.add(new Cesium.PointPrimitiveCollection());

function encuadrar() {
  // vista oblicua desde el sur: así se ven las capas apiladas (desde arriba se taparían unas a otras)
  visor.camera.flyTo({
    destination: Cesium.Cartesian3.fromDegrees(-0.74, 37.53, 7000),
    orientation: { heading: Cesium.Math.toRadians(-25), pitch: Cesium.Math.toRadians(-20), roll: 0 }, duration: 1.2,
  });
}
visor.homeButton.viewModel.command.beforeExecute.addEventListener((e) => { e.cancel = true; encuadrar(); });

// ------------------------------------------------------------------ colores
function hexARgb(h) { return [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16) / 255); }

function colorDe(variable, valor) {
  const rampa = RAMPAS[variable] || RAMPAS.defecto;
  const [lo, hi] = rampa.absoluta || estado.variables[variable].rango_color;
  const t = Math.min(1, Math.max(0, (valor - lo) / (hi - lo)));
  const p = rampa.paradas;
  for (let i = 1; i < p.length; i++) {
    if (t <= p[i][0]) {
      const f = (t - p[i - 1][0]) / (p[i][0] - p[i - 1][0]);
      const a = hexARgb(p[i - 1][1]), b = hexARgb(p[i][1]);
      return new Cesium.Color(a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1]), a[2] + f * (b[2] - a[2]), 0.92);
    }
  }
  return Cesium.Color.fromCssColorString(p[p.length - 1][1]);
}

function dibujarLeyenda() {
  const v = estado.variable;
  const rampa = RAMPAS[v] || RAMPAS.defecto;
  const [lo, hi] = rampa.absoluta || estado.variables[v].rango_color;
  $("rampa").style.background = `linear-gradient(to right, ${rampa.paradas.map(([t, c]) => `${c} ${t * 100}%`).join(", ")})`;
  const u = estado.variables[v].unidad;
  $("etiquetas").innerHTML = `<span>${lo} ${u}</span><span>${NOMBRES[v]}</span><span>${hi}${rampa.absoluta ? "+" : ""} ${u}</span>`;
  $("soloHipoxia").disabled = v !== "o2";
}

// ------------------------------------------------------------------ datos
async function pedir(url) {
  const r = await fetch(API + url);
  if (!r.ok) throw new Error(`${r.status}: ${(await r.json().catch(() => ({}))).detail || r.statusText}`);
  return r.json();
}

async function datosDe(idx, variable) {
  const fecha = estado.campanas[idx].fecha;
  const clave = `${fecha}|${variable}`;
  if (!estado.cache.has(clave)) estado.cache.set(clave, pedir(`/cubo?fecha=${fecha}&variable=${variable}`));
  return estado.cache.get(clave);
}

// ------------------------------------------------------------------ pintar
function pintar() {
  const d = estado.datos;
  if (!d) return;
  puntos.removeAll();
  const solo = $("soloHipoxia").checked && estado.variable === "o2";
  const pmin = parseFloat($("profMin").value), pmax = parseFloat($("profMax").value);
  let visibles = 0;
  for (let i = 0; i < d.n; i++) {
    const prof = d.profundidad[i], valor = d.valor[i];
    if (prof < pmin || prof > pmax || (solo && valor >= HIPOXIA)) continue;
    // superficie arriba: 0,5 m es la capa más alta y 5 m la más baja
    const altura = (PROF_MAX + 0.5 - prof) * estado.exag;
    puntos.add({ position: Cesium.Cartesian3.fromDegrees(d.lon[i], d.lat[i], altura), color: colorDe(estado.variable, valor), pixelSize: 6 });
    visibles++;
  }
  const r = d.resumen;
  $("info").innerHTML =
    `<b>${d.fecha}</b> · ${visibles.toLocaleString("es")} de ${d.n.toLocaleString("es")} bloques visibles<br>` +
    `Volumen con O₂ &lt; 4 mg/L: <b>${r.pct_volumen_hipoxia.toFixed(1)} %</b> · &lt; 2 mg/L: <b>${r.pct_volumen_anoxia.toFixed(1)} %</b><br>` +
    `O₂ medio <b>${r.o2_medio.toFixed(2)} mg/L</b> · temperatura media <b>${r.temp_media.toFixed(1)} °C</b>`;
  visor.scene.requestRender();
}

async function irA(idx) {
  estado.idx = Math.max(0, Math.min(estado.campanas.length - 1, idx));
  $("tiempo").value = estado.idx;
  $("fecha").textContent = estado.campanas[estado.idx].fecha;
  dibujarSerie();
  try {
    $("error").textContent = "";
    estado.datos = await datosDe(estado.idx, estado.variable);
    pintar();
    // precarga la siguiente campaña para que la reproducción sea fluida
    if (estado.idx + 1 < estado.campanas.length) datosDe(estado.idx + 1, estado.variable).catch(() => {});
  } catch (e) {
    $("error").textContent = `No se pudo cargar la campaña: ${e.message}`;
  }
}

// ------------------------------------------------------------------ serie temporal (contexto 4D)
function dibujarSerie() {
  const c = $("serie"), ctx = c.getContext("2d");
  const W = c.width = c.clientWidth * devicePixelRatio, H = c.height = c.clientHeight * devicePixelRatio;
  const v = estado.campanas.map((x) => x.pct_volumen_hipoxia || 0);
  const max = Math.max(10, ...v);
  ctx.clearRect(0, 0, W, H);
  ctx.strokeStyle = "#d6604d"; ctx.lineWidth = 1.5 * devicePixelRatio; ctx.beginPath();
  v.forEach((y, i) => { const px = (i / (v.length - 1)) * W, py = H - (y / max) * (H - 6) - 3; i ? ctx.lineTo(px, py) : ctx.moveTo(px, py); });
  ctx.stroke();
  const x = (estado.idx / (v.length - 1)) * W;
  ctx.strokeStyle = "#ffffff"; ctx.lineWidth = 1 * devicePixelRatio; ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke();
  ctx.fillStyle = "#a9b0bb"; ctx.font = `${10 * devicePixelRatio}px system-ui`;
  ctx.fillText(`${max.toFixed(0)} %`, 4, 11 * devicePixelRatio);
  ctx.fillText(estado.campanas[0].fecha.slice(0, 4), 4, H - 4);
  ctx.fillText(estado.campanas.at(-1).fecha.slice(0, 4), W - 30 * devicePixelRatio, H - 4);
}

// ------------------------------------------------------------------ controles
function reproducir() {
  if (estado.reproduciendo) { clearInterval(estado.reproduciendo); estado.reproduciendo = null; $("play").textContent = "▶"; return; }
  $("play").textContent = "⏸";
  estado.reproduciendo = setInterval(() => irA(estado.idx + 1 < estado.campanas.length ? estado.idx + 1 : 0), 1200);
}

async function iniciar() {
  try {
    const meta = await pedir("/cubo/fechas");
    estado.campanas = meta.campanas;
    estado.variables = meta.variables;
    $("variable").innerHTML = Object.entries(meta.variables)
      .map(([k, v]) => `<option value="${k}">${NOMBRES[k] || k} (${v.unidad})</option>`).join("");
    const opciones = meta.profundidades.map((p) => `<option value="${p}">${p.toLocaleString("es")}</option>`).join("");
    $("profMin").innerHTML = opciones; $("profMax").innerHTML = opciones;
    $("profMin").value = meta.profundidades[0]; $("profMax").value = meta.profundidades.at(-1);
    $("tiempo").max = estado.campanas.length - 1;
    dibujarLeyenda();
    // empezar en la campaña con más volumen en hipoxia: lo más interesante que enseñar
    const peor = estado.campanas.reduce((m, x, i, a) => (x.pct_volumen_hipoxia > a[m].pct_volumen_hipoxia ? i : m), 0);
    encuadrar();
    await irA(peor);
  } catch (e) {
    $("error").textContent = `No se pudo conectar con la API (${e.message}). ¿Está generado el cubo? python scripts/construir_dataset.py`;
  }
}

$("variable").addEventListener("change", (e) => { estado.variable = e.target.value; dibujarLeyenda(); irA(estado.idx); });
$("tiempo").addEventListener("input", (e) => irA(parseInt(e.target.value, 10)));
$("anterior").addEventListener("click", () => irA(estado.idx - 1));
$("siguiente").addEventListener("click", () => irA(estado.idx + 1));
$("play").addEventListener("click", reproducir);
$("soloHipoxia").addEventListener("change", pintar);
$("profMin").addEventListener("change", pintar);
$("profMax").addEventListener("change", pintar);
$("exag").addEventListener("input", (e) => { estado.exag = parseInt(e.target.value, 10); $("exagValor").textContent = estado.exag; pintar(); });
$("serie").addEventListener("click", (e) => {
  const r = e.target.getBoundingClientRect();
  irA(Math.round(((e.clientX - r.left) / r.width) * (estado.campanas.length - 1)));
});
iniciar();
