"""Pipeline reproducible data/raw -> data/interim -> data/processed.

Uso:
    python scripts/construir_dataset.py

Además de los Parquet, escribe data/processed/manifest.json: qué datos, qué config y qué
commit generaron el dataset. En la Fase 3 se adjunta a cada experimento de MLflow (trazabilidad).
"""
import hashlib
import json
import subprocess
import time
from datetime import datetime, timezone

from gemelo.config import DATA_RAW, RAIZ, RUTA_CONFIG, cargar_config
from gemelo.procesado.features import (construir_dataset_hipoxia, construir_dataset_sst,
                                       grupo_variable, referencias_sst)
from gemelo.procesado import meteo_futura
from gemelo.procesado.cubo import construir_cubo, contorno_laguna, resumen_por_campana
from gemelo.procesado.insitu import campanas_insitu
from gemelo.procesado.series import construir_diario

INTERIM = RAIZ / "data" / "interim"
PROCESSED = RAIZ / "data" / "processed"


def paso(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def describir_dataset(df):
    x = [c for c in df.columns if not c.startswith("y_") and c not in ("particion", "anio")]
    return {
        "filas": len(df), "desde": str(df.index.min().date()), "hasta": str(df.index.max().date()),
        "variables": {c: grupo_variable(c) for c in x},
        "objetivos": [c for c in df.columns if c.startswith("y_")],
    }


def manifiesto(cfg, datasets):
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=RAIZ,
                                capture_output=True, text=True).stdout.strip()
        sucio = bool(subprocess.run(["git", "status", "--porcelain"], cwd=RAIZ,
                                    capture_output=True, text=True).stdout.strip())
    except OSError:
        commit, sucio = None, None
    fuentes = {
        str(p.relative_to(DATA_RAW)).replace("\\", "/"): {
            "mb": round(p.stat().st_size / 1e6, 2),
            "modificado": datetime.fromtimestamp(p.stat().st_mtime, timezone.utc).isoformat(timespec="seconds"),
        }
        for p in sorted(DATA_RAW.rglob("*")) if p.is_file() and p.name != ".gitkeep"
    }
    return {
        "creado": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": commit, "git_cambios_sin_commit": sucio,
        "config_sha256": hashlib.sha256(RUTA_CONFIG.read_bytes()).hexdigest(),
        "particion": cfg["particion"], "horizontes_dias": cfg["horizontes_dias"],
        "fuentes_raw": fuentes,
        "datasets": {k: describir_dataset(v) for k, v in datasets.items()},
    }


if __name__ == "__main__":
    cfg = cargar_config()
    INTERIM.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)

    paso("1/5 series diarias por subzona")
    diario, informe = construir_diario(cfg)
    diario.to_parquet(INTERIM / "diario.parquet")
    rell = {k: v["rellenados"] for k, v in informe.items() if v["rellenados"]}
    paso(f"    {diario.shape} · huecos cortos rellenados: {rell or 'ninguno'}")

    paso("2/5 campañas in situ (UPCT)")
    campanas = campanas_insitu(cfg)
    campanas.to_parquet(INTERIM / "insitu_campanas.parquet")
    paso(f"    {campanas.shape} · NaN por variable: "
         f"{campanas.isna().sum()[lambda s: s > 0].to_dict() or 'ninguno'}")

    paso("3/5 dataset SST / olas de calor")
    # Climatología y umbrales por día del año: los usa también la API (misma fuente, sin divergencias)
    ref = referencias_sst(diario, cfg)
    ref.to_parquet(PROCESSED / "referencias_sst.parquet")
    sst = construir_dataset_sst(diario, cfg, ref)
    sst.to_parquet(PROCESSED / "dataset_sst.parquet")
    # Series diarias de origen: el motor de escenarios las perturba y recalcula las variables
    diario.to_parquet(PROCESSED / "series_diarias.parquet")
    n_x = sum(not c.startswith("y_") and c != "particion" for c in sst.columns)
    paso(f"    {sst.shape} · {n_x} variables · filas por partición: {sst.particion.value_counts().to_dict()}")

    paso("4/5 dataset hipoxia")
    hip = construir_dataset_hipoxia(campanas, sst, cfg)
    hip.to_parquet(PROCESSED / "dataset_hipoxia.parquet")
    paso(f"    {hip.shape} · eventos (y_hipoxia=1): {int(hip.y_hipoxia.sum())} · "
         f"por año: {hip.groupby('anio').y_hipoxia.sum().astype(int).to_dict()}")

    paso("5/6 meteorología futura (ERA5 y previsión real corregida)")
    f_era5, f_prev, prev, sesgos = meteo_futura.preparar(cfg, diario, ref, sst)
    f_era5.to_parquet(PROCESSED / "meteo_futura_era5.parquet")
    if f_prev is None:
        paso("    sin previsiones descargadas (scripts/descargar_previsiones.py): solo ERA5")
    else:
        f_prev.to_parquet(PROCESSED / "meteo_futura_prevision.parquet")
        meteo_futura.a_tabla(prev).to_parquet(PROCESSED / "prevision_diaria.parquet")  # para escenarios
        (PROCESSED / "sesgos_prevision.json").write_text(
            json.dumps({pt: s.round(4).to_dict() for pt, s in sesgos.items()}, indent=1), encoding="utf-8")
        completos = f_prev[[c for c in f_prev.columns if c.startswith("prev7_")]].notna().all(axis=1)
        paso(f"    previsión real completa (7 d) para {int(completos.sum())} días base "
             f"({completos[completos].index.min().date()} → {completos[completos].index.max().date()})")

    paso("6/7 cubo 4D in situ (visor CesiumJS)")
    cubo = construir_cubo(cfg)
    cubo.to_parquet(PROCESSED / "cubo_insitu.parquet")
    resumen = resumen_por_campana(cubo, cfg["umbrales"]["hipoxia_mg_l"], cfg["umbrales"]["anoxia_mg_l"])
    resumen.to_parquet(PROCESSED / "cubo_resumen.parquet")
    (PROCESSED / "contorno_laguna.json").write_text(json.dumps(contorno_laguna(cubo)), encoding="utf-8")  # mapa
    paso(f"    {len(cubo):,} filas · {cubo.fecha.nunique()} campañas · "
         f"{len(cubo) / cubo.fecha.nunique():,.0f} puntos 3D por campaña · máx. volumen en hipoxia "
         f"{resumen.pct_volumen_hipoxia.max():.1f} % ({resumen.pct_volumen_hipoxia.idxmax().date()})")

    paso("7/7 manifiesto")
    man = manifiesto(cfg, {"dataset_sst": sst, "dataset_hipoxia": hip})
    (PROCESSED / "manifest.json").write_text(json.dumps(man, indent=2, ensure_ascii=False), encoding="utf-8")
    grupos = {}
    for d in man["datasets"].values():
        for g in d["variables"].values():
            grupos[g] = grupos.get(g, 0) + 1
    paso(f"    commit {man['git_commit']} (cambios sin commit: {man['git_cambios_sin_commit']}) · "
         f"variables por grupo: {grupos}")
