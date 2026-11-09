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
                                       grupo_variable)
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
    sst = construir_dataset_sst(diario, cfg)
    sst.to_parquet(PROCESSED / "dataset_sst.parquet")
    n_x = sum(not c.startswith("y_") and c != "particion" for c in sst.columns)
    paso(f"    {sst.shape} · {n_x} variables · filas por partición: {sst.particion.value_counts().to_dict()}")

    paso("4/5 dataset hipoxia")
    hip = construir_dataset_hipoxia(campanas, sst, cfg)
    hip.to_parquet(PROCESSED / "dataset_hipoxia.parquet")
    paso(f"    {hip.shape} · eventos (y_hipoxia=1): {int(hip.y_hipoxia.sum())} · "
         f"por año: {hip.groupby('anio').y_hipoxia.sum().astype(int).to_dict()}")

    paso("5/5 manifiesto")
    man = manifiesto(cfg, {"dataset_sst": sst, "dataset_hipoxia": hip})
    (PROCESSED / "manifest.json").write_text(json.dumps(man, indent=2, ensure_ascii=False), encoding="utf-8")
    grupos = {}
    for d in man["datasets"].values():
        for g in d["variables"].values():
            grupos[g] = grupos.get(g, 0) + 1
    paso(f"    commit {man['git_commit']} (cambios sin commit: {man['git_cambios_sin_commit']}) · "
         f"variables por grupo: {grupos}")
