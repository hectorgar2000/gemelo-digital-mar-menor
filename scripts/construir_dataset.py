"""Pipeline reproducible data/raw -> data/interim -> data/processed.

Uso:
    python scripts/construir_dataset.py
"""
import time

import pandas as pd

from gemelo.config import RAIZ, cargar_config
from gemelo.procesado.features import construir_dataset_hipoxia, construir_dataset_sst
from gemelo.procesado.oxigeno import campanas_oxigeno
from gemelo.procesado.series import construir_diario

INTERIM = RAIZ / "data" / "interim"
PROCESSED = RAIZ / "data" / "processed"


def paso(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


if __name__ == "__main__":
    cfg = cargar_config()
    INTERIM.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)

    paso("1/4 series diarias por subzona")
    diario, informe = construir_diario(cfg)
    diario.to_parquet(INTERIM / "diario.parquet")
    rell = {k: v["rellenados"] for k, v in informe.items() if v["rellenados"]}
    paso(f"    {diario.shape} · huecos cortos rellenados: {rell or 'ninguno'}")

    paso("2/4 campañas de oxígeno")
    campanas = campanas_oxigeno(cfg)
    campanas.to_parquet(INTERIM / "oxigeno_campanas.parquet")
    paso(f"    {len(campanas)} campañas")

    paso("3/4 dataset SST / olas de calor")
    sst = construir_dataset_sst(diario, cfg)
    sst.to_parquet(PROCESSED / "dataset_sst.parquet")
    n_x = sum(not c.startswith("y_") and c != "particion" for c in sst.columns)
    paso(f"    {sst.shape} · {n_x} variables · filas por partición: {sst.particion.value_counts().to_dict()}")

    paso("4/4 dataset hipoxia")
    hip = construir_dataset_hipoxia(campanas, sst, cfg)
    hip.to_parquet(PROCESSED / "dataset_hipoxia.parquet")
    paso(f"    {hip.shape} · eventos (y_hipoxia=1): {int(hip.y_hipoxia.sum())} · "
         f"por año: {hip.groupby('anio').y_hipoxia.sum().astype(int).to_dict()}")

    # Resumen de objetivos MHW por partición
    resumen = sst.groupby("particion")[[f"y_mhw_{z}_h7" for z in ["laguna", "costa"]]].mean().mul(100).round(1)
    paso("    % días en ola de calor (objetivo h=7) por partición:\n" + resumen.to_string())
