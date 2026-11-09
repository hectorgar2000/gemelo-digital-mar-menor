"""Descarga de productos de Copernicus Marine para la zona de config/zona.yaml.

Las credenciales NO se gestionan aquí: el usuario hace `copernicusmarine login` una vez
y la librería las lee de %USERPROFILE%\\.copernicusmarine.
"""
from pathlib import Path

import copernicusmarine as cm

from gemelo.config import DATA_RAW, cargar_config


def descargar(clave: str, cfg: dict, destino: Path, dry_run: bool = False, sobrescribir: bool = False):
    """Descarga un dataset (identificado por su clave en config) a NetCDF.

    Con dry_run=True no descarga nada: solo devuelve el tamaño estimado y el recorte final.
    """
    ds_cfg = cfg["copernicus"][clave]
    bbox = cfg["zona"]["bbox"]
    salida = destino / f"{clave}.nc"
    if salida.exists() and not sobrescribir and not dry_run:
        print(f"[{clave}] ya existe {salida.name}, se omite")
        return None

    extra = {}
    if "profundidad_max" in ds_cfg:
        extra = dict(minimum_depth=0, maximum_depth=ds_cfg["profundidad_max"])

    return cm.subset(
        dataset_id=ds_cfg["dataset_id"],
        variables=ds_cfg["variables"],
        minimum_latitude=bbox["lat_min"],
        maximum_latitude=bbox["lat_max"],
        minimum_longitude=bbox["lon_min"],
        maximum_longitude=bbox["lon_max"],
        start_datetime=ds_cfg.get("inicio", cfg["periodo"]["inicio"]),
        end_datetime=cfg["periodo"]["fin"],
        output_directory=destino,
        output_filename=salida.name,
        overwrite=sobrescribir,
        dry_run=dry_run,
        disable_progress_bar=True,
        **extra,
    )


def descargar_todo(claves=None, dry_run: bool = False):
    cfg = cargar_config()
    destino = DATA_RAW / "copernicus"
    destino.mkdir(parents=True, exist_ok=True)
    claves = claves or list(cfg["copernicus"])
    for clave in claves:
        r = descargar(clave, cfg, destino, dry_run=dry_run)
        if r is None:
            continue
        if dry_run:
            print(f"[{clave}] tamaño estimado: {r.file_size:.1f} MB")
            for c in r.coordinates_extent:
                print(f"    {c.coordinate_id}: {c.minimum} -> {c.maximum}")
        else:
            print(f"[{clave}] descargado: {r.file_path} ({r.file_size:.1f} MB)")
