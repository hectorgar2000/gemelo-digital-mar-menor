"""Descarga los productos de Copernicus Marine definidos en config/zona.yaml.

Uso:
    python scripts/descargar_copernicus.py --dry-run        # solo estima tamaños
    python scripts/descargar_copernicus.py                  # descarga todo
    python scripts/descargar_copernicus.py sst clorofila    # solo algunos
"""
import argparse

from gemelo.ingesta.copernicus import descargar_todo

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("claves", nargs="*", help="claves de config/zona.yaml > copernicus (vacío = todas)")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    descargar_todo(a.claves or None, dry_run=a.dry_run)
