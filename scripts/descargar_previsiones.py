"""Descarga las previsiones meteorológicas reales archivadas (antelación 1-7 días) a data/raw/meteo/.

    python scripts/descargar_previsiones.py              # solo si no existen
    python scripts/descargar_previsiones.py --rehacer    # vuelve a descargar
"""
import sys

from gemelo.ingesta.previsiones import descargar_previsiones

if __name__ == "__main__":
    descargar_previsiones(sobrescribir="--rehacer" in sys.argv)
