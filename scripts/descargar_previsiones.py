"""Descarga las previsiones meteorológicas reales archivadas (antelación 1-7 días) a data/raw/meteo/."""
from gemelo.ingesta.previsiones import descargar_previsiones

if __name__ == "__main__":
    descargar_previsiones()
