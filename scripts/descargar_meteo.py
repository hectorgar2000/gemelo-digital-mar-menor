"""Descarga la meteorología ERA5 (Open-Meteo) a data/raw/meteo/."""
from gemelo.ingesta.meteo import descargar_meteo

if __name__ == "__main__":
    descargar_meteo()
