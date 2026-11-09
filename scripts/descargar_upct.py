"""Descarga las variables in situ L2 de la UPCT (config/zona.yaml > upct) a data/raw/upct/."""
from gemelo.ingesta.upct import descargar_todo

if __name__ == "__main__":
    descargar_todo()
