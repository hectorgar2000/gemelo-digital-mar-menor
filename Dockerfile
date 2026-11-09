# Imagen del servicio del gemelo (Fase 4). Solo lleva el código y las dependencias de la API;
# los modelos exportados y los datos procesados se MONTAN como volúmenes (se actualizan sin reconstruir).
#
#   docker build -t gemelo-mar-menor .
#   docker compose up            (ver compose.yaml)

FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    GEMELO_RAIZ=/app

WORKDIR /app

# 1) Dependencias (capa cacheable): versiones fijadas con el lock del entorno de desarrollo
COPY pyproject.toml README.md requirements.lock.txt ./
COPY src ./src
RUN pip install -c requirements.lock.txt ".[api]"

# 2) Configuración del gemelo (zona, umbrales, partición)
COPY config ./config

# Usuario sin privilegios
RUN useradd --create-home gemelo && mkdir -p models/servicio data/processed && chown -R gemelo /app
USER gemelo

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/salud')"
CMD ["uvicorn", "gemelo.servicio.api:app", "--host", "0.0.0.0", "--port", "8000"]
