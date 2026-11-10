"""Ciclo de mantenimiento del gemelo: datos -> monitor -> reentrenamiento (si hace falta) -> monitor.

    python scripts/ciclo_mantenimiento.py              # con los datos actuales
    python scripts/ciclo_mantenimiento.py --datos      # reconstruye antes data/processed (tras descargar datos nuevos)

Programarlo (Windows, Programador de tareas), p. ej. cada lunes a las 7:00:
    schtasks /Create /TN "GemeloMarMenor" /SC WEEKLY /D MON /ST 07:00 ^
      /TR "C:\\ruta\\al\\proyecto\\.venv\\Scripts\\python.exe C:\\ruta\\al\\proyecto\\scripts\\ciclo_mantenimiento.py"

¿Por qué no Airflow? Para una sola máquina con 8 GB de RAM y un flujo lineal de 3-4 pasos, Airflow (servidor,
planificador, base de datos, varios contenedores) cuesta más de lo que aporta. Si el gemelo creciera
(varias fuentes en tiempo real, varias máquinas, reintentos por paso), sería el momento de migrar.

Nota: la descarga de datos NUEVOS (ampliar `periodo.fin` y descargar Copernicus/ERA5/previsiones) es un paso
manual: los productos de reanálisis de Copernicus llegan con meses de retraso y para tiempo real harían falta
los productos NRT, todavía sin verificar.
"""
import argparse
import subprocess
import sys
import time

from gemelo.config import RAIZ

SCRIPTS = RAIZ / "scripts"


def ejecutar(nombre, *args):
    print(f"\n[{time.strftime('%H:%M:%S')}] ===== {nombre} {' '.join(args)}", flush=True)
    subprocess.run([sys.executable, str(SCRIPTS / nombre), *args], check=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datos", action="store_true", help="reconstruir data/processed antes de vigilar")
    a = ap.parse_args()
    if a.datos:
        ejecutar("construir_dataset.py")
    ejecutar("monitorizar.py")      # informe -> data/processed/monitor.json
    ejecutar("reentrenar.py")       # solo los modelos que el monitor marque; promociona solo si mejoran
    ejecutar("monitorizar.py")      # informe final, ya con los modelos nuevos (lo sirve GET /monitor)
    print("\nCiclo completado. Si la API está en marcha, reiníciala para cargar los modelos promocionados:"
          "\n    docker compose restart api")
