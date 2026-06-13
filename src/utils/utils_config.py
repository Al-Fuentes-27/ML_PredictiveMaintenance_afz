"""
utils/config.py
===============
Carga centralizada de configuración desde dos archivos JSON separados:

    paths.json  — rutas de archivos      → src/utils/paths.json
    params.json — parámetros y modelos   → src/params.json

Ambas rutas están definidas UNA sola vez aquí como constantes.
Si mueves alguno de los archivos JSON, solo cambia la constante correspondiente.

Todos los scripts del proyecto importan `cfg` desde aquí:

    from utils.config import cfg

    ruta_datos  = cfg["paths"]["data"]
    test_size   = cfg["params"]["test_size"]
    hp_rf       = cfg["hyperparameters"]["random_forest"]
"""

import json
from pathlib import Path


# ┌─────────────────────────────────────────────────────────────────────────┐
# │  RUTAS A LOS ARCHIVOS DE CONFIGURACIÓN                                  │
# │  Modifica estas líneas si mueves los archivos JSON.                     │
# │                                                                         │
# │  paths.json  → mismo directorio que este módulo (src/utils/)            │
PATHS_JSON:  Path = Path(__file__).resolve().parent / "utils_paths.json"        #│
# │                                                                         │
# │  params.json → directorio padre (src/), junto a los scripts             │
PARAMS_JSON: Path = Path(__file__).resolve().parent.parent / "params.json"#│
# └─────────────────────────────────────────────────────────────────────────┘


def _cargar(ruta: Path) -> dict:
    """Carga un archivo JSON y lanza FileNotFoundError si no existe."""
    if not ruta.exists():
        raise FileNotFoundError(
            f"Archivo de configuración no encontrado: {ruta}"
        )
    with open(ruta, "r", encoding="utf-8") as f:
        return json.load(f)


def cargar_config() -> dict:
    """
    Carga paths.json y params.json y los combina en un único diccionario.

    Retorna
    -------
    dict con las claves:
        cfg["paths"]           → contenido de paths.json
        cfg["params"]          → sección params de params.json
        cfg["hyperparameters"] → sección hyperparameters de params.json
    """
    raw_paths  = _cargar(PATHS_JSON)
    raw_params = _cargar(PARAMS_JSON)

    # Eliminar claves de metadatos (_comment, _ubicacion) antes de exponer
    paths = {k: v for k, v in raw_paths.items()  if not k.startswith("_")}
    rest  = {k: v for k, v in raw_params.items() if not k.startswith("_")}

    return {
        "paths":           paths,
        "params":          rest.get("params",          {}),
        "hyperparameters": rest.get("hyperparameters", {}),
    }


# Instancia global: todos los scripts importan esta variable
cfg: dict = cargar_config()
