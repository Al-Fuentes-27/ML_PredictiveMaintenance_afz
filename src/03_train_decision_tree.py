"""
03_train_decision_tree.py
=========================
Entrenamiento y evaluación del modelo Árbol de Decisión
para la clasificación de fallas en equipos industriales.

Uso:
    python src/03_train_decision_tree.py
    python src/03_train_decision_tree.py --data data/raw/dataset_ai4i2020.csv
    python src/03_train_decision_tree.py --max-depth 6 --random-state 42

Salidas:
    models/decision_tree_v1.pkl
    results/metrics_decision_tree.json

Dependencias: scikit-learn 1.8.0, imbalanced-learn 0.14.1,
              numpy 2.4.4, pandas 3.0.2

Referencias:
    Breiman, L., Friedman, J., Stone, C. J., & Olshen, R. A. (1984).
    Classification and regression trees. Chapman & Hall/CRC.

    Matzka, S. (2020). AI4I 2020 Predictive Maintenance Dataset.
    UCI Machine Learning Repository. https://doi.org/10.24432/C5HS5C

Autor: Aldo Fuentes Zaldivar — 2025-2026
"""

import sys
import json
import pickle
import argparse
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))

# ┌─────────────────────────────────────────────────────────────────────────┐
# │  CONFIGURACIÓN CENTRALIZADA                                             │
# │  La ruta a config.json está definida en utils/config.py → CONFIG_PATH  │
from utils.utils_config import cfg                                               #│
# └─────────────────────────────────────────────────────────────────────────┘

from sklearn.tree import DecisionTreeClassifier

from utils.utils_preprocessing import cargar_datos_procesados

from utils.utils_metrics import (
    calcular_metricas,
    guardar_metricas_json,
    imprimir_reporte,
)


# ── Hiperparámetros leídos desde config.json ──────────────────────────────────
_HP = cfg["hyperparameters"]["decision_tree"]
HIPERPARAMETROS = {
    "max_depth":    _HP["max_depth"],
    "random_state": _HP["random_state"],
}

NOMBRE_MODELO  = "Árbol de Decisión"
VERSION_MODELO = "v1"


# ── CLI ───────────────────────────────────────────────────────────────────────
def _parse_args() -> argparse.Namespace:
    
    parser = argparse.ArgumentParser(
        description="Entrenamiento del Árbol de Decisión — AI4I 2020."
    )
    
    parser.add_argument(
        "--max-depth", type=int,
        default=HIPERPARAMETROS["max_depth"],
        help=f"Profundidad máxima del árbol (default: {HIPERPARAMETROS['max_depth']})",
    )
    parser.add_argument(
        "--random-state", type=int,
        default=HIPERPARAMETROS["random_state"],
        help="Semilla aleatoria (default: 42)",
    )
    parser.add_argument(
        "--output-models",
        default=cfg["paths"]["models_dir"],
        help=f"Carpeta para guardar el modelo serializado (default: {cfg['paths']['models_dir']})",
    )
    parser.add_argument(
        "--output-results",
        default=cfg["paths"]["results_dir"],
        help=f"Carpeta para métricas JSON (default: {cfg['paths']['results_dir']})",
    )
    
    return parser.parse_args()


# ── Construir modelo ──────────────────────────────────────────────────────────
def construir_modelo(max_depth: int, random_state: int) -> DecisionTreeClassifier:
    """
    Instancia el Árbol de Decisión con los hiperparámetros especificados.
    Los valores por defecto provienen de config.json → hyperparameters.decision_tree.
    """
    hp = cfg["hyperparameters"]["decision_tree"]
    return DecisionTreeClassifier(
        max_depth=max_depth,
        random_state=random_state,
        class_weight=hp["class_weight"],
        criterion=hp["criterion"],
    )


# ── Guardar modelo ────────────────────────────────────────────────────────────
def guardar_modelo(modelo, carpeta: str, version: str = VERSION_MODELO) -> str:
    Path(carpeta).mkdir(parents=True, exist_ok=True)
    # Nombre de archivo tomado de config.json → paths.models.decision_tree
    nombre   = Path(cfg["paths"]["models"]["decision_tree"]).name
    ruta_pkl = str(Path(carpeta) / nombre)
    with open(ruta_pkl, "wb") as f:
        pickle.dump(modelo, f)
    print(f"  [OK] Modelo guardado en: {ruta_pkl}")
    return ruta_pkl


# ── Imprimir parámetros ───────────────────────────────────────────────────────
def imprimir_parametros(modelo: DecisionTreeClassifier) -> None:
    params = modelo.get_params()
    print("\n  HIPERPARAMETROS CONFIGURADOS")
    print("  " + "-" * 40)
    for k, v in params.items():
        print(f"  {k:<25}: {v}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    args = _parse_args()

    print("\n" + "=" * 65)
    print(f"  03_train_decision_tree.py")
    print(f"  Modelo: {NOMBRE_MODELO} — AI4I 2020 (Matzka, 2020)")
    print("=" * 65)


    print("\n[1/2] Cargando datos preprocesados...")
    X_train_bal, y_train_bal, X_test_s, y_test, feature_names = cargar_datos_procesados()

    print("\n[2/2] Construyendo y entrenando el modelo...")
    modelo = construir_modelo(
        max_depth=args.max_depth,
        random_state=args.random_state,
    )
    imprimir_parametros(modelo)

    print(f"\n  Entrenando sobre {len(X_train_bal):,} muestras "
          f"(balanceadas con SMOTE)...")
    modelo.fit(X_train_bal, y_train_bal)

    print(f"  Profundidad real del árbol : {modelo.get_depth()}")
    print(f"  Número de hojas            : {modelo.get_n_leaves()}")
    print(f"  Nodos internos             : "
          f"{modelo.tree_.node_count - modelo.get_n_leaves()}")

    print(f"\n  Evaluando sobre {len(X_test_s):,} muestras "
          f"(distribución real, sin SMOTE)...")
    metricas = calcular_metricas(
        nombre=NOMBRE_MODELO,
        modelo=modelo,
        X_test=X_test_s,
        y_test=y_test,
        X_train_bal=X_train_bal,
        y_train_bal=y_train_bal,
    )
    imprimir_reporte({NOMBRE_MODELO: metricas})

    print("\n  Guardando modelo y metricas...")
    ruta_pkl = guardar_modelo(modelo, args.output_models)

    hp = cfg["hyperparameters"]["decision_tree"]
    metricas["hiperparametros"] = {
        "max_depth":    args.max_depth,
        "random_state": args.random_state,
        "class_weight": hp["class_weight"],
        "criterion":    hp["criterion"],
    }
    metricas["artefactos"] = {
        "modelo_pkl":    ruta_pkl,
        "feature_names": feature_names,
    }
    metricas["profundidad_real"] = modelo.get_depth()
    metricas["num_hojas"]        = modelo.get_n_leaves()

    # Ruta de salida tomada de config.json → paths.metrics.decision_tree
    guardar_metricas_json(
        resultados={NOMBRE_MODELO: metricas},
        ruta_salida=str(
            Path(args.output_results) /
            Path(cfg["paths"]["metrics"]["decision_tree"]).name
        ),
        meta={
            "dataset":     "AI4I 2020 Predictive Maintenance (Matzka, 2020)",
            "doi":         "https://doi.org/10.24432/C5HS5C",
            "autor":       "Fuentes Zaldivar, A.",
            "descripcion": (
                "Arbol de Decisión entrenado con SMOTE + StandardScaler. "
                "Modelo interpretable de baja complejidad (Breiman et al., 1984)."
            ),
        },
    )


    print("\n" + "=" * 65)
    print(f"  ENTRENAMIENTO COMPLETADO — {NOMBRE_MODELO}")
    print("=" * 65)
    print(f"  Accuracy  : {metricas['accuracy']:.4f}")
    print(f"  Precision : {metricas['precision']:.4f}")
    print(f"  Recall    : {metricas['recall']:.4f}  "
          f"(umbral hipotesis >= {metricas['hipotesis']['recall_minimo']})")
    print(f"  F1-Score  : {metricas['f1']:.4f}")
    print(f"  AUC-ROC   : {metricas['auc_roc']:.4f}")
    if metricas.get("cv_f1_mean"):
        print(f"  CV-F1     : {metricas['cv_f1_mean']:.4f} "
              f"+/- {metricas['cv_f1_std']:.4f}")
    estado = "NO FALSADA" if metricas["hipotesis"]["verificada"] else "FALSADA"
    print(f"\n  Hipotesis de solucion: {estado}")
    print(f"\n  Siguiente paso: python src/03_train_random_forest.py\n")


if __name__ == "__main__":
    main()
