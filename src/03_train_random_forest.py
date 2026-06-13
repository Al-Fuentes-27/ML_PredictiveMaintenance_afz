"""
03_train_random_forest.py
=========================
Entrenamiento y evaluación del modelo Bosque Aleatorio (Random Forest)
para la clasificación de fallas en equipos industriales.

Mejor modelo del proyecto: AUC-ROC=0.9710, Recall=0.8824.
Supera el umbral mínimo de la hipótesis de solución (Recall >= 0.80).

Uso:
    python src/03_train_random_forest.py
    python src/03_train_random_forest.py --n-estimators 100 --max-depth 8

Salidas:
    models/random_forest_v1.pkl
    results/metrics_random_forest.json

Dependencias: scikit-learn 1.8.0, imbalanced-learn 0.14.1

Referencias:
    Breiman, L. (2001). Random forests. Machine Learning, 45(1), 5-32.
    https://doi.org/10.1023/A:1010933404324

    Matzka, S. (2020). AI4I 2020 Predictive Maintenance Dataset.
    UCI Machine Learning Repository. https://doi.org/10.24432/C5HS5C

Autor: Aldo Fuentes Zaldivar — 2025-2026
"""

import sys
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

from sklearn.ensemble import RandomForestClassifier

from utils.utils_preprocessing import cargar_datos_procesados
from utils.utils_metrics import calcular_metricas, guardar_metricas_json, imprimir_reporte


# ── Hiperparámetros leídos desde config.json ──────────────────────────────────
_HP = cfg["hyperparameters"]["random_forest"]
NOMBRE_MODELO  = "Bosque Aleatorio"
VERSION_MODELO = "v1"
HIPERPARAMETROS = {k: v for k, v in _HP.items()}


# ── CLI ───────────────────────────────────────────────────────────────────────
def _parse_args():
    p = argparse.ArgumentParser(description="Bosque Aleatorio — AI4I 2020.")
    p.add_argument("--n-estimators",   type=int,   default=_HP["n_estimators"])
    p.add_argument("--max-depth",      type=int,   default=_HP["max_depth"])
    p.add_argument("--random-state",   type=int,   default=_HP["random_state"])
    p.add_argument("--output-models",  default=cfg["paths"]["models_dir"])
    p.add_argument("--output-results", default=cfg["paths"]["results_dir"])
    return p.parse_args()


# ── Importancia de variables ──────────────────────────────────────────────────
def calcular_importancia(modelo, feature_names: list) -> dict:
    """
    Extrae la importancia de variables medida por reducción de impureza Gini.
    """
    ranking = sorted(
        zip(feature_names, modelo.feature_importances_),
        key=lambda x: x[1], reverse=True,
    )
    print("\n  IMPORTANCIA DE VARIABLES (Gini Impurity Reduction)")
    print("  " + "-" * 55)
    for nombre, imp in ranking:
        barra = "X" * int(imp * 50)
        print(f"  {nombre:<18}: {imp:.4f}  {barra}")
    return {nombre: round(float(imp), 4) for nombre, imp in ranking}


# ── Guardar modelo ────────────────────────────────────────────────────────────
def guardar_modelo(modelo, carpeta: str, version: str = VERSION_MODELO) -> str:
    Path(carpeta).mkdir(parents=True, exist_ok=True)
    # Nombre de archivo tomado de config.json → paths.models.random_forest
    nombre = Path(cfg["paths"]["models"]["random_forest"]).name
    ruta   = str(Path(carpeta) / nombre)
    with open(ruta, "wb") as f:
        pickle.dump(modelo, f)
    print(f"  [OK] Modelo guardado en: {ruta}")
    return ruta


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = _parse_args()

    print("\n" + "=" * 65)
    print("  03_train_random_forest.py")
    print(f"  Modelo: {NOMBRE_MODELO} — AI4I 2020 (Matzka, 2020)")
    print("=" * 65)
    
    
    print("\n[1/2] Cargando datos preprocesados...")
    X_train_bal, y_train_bal, X_test_s, y_test, feature_names = cargar_datos_procesados()

    print("\n[2/2] Construyendo y entrenando el modelo...")
    modelo = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.random_state,
        n_jobs=_HP["n_jobs"],
        max_features=_HP["max_features"],
        bootstrap=_HP["bootstrap"],
        oob_score=_HP["oob_score"],
        class_weight=_HP["class_weight"],
    )

    print(f"  n_estimators : {args.n_estimators}  max_depth: {args.max_depth}")
    print(f"  max_features : {_HP['max_features']}  oob_score: {_HP['oob_score']}  bootstrap: {_HP['bootstrap']}")

    print(f"\n  Entrenando {args.n_estimators} arboles sobre "
          f"{len(X_train_bal):,} muestras (SMOTE)...")

    modelo.fit(X_train_bal, y_train_bal)

    print(f"  OOB Score: {modelo.oob_score_:.4f}")

    print(f"\n  Evaluando sobre {len(X_test_s):,} muestras (real)...")
    importancia_vars = calcular_importancia(modelo, feature_names)
    metricas = calcular_metricas(
        NOMBRE_MODELO, modelo, X_test_s, y_test, X_train_bal, y_train_bal
    )

    metricas["feature_importance"] = importancia_vars

    imprimir_reporte({NOMBRE_MODELO: metricas})


    print("\n Guardando artefactos...")
    ruta_pkl = guardar_modelo(modelo, args.output_models)
    metricas["hiperparametros"] = {
        "n_estimators": args.n_estimators,
        "max_depth":    args.max_depth,
        "max_features": _HP["max_features"],
        "oob_score":    _HP["oob_score"],
        "class_weight": _HP["class_weight"],
        "random_state": args.random_state,
    }
    metricas["oob_score"]  = round(float(modelo.oob_score_), 4)
    metricas["artefactos"] = {"modelo_pkl": ruta_pkl, "feature_names": feature_names}

    # Ruta de salida tomada de config.json → paths.metrics.random_forest
    guardar_metricas_json(
        resultados={NOMBRE_MODELO: metricas},
        ruta_salida=str(
            Path(args.output_results) /
            Path(cfg["paths"]["metrics"]["random_forest"]).name
        ),
        meta={
            "dataset":     "AI4I 2020 Predictive Maintenance (Matzka, 2020)",
            "doi":         "https://doi.org/10.24432/C5HS5C",
            "autor":       "Fuentes Zaldivar, A.",
            "descripcion": ("Bosque Aleatorio (100 arboles) con SMOTE + StandardScaler. "
                            "Mejor modelo: AUC=0.9710, Recall=0.8824 (Breiman, 2001)."),
        },
    )

    print("\n" + "=" * 65)
    print(f"  COMPLETADO — {NOMBRE_MODELO}")
    print("=" * 65)
    print(f"  Recall  : {metricas['recall']:.4f} >= {metricas['hipotesis']['recall_minimo']} "
          f"→ {'NO FALSADA' if metricas['hipotesis']['verificada'] else 'FALSADA'}")
    print(f"  AUC-ROC : {metricas['auc_roc']:.4f}")
    print(f"  OOB     : {metricas['oob_score']:.4f}")
    print(f"\n  Siguiente paso: python src/03_train_logistic_regression.py\n")


if __name__ == "__main__":
    main()
