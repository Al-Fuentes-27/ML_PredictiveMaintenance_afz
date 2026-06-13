"""
03_train_logistic_regression.py
================================
Entrenamiento y evaluación del modelo Regresión Logística
como modelo de referencia (baseline) para la clasificación de fallas.

Uso:
    python src/03_train_logistic_regression.py
    python src/03_train_logistic_regression.py --C 1.0 --max-iter 1000

Salidas:
    models/logistic_regression_v1.pkl
    results/metrics_logistic_regression.json

Dependencias: scikit-learn 1.8.0, imbalanced-learn 0.14.1

Referencias:
    James, G., Witten, D., Hastie, T., & Tibshirani, R. (2013).
    An introduction to statistical learning. Springer.
    https://doi.org/10.1007/978-1-4614-7138-7

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

from sklearn.linear_model import LogisticRegression

from utils.utils_preprocessing import cargar_datos_procesados
from utils.utils_metrics import calcular_metricas, guardar_metricas_json, imprimir_reporte


# ── Hiperparámetros leídos desde config.json ──────────────────────────────────
_HP = cfg["hyperparameters"]["logistic_regression"]
NOMBRE_MODELO  = "Regresión Logística"
VERSION_MODELO = "v1"
HIPERPARAMETROS = {k: v for k, v in _HP.items()}


# ── CLI ───────────────────────────────────────────────────────────────────────
def _parse_args():
    p = argparse.ArgumentParser(description="Regresion Logistica — AI4I 2020.")
    p.add_argument("--C",              type=float, default=_HP["C"],
                   help=f"Parametro de regularizacion L2 (default: {_HP['C']})")
    p.add_argument("--max-iter",       type=int,   default=_HP["max_iter"])
    p.add_argument("--random-state",   type=int,   default=_HP["random_state"])
    p.add_argument("--output-models",  default=cfg["paths"]["models_dir"])
    p.add_argument("--output-results", default=cfg["paths"]["results_dir"])
    return p.parse_args()


# ── Coeficientes del modelo ───────────────────────────────────────────────────
def analizar_coeficientes(
    modelo:        LogisticRegression,
    feature_names: list,
) -> dict:
    """
    Extrae e interpreta los coeficientes de la Regresión Logística.
    """
    coefs   = modelo.coef_[0]
    ranking = sorted(
        zip(feature_names, coefs),
        key=lambda x: abs(x[1]), reverse=True,
    )

    print("\n  COEFICIENTES DE LA REGRESION LOGISTICA")
    print("  (Impacto sobre log-odds de falla, variables normalizadas)")
    print("  " + "-" * 58)
    for nombre, coef in ranking:
        direccion = "[+]" if coef > 0 else "[-]"
        barra     = "X" * int(abs(coef) * 8)
        print(f"  {nombre:<18}: {coef:>+8.4f}  {direccion}  {barra}")
    print(f"\n  Intercepto: {modelo.intercept_[0]:.4f}")

    return {nombre: round(float(coef), 4) for nombre, coef in ranking}


# ── Guardar modelo ────────────────────────────────────────────────────────────
def guardar_modelo(modelo, carpeta: str, version: str = VERSION_MODELO) -> str:
    Path(carpeta).mkdir(parents=True, exist_ok=True)
    # Nombre de archivo tomado de config.json → paths.models.logistic_regression
    nombre = Path(cfg["paths"]["models"]["logistic_regression"]).name
    ruta   = str(Path(carpeta) / nombre)
    with open(ruta, "wb") as f:
        pickle.dump(modelo, f)
    print(f"  [OK] Modelo guardado en: {ruta}")
    return ruta


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = _parse_args()

    print("\n" + "=" * 65)
    print("  03_train_logistic_regression.py")
    print(f"  Modelo: {NOMBRE_MODELO} (baseline) — AI4I 2020 (Matzka, 2020)")
    print("=" * 65)
    print("\n  Rol del modelo: referencia lineal para comparar contra")
    print("  modelos no lineales (Arbol de Decision y Bosque Aleatorio).")

    print("\n[1/2] Cargando datos preprocesados...")
    X_train_bal, y_train_bal, X_test_s, y_test, feature_names = cargar_datos_procesados()

    print("\n[2/2] Construyendo y entrenando el modelo...")
    modelo = LogisticRegression(
        C=args.C,
        max_iter=args.max_iter,
        random_state=args.random_state,
        solver=_HP["solver"],
        penalty=_HP["penalty"],
        multi_class="auto",
    )
    print(f"  C (regularizacion) : {args.C}")
    print(f"  max_iter           : {args.max_iter}")
    print(f"  solver             : {_HP['solver']}")
    print(f"  penalty            : {_HP['penalty']}")

    print(f"\n  Entrenando sobre {len(X_train_bal):,} muestras (SMOTE)...")
    modelo.fit(X_train_bal, y_train_bal)

    if hasattr(modelo, "n_iter_"):
        iters_usadas = modelo.n_iter_[0]
        convergio    = iters_usadas < args.max_iter
        estado_conv  = "OK" if convergio else "ADVERTENCIA: no convergido"
        print(f"  Iteraciones usadas : {iters_usadas}/{args.max_iter}  [{estado_conv}]")

    print(f"\n  Evaluando sobre {len(X_test_s):,} muestras (real)...")
    coeficientes = analizar_coeficientes(modelo, feature_names)
    metricas = calcular_metricas(
        NOMBRE_MODELO, modelo, X_test_s, y_test, X_train_bal, y_train_bal
    )
    metricas["coeficientes"] = coeficientes
    metricas["intercepto"]   = round(float(modelo.intercept_[0]), 4)
    imprimir_reporte({NOMBRE_MODELO: metricas})


    print("\n Guardando artefactos...")
    ruta_pkl = guardar_modelo(modelo, args.output_models)
    metricas["hiperparametros"] = {
        "C":            args.C,
        "max_iter":     args.max_iter,
        "solver":       _HP["solver"],
        "penalty":      _HP["penalty"],
        "random_state": args.random_state,
    }
    metricas["artefactos"] = {"modelo_pkl": ruta_pkl, "feature_names": feature_names}

    # Ruta de salida tomada de config.json → paths.metrics.logistic_regression
    guardar_metricas_json(
        resultados={NOMBRE_MODELO: metricas},
        ruta_salida=str(
            Path(args.output_results) /
            Path(cfg["paths"]["metrics"]["logistic_regression"]).name
        ),
        meta={
            "dataset":     "AI4I 2020 Predictive Maintenance (Matzka, 2020)",
            "doi":         "https://doi.org/10.24432/C5HS5C",
            "autor":       "Fuentes Zaldivar, A.",
            "descripcion": (
                "Regresion Logistica (baseline) con SMOTE + StandardScaler. "
                "Evalua separabilidad lineal de las clases (James et al., 2013)."
            ),
        },
    )

    print("\n" + "=" * 65)
    print(f"  COMPLETADO — {NOMBRE_MODELO}")
    print("=" * 65)
    print(f"  Recall  : {metricas['recall']:.4f} >= {metricas['hipotesis']['recall_minimo']} "
          f"-> {'NO FALSADA' if metricas['hipotesis']['verificada'] else 'FALSADA'}")
    print(f"  AUC-ROC : {metricas['auc_roc']:.4f}")
    print(f"\n  Siguiente paso: python src/04_evaluate_models.py\n")


if __name__ == "__main__":
    main()
