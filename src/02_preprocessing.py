"""
02_preprocessing.py
===================
Pipeline de preparación de datos para el proyecto de mantenimiento predictivo.

Ejecuta en orden: carga, codificación, división estratificada,
normalización y balanceo SMOTE. Serializa los artefactos resultantes
(scaler, índices de división) en models/ para que los scripts de
entrenamiento los consuman sin re-procesar los datos.

Uso:
    python src/02_preprocessing.py
    python src/02_preprocessing.py --data data/raw/dataset_ai4i2020.csv
    python src/02_preprocessing.py --test-size 0.20 --random-state 42

Salidas:
    models/scaler_v1.pkl          — StandardScaler ajustado sobre X_train
    results/figures/fig7_preparacion_dataset.png
    results/preprocessing_summary.json

Dependencias: pandas 3.0.2, numpy 2.4.4, scikit-learn 1.8.0,
              imbalanced-learn 0.14.1

Referencias:
    Matzka, S. (2020). AI4I 2020 Predictive Maintenance Dataset.
    UCI Machine Learning Repository. https://doi.org/10.24432/C5HS5C

    Lemaître, G., Nogueira, F., & Aridas, C. K. (2017).
    Imbalanced-learn. JMLR, 18(17), 1-5.
    https://jmlr.org/papers/v18/16-365.html

Autor: Aldo Fuentes Zaldívar — 2025-2026
"""

import sys
import json
import pickle
import argparse
import warnings
from pathlib import Path
from datetime import datetime

import numpy  as np

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))

# ┌─────────────────────────────────────────────────────────────────────────┐
# │  CONFIGURACIÓN CENTRALIZADA                                             │
# │  La ruta a config.json está definida en utils/config.py → CONFIG_PATH  │
from utils.utils_config import cfg                                               #│
# └─────────────────────────────────────────────────────────────────────────┘

from utils.utils_preprocessing import (
    cargar_datos,
    preparar_features,
    dividir_datos,
    normalizar,
    balancear_smote,
    guardar_datos_procesados,
    FEATURE_NAMES,
    INTERVALOS_RIESGO,
)
from utils.utils_visualization import graficar_preparacion


# ── CLI ───────────────────────────────────────────────────────────────────────
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pipeline de preparación de datos — AI4I 2020."
    )
    parser.add_argument(
        "--data",
        default=cfg["paths"]["data"],
        help=f"Ruta al dataset CSV (default: {cfg['paths']['data']})",
    )
    parser.add_argument(
        "--test-size", type=float,
        default=cfg["params"]["test_size"],
        help=f"Proporción del conjunto de prueba (default: {cfg['params']['test_size']})",
    )
    parser.add_argument(
        "--random-state", type=int,
        default=cfg["params"]["random_state"],
        help=f"Semilla aleatoria para reproducibilidad (default: {cfg['params']['random_state']})",
    )
    parser.add_argument(
        "--output-models",
        default=cfg["paths"]["models_dir"],
        help=f"Carpeta para guardar el scaler (default: {cfg['paths']['models_dir']})",
    )
    parser.add_argument(
        "--output-figures",
        default=cfg["paths"]["figures_dir"],
        help=f"Carpeta para figuras (default: {cfg['paths']['figures_dir']})",
    )
    parser.add_argument(
        "--output-results",
        default=cfg["paths"]["results_dir"],
        help=f"Carpeta para resumen JSON (default: {cfg['paths']['results_dir']})",
    )
    return parser.parse_args()


# ── Guardar artefactos ────────────────────────────────────────────────────────
def guardar_scaler(scaler, carpeta: str, version: str = "v1") -> str:
    """
    Serializa el StandardScaler como archivo .pkl.

    El scaler debe guardarse junto a los modelos porque en producción
    los datos nuevos deben normalizarse con la misma media y desviación
    estándar que se usaron durante el entrenamiento.
    """
    Path(carpeta).mkdir(parents=True, exist_ok=True)
    # Nombre de archivo tomado de config.json → paths.scaler
    ruta = str(Path(carpeta) / Path(cfg["paths"]["scaler"]).name)
    with open(ruta, "wb") as f:
        pickle.dump(scaler, f)
    print(f"  [OK] Scaler guardado en: {ruta}")
    return ruta


def guardar_resumen_json(resumen: dict, carpeta: str) -> None:
    """Guarda el resumen del pipeline de preprocesamiento en JSON."""
    Path(carpeta).mkdir(parents=True, exist_ok=True)
    # Nombre de archivo tomado de config.json → paths.metrics.preprocessing
    ruta = Path(carpeta) / Path(cfg["paths"]["metrics"]["preprocessing"]).name
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(resumen, f, ensure_ascii=False, indent=2)
    print(f"  [OK] Resumen guardado en: {ruta}")


# ── Validar hipótesis de intervalos ──────────────────────────────────────────
def validar_intervalos_hipotesis(df) -> None:
    falla_df = df[df["falla"] == 1]

    print("\n" + "=" * 65)
    print("  VERIFICACION DE INTERVALOS — HIPOTESIS DE SOLUCION")
    print("=" * 65)
    print(f"  {'Variable':<24} {'Umbral hipotesis':>18} "
          f"{'Q1 observado':>14} {'Estado':>10}")
    print("  " + "-" * 65)

    for var, config in INTERVALOS_RIESGO.items():
        umbral    = config["umbral"]
        direccion = config["direccion"]
        unidad    = config["unidad"]
        q1_obs    = round(falla_df[var].quantile(0.25), 2)
        if direccion == ">":
            ok = q1_obs >= umbral * 0.95
        else:
            ok = q1_obs <= umbral * 1.05
        estado = "OK" if ok else "REVISAR"
        print(f"  {var:<24} {direccion} {umbral:>8.2f} {unidad:<5} "
              f"{q1_obs:>12.2f}   {estado:>8}")
    print()


# ── Reporte de balanceo ───────────────────────────────────────────────────────
def reporte_balanceo(y_train: np.ndarray, y_train_bal: np.ndarray) -> None:
    print("\n  EFECTO DE SMOTE SOBRE EL CONJUNTO DE ENTRENAMIENTO")
    print("  " + "-" * 50)
    print(f"  Antes  — Normal: {sum(y_train==0):>6,}  "
          f"Falla: {sum(y_train==1):>4,}  "
          f"({sum(y_train==1)/len(y_train)*100:.2f}%)")
    print(f"  Despues — Normal: {sum(y_train_bal==0):>6,}  "
          f"Falla: {sum(y_train_bal==1):>4,}  "
          f"(50.00%)")
    print(f"  Muestras sinteticas generadas: "
          f"{sum(y_train_bal==1) - sum(y_train==1):,}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    args = _parse_args()

    print("\n" + "=" * 65)
    print("  02_preprocessing.py")
    print("  Pipeline de Preparacion de Datos — AI4I 2020 (Matzka, 2020)")
    print("=" * 65)

    print("\n[1/6] Cargando dataset...")
    df = cargar_datos(args.data)

    print("\n[2/6] Validando intervalos de la hipotesis...")
    validar_intervalos_hipotesis(df)

    print("\n[3/6] Preparando features y variable objetivo...")
    X, y, feature_names = preparar_features(df)

    print("\n[4/6] Dividiendo dataset (estratificado)...")
    X_train, X_test, y_train, y_test = dividir_datos(
        X, y,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    print("\n[5/6] Normalizando con StandardScaler...")
    X_train_s, X_test_s, scaler, _ = normalizar(X_train, X_test)












    print("\n[6/7] Aplicando SMOTE al conjunto de entrenamiento...")

    X_train_bal, y_train_bal = balancear_smote(
        X_train_s, y_train,
        random_state=args.random_state,
    )

    reporte_balanceo(y_train, y_train_bal)
    
    print("\n[7/7] Guardando datos procesados en data/processed/...")
    
    guardar_datos_procesados(X_train_bal, y_train_bal, X_test_s, y_test, feature_names)
    
    ruta_scaler = guardar_scaler(scaler, args.output_models)

    Path(args.output_figures).mkdir(parents=True, exist_ok=True)

    graficar_preparacion(
        y_train, y_train_bal,
        X_train_size=len(X_train),
        X_test_size=len(X_test),
        ruta_salida=args.output_figures,
    )


    resumen = {
        "timestamp":       datetime.now().isoformat(),
        "dataset":         args.data,
        "total_registros": int(len(df)),
        "features":        feature_names,
        "test_size":       args.test_size,
        "random_state":    args.random_state,
        "split": {
            "X_train": int(len(X_train)),
            "X_test":  int(len(X_test)),
            "y_train_normal": int(sum(y_train == 0)),
            "y_train_falla":  int(sum(y_train == 1)),
            "y_test_normal":  int(sum(y_test  == 0)),
            "y_test_falla":   int(sum(y_test  == 1)),
        },
        "smote": {
            "y_train_bal_normal":  int(sum(y_train_bal == 0)),
            "y_train_bal_falla":   int(sum(y_train_bal == 1)),
            "muestras_sinteticas": int(
                sum(y_train_bal == 1) - sum(y_train == 1)
            ),
        },
        "scaler_path": ruta_scaler,
    }
    guardar_resumen_json(resumen, args.output_results)

    print("\n" + "=" * 65)
    print("  PIPELINE COMPLETADO")
    print("=" * 65)
    print(f"  Total registros    : {len(df):,}")
    print(f"  Entrenamiento      : {len(X_train):,} "
          f"→ SMOTE: {len(X_train_bal):,}")
    print(f"  Prueba             : {len(X_test):,} (distribucion real)")
    print(f"  Scaler guardado    : {ruta_scaler}")
    print(f"\n  Siguiente paso: python src/03_train_decision_tree.py")
    print(f"                  python src/03_train_random_forest.py")
    print(f"                  python src/03_train_logistic_regression.py\n")

    return {
        "X_train_bal": X_train_bal,
        "y_train_bal": y_train_bal,
        "X_test_s":    X_test_s,
        "y_test":      y_test,
        "scaler":      scaler,
        "feature_names": feature_names,
    }


if __name__ == "__main__":
    main()
