"""
04_evaluate_models.py
=====================
Comparación y evaluación de los tres modelos entrenados.

Carga los tres archivos .pkl de models/, los evalúa sobre el mismo
conjunto de prueba con distribución real, genera las Figuras 8 a 12
del informe técnico y produce el archivo consolidado metrics.json.

Uso:
    python src/04_evaluate_models.py
    python src/04_evaluate_models.py --data data/raw/dataset_ai4i2020.csv
    python src/04_evaluate_models.py --output-figures results/figures

Salidas:
    results/figures/fig8_matrices_confusion.png
    results/figures/fig9_comparacion_metricas.png
    results/figures/fig10_curvas_roc.png
    results/figures/fig11_importancia_variables.png
    results/figures/fig12_reporte_clasificacion.png
    results/metrics.json

Dependencias: scikit-learn 1.8.0, imbalanced-learn 0.14.1,
              matplotlib 3.10.8, seaborn 0.13.2

Referencias:
    Fawcett, T. (2006). An introduction to ROC analysis.
    Pattern Recognition Letters, 27(8), 861-874.
    https://doi.org/10.1016/j.patrec.2005.10.010

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

from utils.utils_preprocessing import cargar_datos_procesados
from utils.utils_metrics import calcular_metricas, guardar_metricas_json, imprimir_reporte
from utils.utils_visualization import (
    graficar_confusion,
    graficar_metricas_comparacion,
    graficar_roc,
    graficar_importancia_variables,
    graficar_reporte_clasificacion,
)


# ── Modelos a evaluar — rutas leídas desde config.json ────────────────────────
_M = cfg["paths"]["models"]
_R = cfg["paths"]["metrics"]

MODELOS_CONFIG = [
    {
        "nombre":   "Árbol de Decisión",
        "pkl":      _M["decision_tree"],
        "json_ind": _R["decision_tree"],
    },
    {
        "nombre":   "Bosque Aleatorio",
        "pkl":      _M["random_forest"],
        "json_ind": _R["random_forest"],
    },
    {
        "nombre":   "Regresión Logística",
        "pkl":      _M["logistic_regression"],
        "json_ind": _R["logistic_regression"],
    },
]

MEJOR_MODELO_NOMBRE = "Bosque Aleatorio"


# ── CLI ───────────────────────────────────────────────────────────────────────
def _parse_args():
    p = argparse.ArgumentParser(description="Evaluacion y comparacion de modelos.")
    p.add_argument("--models-dir",     default=cfg["paths"]["models_dir"])
    p.add_argument("--output-figures", default=cfg["paths"]["figures_dir"])
    p.add_argument("--output-results", default=cfg["paths"]["results_dir"])
    return p.parse_args()


# ── Cargar modelo serializado ─────────────────────────────────────────────────
def cargar_modelo_pkl(ruta: str):
    """
    Carga un modelo scikit-learn serializado desde un archivo .pkl.
    Las rutas de los archivos .pkl provienen de config.json → paths.models.
    """
    ruta_p = Path(ruta)
    if not ruta_p.exists():
        raise FileNotFoundError(
            f"Modelo no encontrado: {ruta}\n"
            f"Ejecuta primero el script de entrenamiento correspondiente."
        )
    with open(ruta_p, "rb") as f:
        modelo = pickle.load(f)
    print(f"  [OK] Cargado: {ruta}")
    return modelo


# ── Cargar métricas individuales ──────────────────────────────────────────────
def cargar_metricas_individuales(config: dict) -> dict:
    ruta = Path(config["json_ind"])
    if ruta.exists():
        with open(ruta, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("modelos", {}).get(config["nombre"])
    return None


# ── Tabla comparativa en consola ──────────────────────────────────────────────
def tabla_comparativa(resultados: dict) -> str:
    sep   = "=" * 80
    línea = "-" * 80
    print(f"\n{sep}")
    print("  TABLA COMPARATIVA — TODOS LOS MODELOS")
    print(sep)
    print(f"  {'Modelo':<28} {'Accuracy':>9} {'Precision':>9} "
          f"{'Recall':>9} {'F1':>9} {'AUC':>9} {'CV-F1':>12}")
    print(línea)

    # ═══════════════════════════════════════════════════════════════
    # MISMA LÓGICA DE FILTRO DE HIPÓTESIS PARA LA TABLA DE CONSOLA
    # ═══════════════════════════════════════════════════════════════
    recall_minimo = cfg["params"]["recall_minimo"]

    modelos_validos = {
        k: v for k, v in resultados.items() 
        if v["recall"] >= recall_minimo
    }

    if not modelos_validos:
        modelos_validos = resultados
        
    mejor = max(modelos_validos, key=lambda k: (
        modelos_validos[k]["f1"], modelos_validos[k]["auc_roc"]
    ))
    
    for nombre, m in resultados.items():
        marca = " ★ " if nombre == mejor else " "

        cv_str = (f"{m['cv_f1_mean']:.3f}±{m['cv_f1_std']:.3f}"
                  if m.get("cv_f1_mean") else "   N/A   ")

        print(f"  {nombre + marca:<30} "
              f"{m['accuracy']:>9.4f} {m['precision']:>9.4f} "
              f"{m['recall']:>9.4f} {m['f1']:>9.4f} "
              f"{m['auc_roc']:>9.4f} {cv_str:>12}")

        h = m["hipotesis"]
        estado = "NO FALSADA" if h["verificada"] else "FALSADA"

        print(f"  {'':30} Hipotesis (Recall>={h['recall_minimo']}): "
              f"Recall={h['recall_obtenido']:.4f} -> {estado}")

    print(sep)
    print(f"  ★ Mejor modelo por F1: {mejor}\n")
    return mejor


# ── Verificación de hipótesis ─────────────────────────────────────────────────
def verificar_hipotesis(resultados: dict) -> None:
    print("\n" + "=" * 65)
    print("  VERIFICACION DE HIPOTESIS DE SOLUCION")
    print(f"  (Recall >= {cfg['params']['recall_minimo']} como criterio minimo de exito)")
    print("=" * 65)
    for nombre, m in resultados.items():
        h      = m["hipotesis"]
        estado = "OK — NO FALSADA" if h["verificada"] else "FALLO — FALSADA"
        print(f"  {nombre:<28}: Recall={h['recall_obtenido']:.4f}  {estado}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = _parse_args()

    print("\n" + "=" * 65)
    print("  04_evaluate_models.py")
    print("  Comparacion y Evaluacion — AI4I 2020 (Matzka, 2020)")
    print("=" * 65)

    
    # Upload the preprocess data
    print("\n[1/4] Cargando datos preprocesados...")
    _, _, X_test_s, y_test, feature_names = cargar_datos_procesados()
    print(f"  Conjunto de prueba: {len(X_test_s):,} muestras "
          f"({sum(y_test==1)} fallas reales, distribucion original)")


    # Upload and evaluate the production ready ML models
    print("\n[2/4] Cargando modelos y calculando metricas...")
    resultados       = {}
    modelos_cargados = {}

    for cfg_m in MODELOS_CONFIG:
        nombre = cfg_m["nombre"]
        print(f"\n  [{nombre}]")
        try:
            modelo = cargar_modelo_pkl(cfg_m["pkl"])
            modelos_cargados[nombre] = modelo

            m_prev = cargar_metricas_individuales(cfg_m)
            m      = calcular_metricas(
                nombre, modelo, X_test_s, y_test, X_train_bal=None, y_train_bal=None
            )

            if m_prev:
                for key in ("feature_importance", "coeficientes", 
                            "hiperparametros", "cv_f1_mean", "cv_f1_std"):
                    if key in m_prev:
                        m[key] = m_prev[key]

            resultados[nombre] = m

        except FileNotFoundError as e:
            print(f"  [OMITIDO] {e}")

    if not resultados:
        print("\n  ERROR: Ningún modelo encontrado.")
        print("  Ejecuta primero los scripts 03_train_*.py")
        sys.exit(1)

    mejor_nombre = tabla_comparativa(resultados)
    verificar_hipotesis(resultados)


    # Generate the figures
    print("\n[3/4] Generando figuras 8-12...")
    Path(args.output_figures).mkdir(parents=True, exist_ok=True)

    graficar_confusion(resultados, args.output_figures)
    graficar_metricas_comparacion(resultados, args.output_figures)
    graficar_roc(resultados, args.output_figures)

    if mejor_nombre in resultados and "feature_importance" in resultados[mejor_nombre]:
        graficar_importancia_variables(
            resultados, mejor_nombre, feature_names, args.output_figures
        )
    else:
        print(f"  [!] Fig. 11 omitida: {mejor_nombre} no tiene feature_importance")

    graficar_reporte_clasificacion(resultados, args.output_figures)


    # Save the metrics output
    print("\n[4/4] Guardando metricas consolidadas...")
    # Ruta de salida tomada de config.json → paths.metrics.consolidated
    guardar_metricas_json(
        resultados=resultados,
        ruta_salida=str(
            Path(args.output_results) /
            Path(cfg["paths"]["metrics"]["consolidated"]).name
        ),
        meta={
            "dataset":      "AI4I 2020 Predictive Maintenance (Matzka, 2020)",
            "doi":          "https://doi.org/10.24432/C5HS5C",
            "autor":        "Fuentes Zaldivar, A.",
            "proyecto":     "Mantenimiento Predictivo con ML",
            "descripcion":  (
                "Comparacion de Arbol de Decision, Bosque Aleatorio y "
                "Regresion Logistica. Mejor modelo: Bosque Aleatorio "
                "(AUC=0.9710, Recall=0.8824). Fawcett (2006)."
            ),
            "hipotesis_intervalos": {
                "temp_aire_K":      "> 299.10",
                "temp_proceso_K":   "> 309.50",
                "vel_rotacion_rpm": "< 1421.50",
                "torque_Nm":        "> 45.95",
                "desgaste_min":     "> 84.50",
                "recall_minimo":    cfg["params"]["recall_minimo"],
            },
        },
    )

    print("\n" + "=" * 65)
    print("  EVALUACION COMPLETADA")
    print("=" * 65)
    print(f"  Modelos evaluados  : {len(resultados)}")
    print(f"  Mejor modelo       : {mejor_nombre}")
    print(f"  Figuras generadas  : {args.output_figures}/")
    print(f"  JSON consolidado   : {args.output_results}/metrics.json")
    print(f"\n  Siguiente paso: python src/05_save_results.py\n")




if __name__ == "__main__":
    main()



