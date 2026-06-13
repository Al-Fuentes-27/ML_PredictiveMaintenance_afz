"""
utils/metrics.py
================
Funciones reutilizables para el cálculo, serialización e impresión
de métricas de evaluación de modelos de clasificación supervisada.

Funciones:
    calcular_metricas(nombre, modelo, X_test, y_test)
        → dict con todas las métricas del modelo

    guardar_metricas_json(resultados, ruta_salida)
        → escribe results/metrics.json

    imprimir_reporte(resultados)
        → imprime tabla resumen en consola

Dependencias: scikit-learn 1.8.0, numpy 2.4.4, scipy 1.17.1

Referencia del dataset:
    Matzka, S. (2020). AI4I 2020 Predictive Maintenance Dataset.
    UCI Machine Learning Repository. https://doi.org/10.24432/C5HS5C
"""

import json
import numpy as np
from datetime import datetime
from pathlib import Path

# ┌─────────────────────────────────────────────────────────────────────────┐
# │  CONFIGURACIÓN CENTRALIZADA                                             │
# │  La ruta a config.json está definida en utils/config.py → CONFIG_PATH  │
from utils.utils_config import cfg                                               #│
# └─────────────────────────────────────────────────────────────────────────┘

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import cross_val_score, StratifiedKFold


# ── Constantes leídas desde config.json ───────────────────────────────────────
CLASES        = ["Normal", "Falla"]
CV_SPLITS     = cfg["params"]["cv_splits"]
RANDOM_STATE  = cfg["params"]["random_state"]
RECALL_MINIMO = cfg["params"]["recall_minimo"]


def _convert_to_serializable(obj):
    """Recursivamente convierte objetos NumPy a tipos nativos de Python."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: _convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_convert_to_serializable(i) for i in obj]
    elif isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    else:
        try:
            return str(obj)
        except:
            raise TypeError(f"Objeto no serializable: {type(obj)}")


# ── Función principal ─────────────────────────────────────────────────────────
def calcular_metricas(
    nombre: str,
    modelo,
    X_test:      np.ndarray,
    y_test:      np.ndarray,
    X_train_bal: np.ndarray = None,
    y_train_bal: np.ndarray = None,
) -> dict:
    """
    Calcula el conjunto completo de métricas de evaluación para un modelo.

    Parámetros
    ----------
    nombre : str
        Nombre descriptivo del modelo (ej. "Bosque Aleatorio").
    modelo : estimador sklearn
        Modelo ya entrenado (fit) con predict() y predict_proba().
    X_test : np.ndarray
        Variables de entrada del conjunto de prueba (normalizado, sin SMOTE).
    y_test : np.ndarray
        Etiquetas reales del conjunto de prueba.
    X_train_bal : np.ndarray, opcional
        Conjunto de entrenamiento balanceado. Requerido para validación cruzada.
    y_train_bal : np.ndarray, opcional
        Etiquetas del conjunto de entrenamiento balanceado.

    Retorna
    -------
    dict — Diccionario con todas las métricas. Compatible con guardar_metricas_json().
    """
    y_pred = modelo.predict(X_test)
    y_prob = modelo.predict_proba(X_test)[:, 1]

    accuracy  = float(accuracy_score(y_test, y_pred))
    precision = float(precision_score(y_test, y_pred, zero_division=0))
    recall    = float(recall_score(y_test, y_pred, zero_division=0))
    f1        = float(f1_score(y_test, y_pred, zero_division=0))
    auc_roc   = float(roc_auc_score(y_test, y_prob))

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    report = classification_report(
        y_test, y_pred,
        target_names=CLASES,
        output_dict=True,
        zero_division=0,
    )

    fpr, tpr, thresholds = roc_curve(y_test, y_prob)

    cv_mean, cv_std = None, None
    if X_train_bal is not None and y_train_bal is not None:
        cv        = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
        cv_scores = cross_val_score(modelo, X_train_bal, y_train_bal, cv=cv,
                                    scoring="f1", n_jobs=-1)
        cv_mean = float(cv_scores.mean())
        cv_std  = float(cv_scores.std())

    hipotesis_verificada = recall >= RECALL_MINIMO
    
    result = {
        "nombre":    nombre,
        "timestamp": datetime.now().isoformat(),

        "accuracy":  round(accuracy,  4),
        "precision": round(precision, 4),
        "recall":    round(recall,    4),
        "f1":        round(f1,        4),
        "auc_roc":   round(auc_roc,   4),

        "cv_f1_mean": round(cv_mean, 4) if cv_mean is not None else None,
        "cv_f1_std":  round(cv_std,  4) if cv_std  is not None else None,

        "confusion_matrix": {
            "tn": int(tn), "fp": int(fp),
            "fn": int(fn), "tp": int(tp),
            "matrix": cm.tolist(),
        },

        "por_clase": {
            "Normal": {
                "precision": round(report["Normal"]["precision"], 4),
                "recall":    round(report["Normal"]["recall"],    4),
                "f1":        round(report["Normal"]["f1-score"],  4),
                "support":   int(report["Normal"]["support"]),
            },
            "Falla": {
                "precision": round(report["Falla"]["precision"], 4),
                "recall":    round(report["Falla"]["recall"],    4),
                "f1":        round(report["Falla"]["f1-score"],  4),
                "support":   int(report["Falla"]["support"]),
            },
        },

        "curva_roc": {
            "fpr": [round(float(v), 4) for v in fpr],
            "tpr": [round(float(v), 4) for v in tpr],
        },

        "hipotesis": {
            "recall_minimo":   RECALL_MINIMO,
            "recall_obtenido": round(recall, 4),
            "verificada":      hipotesis_verificada,
        },
    }
    
    # Solo añadir cv_f1_mean y cv_f1_std si se calcularon (no son None)
    if cv_mean is not None:
        result["cv_f1_mean"] = round(cv_mean, 4)
        result["cv_f1_std"]  = round(cv_std,  4)

    return result



# ── Guardar JSON ──────────────────────────────────────────────────────────────
def guardar_metricas_json(
    resultados:  dict,
    ruta_salida: str  = None,
    meta:        dict = None,
) -> None:
    """
    Serializa el diccionario de resultados a un archivo JSON.

    Parámetros
    ----------
    resultados  : dict — {nombre_modelo: dict_metricas} de calcular_metricas().
    ruta_salida : str  — Ruta de salida. Si es None, usa config.json →
                         paths.metrics.consolidated.
    meta        : dict — Metadatos adicionales del proyecto.
    """
    # Ruta por defecto tomada de config.json
    if ruta_salida is None:
        ruta_salida = cfg["paths"]["metrics"]["consolidated"]

    salida = Path(ruta_salida)
    salida.parent.mkdir(parents=True, exist_ok=True)

    # ═══════════════════════════════════════════════════════════════
    # SELECCIÓN DEL MEJOR MODELO: ENFOQUE DE FILTRO DE HIPÓTESIS
    # ═══════════════════════════════════════════════════════════════
    # Paso 1: El Recall actúa como "Gatekeeper" (Filtro de Viabilidad)
    recall_minimo = cfg["params"]["recall_minimo"]
    modelos_validos = {
        k: v for k, v in resultados.items()
        if v["recall"] >= recall_minimo
    }
    
    # Si ningún modelo cumple (caso extremo), usar todos para no romper el código
    if not modelos_validos:
        modelos_validos = resultados

    # Paso 2: Entre los viables, optimizar por F1-Score y desempatar con AUC-ROC
    mejor_nombre = max(modelos_validos, key=lambda k: (
        modelos_validos[k]["f1"], modelos_validos[k]["auc_roc"]
    ))

















    payload = {
        "meta":                    _convert_to_serializable(meta or {}),
        "generado_en":             datetime.now().isoformat(),
        "mejor_modelo":            mejor_nombre,
        "recall_minimo_hipotesis": RECALL_MINIMO,
        "hipotesis_verificada":    resultados[mejor_nombre]["hipotesis"]["verificada"],
        "modelos":                 _convert_to_serializable(resultados),
    }

    with open(salida, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"  [✓] Métricas guardadas en: {salida}")
    print(f"  [✓] Mejor modelo: {mejor_nombre} "
          f"(Recall={resultados[mejor_nombre]['recall']:.4f}, "
          f"AUC={resultados[mejor_nombre]['auc_roc']:.4f})")


# ── Impresión en consola ──────────────────────────────────────────────────────
def imprimir_reporte(resultados: dict) -> None:
    """Imprime una tabla resumen de métricas en consola."""
    sep = "─" * 80
    print(f"\n{sep}")
    print(f"  {'MODELO':<26} {'Accuracy':>9} {'Precision':>9} "
          f"{'Recall':>9} {'F1':>9} {'AUC':>9}")
    print(sep)

    mejor = max(resultados, key=lambda k: resultados[k]["f1"])

    for nombre, m in resultados.items():
        marca  = " ★" if nombre == mejor else "  "
        cv_str = (f"  CV={m['cv_f1_mean']:.3f}±{m['cv_f1_std']:.3f}"
                  if m.get("cv_f1_mean") else "")
        print(f"  {nombre + marca:<28} "
              f"{m['accuracy']:>9.4f} {m['precision']:>9.4f} "
              f"{m['recall']:>9.4f} {m['f1']:>9.4f} {m['auc_roc']:>9.4f}"
              f"{cv_str}")

        hipotesis = m["hipotesis"]
        estado    = "✅ NO FALSADA" if hipotesis["verificada"] else "❌ FALSADA"
        print(f"    Hipótesis (Recall≥{hipotesis['recall_minimo']}): "
              f"Recall obtenido={hipotesis['recall_obtenido']:.4f} → {estado}")

    print(sep)
    print(f"  ★ Mejor modelo por F1-Score: {mejor}\n")
