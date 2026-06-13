"""
05_save_results.py
==================
Consolidación final de resultados del proyecto de mantenimiento predictivo.

Lee el metrics.json generado por 04_evaluate_models.py y produce:
  1. Reporte de texto plano (metrics_summary.txt)
  2. CSV con métricas de todos los modelos (metrics_summary.csv)
  3. JSON enriquecido listo para el dashboard HTML interactivo (dashboard_data.json)
  4. Imprime el resumen final en consola con verificación de hipótesis

Uso:
    python src/05_save_results.py
    python src/05_save_results.py --metrics results/metrics.json
    python src/05_save_results.py --output results

Salidas:
    results/reports/metrics_summary.txt
    results/reports/metrics_summary.csv
    dashboard/src/data/dashboard_data.json

Dependencias: pandas 3.0.2

Referencias:
    Matzka, S. (2020). AI4I 2020 Predictive Maintenance Dataset.
    UCI Machine Learning Repository. https://doi.org/10.24432/C5HS5C

Autor: Aldo Fuentes Zaldivar — 2025-2026
"""

import sys
import csv
import json
import argparse
import warnings
from pathlib import Path
from datetime import datetime

import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

# ┌─────────────────────────────────────────────────────────────────────────┐
# │  CONFIGURACIÓN CENTRALIZADA                                             │
# │  La ruta a config.json está definida en utils/config.py → CONFIG_PATH  │
from utils.utils_config import cfg                                               #│
# └─────────────────────────────────────────────────────────────────────────┘


# ── CLI ───────────────────────────────────────────────────────────────────────
def _parse_args():
    p = argparse.ArgumentParser(description="Consolidacion final de resultados.")
    p.add_argument(
        "--metrics",
        default=cfg["paths"]["metrics"]["consolidated"],
        help=f"Ruta al JSON consolidado de metricas (default: {cfg['paths']['metrics']['consolidated']})",
    )
    p.add_argument(
        "--output",
        default=cfg["paths"]["results_dir"],
        help=f"Carpeta raiz de salida (default: {cfg['paths']['results_dir']})",
    )
    p.add_argument(
        "--dashboard-output",
        default=cfg["paths"]["metrics"]["dashboard_data_dir"],
        help=f"Carpeta para dashboard_data.json (default: {cfg['paths']['metrics']['dashboard_data_dir']})",
    )
    return p.parse_args()


# ── Cargar métricas ───────────────────────────────────────────────────────────
def cargar_metrics_json(ruta: str) -> dict:
    """
    Carga el JSON de métricas consolidadas generado por 04_evaluate_models.py.
    La ruta por defecto proviene de config.json → paths.metrics.consolidated.
    """
    ruta_p = Path(ruta)
    if not ruta_p.exists():
        raise FileNotFoundError(
            f"Archivo de metricas no encontrado: {ruta}\n"
            "Ejecuta primero: python src/04_evaluate_models.py"
        )
    with open(ruta_p, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"  [OK] Metricas cargadas: {ruta}")
    return data


# ── Reporte de texto ──────────────────────────────────────────────────────────
def generar_reporte_texto(data: dict, carpeta: str) -> str:
    Path(carpeta).mkdir(parents=True, exist_ok=True)
    ruta = str(Path(carpeta) / "metrics_summary.txt")
    sep  = "=" * 70

    with open(ruta, "w", encoding="utf-8") as f:
        f.write(f"{sep}\n")
        f.write("  RESUMEN DE METRICAS — MANTENIMIENTO PREDICTIVO\n")
        f.write("  AI4I 2020 Predictive Maintenance Dataset (Matzka, 2020)\n")
        f.write(f"  Generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{sep}\n\n")

        meta = data.get("meta", {})
        f.write(f"Dataset : {meta.get('dataset', 'AI4I 2020')}\n")
        f.write(f"DOI     : {meta.get('doi', 'https://doi.org/10.24432/C5HS5C')}\n")
        f.write(f"Autor   : {meta.get('autor', 'Fuentes Zaldivar, A.')}\n\n")

        f.write(f"Mejor modelo     : {data.get('mejor_modelo', 'N/A')}\n")
        f.write(f"Recall minimo    : {data.get('recall_minimo_hipotesis', cfg['params']['recall_minimo'])}\n")
        est_hip = "NO FALSADA" if data.get("hipotesis_verificada") else "FALSADA"
        f.write(f"Hipotesis        : {est_hip}\n\n")

        f.write(f"{sep}\n")
        f.write("  METRICAS POR MODELO\n")
        f.write(f"{sep}\n")
        for nombre, m in data.get("modelos", {}).items():
            f.write(f"\n  [{nombre}]\n")
            f.write(f"  {'Accuracy':<14}: {m['accuracy']:.4f}\n")
            f.write(f"  {'Precision':<14}: {m['precision']:.4f}\n")
            f.write(f"  {'Recall':<14}: {m['recall']:.4f}\n")
            f.write(f"  {'F1-Score':<14}: {m['f1']:.4f}\n")
            f.write(f"  {'AUC-ROC':<14}: {m['auc_roc']:.4f}\n")
            if m.get("cv_f1_mean"):
                f.write(f"  {'CV-F1':<14}: {m['cv_f1_mean']:.4f} "
                        f"+/- {m['cv_f1_std']:.4f}\n")
            cm = m.get("confusion_matrix", {})
            f.write(f"  {'TP (fallas det.)':<14}: {cm.get('tp', 'N/A')}\n")
            f.write(f"  {'FN (fallas perd.)':<14}: {cm.get('fn', 'N/A')}\n")
            f.write(f"  {'FP (falsas alarm.)':<14}: {cm.get('fp', 'N/A')}\n")

            h   = m.get("hipotesis", {})
            est = "NO FALSADA" if h.get("verificada") else "FALSADA"
            f.write(f"\n  Hipotesis (Recall>={h.get('recall_minimo', cfg['params']['recall_minimo'])}): "
                    f"Recall={h.get('recall_obtenido',0):.4f} -> {est}\n")

            if "feature_importance" in m:
                f.write(f"\n  Importancia de variables:\n")
                for var, imp in sorted(
                    m["feature_importance"].items(), key=lambda x: x[1], reverse=True
                ):
                    f.write(f"    {var:<18}: {imp:.4f}\n")

            f.write("\n" + "-" * 50 + "\n")

        hip = data.get("meta", {}).get("hipotesis_intervalos", {})
        if hip:
            f.write(f"\n{sep}\n")
            f.write("  HIPOTESIS DE SOLUCION — INTERVALOS DE RIESGO\n")
            f.write(f"{sep}\n")
            for var, umbral in hip.items():
                if var != "recall_minimo":
                    f.write(f"  {var:<25}: {umbral}\n")
            f.write(f"  {'Recall minimo':<25}: >= {hip.get('recall_minimo', cfg['params']['recall_minimo'])}\n")

    print(f"  [OK] Reporte guardado en: {ruta}")
    return ruta


# ── CSV de métricas ───────────────────────────────────────────────────────────
def generar_csv_metricas(data: dict, carpeta: str) -> str:
    Path(carpeta).mkdir(parents=True, exist_ok=True)
    ruta = str(Path(carpeta) / "metrics_summary.csv")

    filas = []
    for nombre, m in data.get("modelos", {}).items():
        cm = m.get("confusion_matrix", {})
        h  = m.get("hipotesis", {})
        filas.append({
            "Modelo":             nombre,
            "Accuracy":           m["accuracy"],
            "Precision":          m["precision"],
            "Recall":             m["recall"],
            "F1_Score":           m["f1"],
            "AUC_ROC":            m["auc_roc"],
            "CV_F1_Mean":         m.get("cv_f1_mean"),
            "CV_F1_Std":          m.get("cv_f1_std"),
            "TP":                 cm.get("tp"),
            "FP":                 cm.get("fp"),
            "TN":                 cm.get("tn"),
            "FN":                 cm.get("fn"),
            "Precision_Normal":   m.get("por_clase", {}).get("Normal", {}).get("precision"),
            "Recall_Normal":      m.get("por_clase", {}).get("Normal", {}).get("recall"),
            "F1_Normal":          m.get("por_clase", {}).get("Normal", {}).get("f1"),
            "Precision_Falla":    m.get("por_clase", {}).get("Falla", {}).get("precision"),
            "Recall_Falla":       m.get("por_clase", {}).get("Falla", {}).get("recall"),
            "F1_Falla":           m.get("por_clase", {}).get("Falla", {}).get("f1"),
            "Hipotesis_Recall":   h.get("recall_minimo"),
            "Hipotesis_Cumplida": h.get("verificada"),
            "Mejor_Modelo":       nombre == data.get("mejor_modelo"),
        })

    df = pd.DataFrame(filas)
    df.to_csv(ruta, index=False, encoding="utf-8")
    print(f"  [OK] CSV guardado en: {ruta}")
    return ruta


# ── JSON para dashboard React ─────────────────────────────────────────────────
def generar_dashboard_json(data: dict, carpeta: str) -> str:
    """
    Genera el archivo dashboard_data.json consumido por la app React.
    La carpeta de destino proviene de config.json → paths.dashboard_dir.
    """
    Path(carpeta).mkdir(parents=True, exist_ok=True)
    ruta = str(Path(carpeta) / "dashboard_data.json")

    mejor_nombre = data.get("mejor_modelo", "Bosque Aleatorio")
    modelos      = data.get("modelos", {})

    nombres     = list(modelos.keys())
    comparacion = {
        "nombres":   nombres,
        "accuracy":  [modelos[n]["accuracy"]  for n in nombres],
        "precision": [modelos[n]["precision"] for n in nombres],
        "recall":    [modelos[n]["recall"]    for n in nombres],
        "f1":        [modelos[n]["f1"]        for n in nombres],
        "auc_roc":   [modelos[n]["auc_roc"]   for n in nombres],
        "cv_f1":     [modelos[n].get("cv_f1_mean") or 0 for n in nombres],
    }

    roc_curves = {}
    for nombre, m in modelos.items():
        if "curva_roc" in m:
            roc_curves[nombre] = {
                "fpr":     m["curva_roc"]["fpr"],
                "tpr":     m["curva_roc"]["tpr"],
                "auc_roc": m["auc_roc"],
            }

    feature_importance      = modelos.get(mejor_nombre, {}).get("feature_importance", {})
    feature_importance_list = [
        {"variable": k, "importancia": v}
        for k, v in sorted(
            feature_importance.items(), key=lambda x: x[1], reverse=True
        )
    ]

    confusion_matrices = {}
    for nombre, m in modelos.items():
        cm = m.get("confusion_matrix", {})
        confusion_matrices[nombre] = {
            "matrix": cm.get("matrix", [[0, 0], [0, 0]]),
            "tp": cm.get("tp", 0), "fp": cm.get("fp", 0),
            "tn": cm.get("tn", 0), "fn": cm.get("fn", 0),
        }

    por_clase = {nombre: m.get("por_clase", {}) for nombre, m in modelos.items()}

    hipotesis = {
        "recall_minimo":  data.get("recall_minimo_hipotesis", cfg["params"]["recall_minimo"]),
        "verificada":     data.get("hipotesis_verificada", False),
        "mejor_modelo":   mejor_nombre,
        "mejor_recall":   modelos.get(mejor_nombre, {}).get("recall", 0),
        "intervalos":     data.get("meta", {}).get("hipotesis_intervalos", {
            "temp_aire_K":      "> 299.10",
            "temp_proceso_K":   "> 309.50",
            "vel_rotacion_rpm": "< 1421.50",
            "torque_Nm":        "> 45.95",
            "desgaste_min":     "> 84.50",
        }),
    }

    dataset_info = {
        "nombre":           "AI4I 2020 Predictive Maintenance Dataset",
        "doi":              "https://doi.org/10.24432/C5HS5C",
        "total_registros":  10000,
        "total_fallas":     339,
        "tasa_falla_pct":   3.39,
        "modos_falla":      ["TWF", "HDF", "PWF", "OSF", "RNF"],
        "frecuencias_modo": {"TWF": 46, "HDF": 115, "PWF": 95, "OSF": 98, "RNF": 19},
    }

    payload = {
        "generado_en":        datetime.now().isoformat(),
        "version":            "1.0",
        "meta":               data.get("meta", {}),
        "dataset":            dataset_info,
        "mejor_modelo":       mejor_nombre,
        "hipotesis":          hipotesis,
        "modelos": {
            n: {
                "accuracy":   m["accuracy"],
                "precision":  m["precision"],
                "recall":     m["recall"],
                "f1":         m["f1"],
                "auc_roc":    m["auc_roc"],
                "cv_f1_mean": m.get("cv_f1_mean"),
                "cv_f1_std":  m.get("cv_f1_std"),
                "hipotesis":  m.get("hipotesis", {}),
            }
            for n, m in modelos.items()
        },
        "comparacion":        comparacion,
        "roc_curves":         roc_curves,
        "feature_importance": feature_importance_list,
        "confusion_matrices": confusion_matrices,
        "por_clase":          por_clase,
    }

    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"  [OK] Dashboard JSON guardado en: {ruta}")
    return ruta




# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = _parse_args()

    print("\n" + "=" * 65)
    print("  05_save_results.py")
    print("  Consolidacion Final de Resultados — AI4I 2020 (Matzka, 2020)")
    print("=" * 65)

    print("\n[1/4] Cargando metricas consolidadas...")
    data         = cargar_metrics_json(args.metrics)
    mejor_nombre = data.get("mejor_modelo", "N/A")
    print(f"  Mejor modelo: {mejor_nombre}")
    print(f"  Modelos disponibles: {list(data.get('modelos', {}).keys())}")

    print("\n[2/4] Generando reporte de texto...")
    ruta_txt = generar_reporte_texto(
        data, carpeta=str(Path(args.output) / "reports")
    )

    print("\n[3/4] Exportando metricas a CSV...")
    ruta_csv = generar_csv_metricas(
        data, carpeta=str(Path(args.output) / "reports")
    )

    print("\n[4/4] Generando dashboard_data.json para React...")
    ruta_dash = generar_dashboard_json(data, carpeta=args.dashboard_output)

    modelos = data.get("modelos", {})
    print("\n" + "=" * 65)
    print("  PIPELINE COMPLETO — RESUMEN FINAL")
    print("=" * 65)
    print(f"\n  Dataset  : AI4I 2020 (Matzka, 2020) | 10,000 registros")
    print(f"  Modelos  : {', '.join(modelos.keys())}")
    print(f"\n  {'Modelo':<28} {'Recall':>8} {'AUC':>8} {'F1':>8} {'Hipotesis':>12}")
    print("  " + "-" * 68)
    for nombre, m in modelos.items():
        est   = "NO FALSADA" if m["hipotesis"]["verificada"] else "FALSADA"
        marca = " ★" if nombre == mejor_nombre else "  "
        print(f"  {nombre + marca:<30} "
              f"{m['recall']:>8.4f} {m['auc_roc']:>8.4f} "
              f"{m['f1']:>8.4f} {est:>12}")

    print(f"\n  Archivos generados:")
    print(f"    {ruta_txt}")
    print(f"    {ruta_csv}")
    print(f"    {ruta_dash}")
    print(f"\n 🚀 Siguiente paso: python src/06_html_dashboard.py")
    print(f"    Luego abre dashboard/dashboard.html en tu navegador.\n")
    


if __name__ == "__main__":
    main()




