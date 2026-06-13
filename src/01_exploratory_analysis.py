"""
01_exploratory_analysis.py
==========================
Análisis Exploratorio de Datos (EDA) del AI4I 2020 Predictive Maintenance Dataset.

Genera las Figuras 1 a 6 del informe técnico y un reporte estadístico
en consola con todas las métricas descriptivas e inferenciales.

Uso:
    python src/01_exploratory_analysis.py
    python src/01_exploratory_analysis.py --data data/raw/dataset_ai4i2020.csv
    python src/01_exploratory_analysis.py --output results/figures

Salidas:
    results/figures/fig1_distribucion_clases.png
    results/figures/fig2_histogramas_variables.png
    results/figures/fig3_boxplots.png
    results/figures/fig4_correlacion.png
    results/figures/fig5_dispersion.png
    results/figures/fig6_tipo_producto.png
    results/eda_estadisticas_descriptivas.csv

Dependencias: pandas 3.0.2, numpy 2.4.4, matplotlib 3.10.8,
              seaborn 0.13.2, scipy 1.17.1

Referencias:
    Matzka, S. (2020). AI4I 2020 Predictive Maintenance Dataset.
    UCI Machine Learning Repository. https://doi.org/10.24432/C5HS5C

    Carvalho, T. P. et al. (2021). Predictive maintenance in the
    automotive sector. Algorithms, 27(1), 2.
    https://doi.org/10.3390/mca27010002

Autor: Aldo Fuentes Zaldívar — 2025-2026
"""

import sys
import argparse
import warnings
from pathlib import Path

import numpy  as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))

# ┌─────────────────────────────────────────────────────────────────────────┐
# │  CONFIGURACIÓN CENTRALIZADA                                             │
# │  La ruta a config.json está definida en utils/config.py → CONFIG_PATH  │
from utils.utils_config import cfg                                               #│
# └─────────────────────────────────────────────────────────────────────────┘

from utils.utils_preprocessing import cargar_datos, FEATURE_NAMES, FEATURE_LABELS, FAILURE_MODES
from utils.utils_visualization import (
    graficar_distribucion_clases,
    graficar_histogramas,
    graficar_boxplots,
    graficar_correlacion,
    graficar_dispersion,
    graficar_tipo_producto,
)


# ── CLI ───────────────────────────────────────────────────────────────────────
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="EDA del AI4I 2020 Predictive Maintenance Dataset."
    )
    parser.add_argument(
        "--data",
        default=cfg["paths"]["data"],
        help=f"Ruta al dataset CSV (default: {cfg['paths']['data']})",
    )
    parser.add_argument(
        "--output",
        default=cfg["paths"]["figures_dir"],
        help=f"Carpeta de salida para figuras (default: {cfg['paths']['figures_dir']})",
    )
    return parser.parse_args()


# ── Estadística descriptiva ───────────────────────────────────────────────────
def estadistica_descriptiva(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula y reporta estadísticas por variable para Normal y Falla."""
    normal = df[df["falla"] == 0]
    falla  = df[df["falla"] == 1]

    print("\n" + "=" * 75)
    print("  ESTADÍSTICA DESCRIPTIVA — Variables Operativas")
    print("=" * 75)

    filas = []
    for v in FEATURE_NAMES:
        n = normal[v]; f = falla[v]
        t_stat, p_val = stats.ttest_ind(n, f)
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."

        print(f"\n  {FEATURE_LABELS[v]}")
        print(f"    Normal -> mu={n.mean():.2f}  sigma={n.std():.2f}  "
              f"Q1={n.quantile(0.25):.2f}  Mediana={n.median():.2f}  "
              f"Q3={n.quantile(0.75):.2f}")
        print(f"    Falla  -> mu={f.mean():.2f}  sigma={f.std():.2f}  "
              f"Q1={f.quantile(0.25):.2f}  Mediana={f.median():.2f}  "
              f"Q3={f.quantile(0.75):.2f}")
        print(f"    Prueba t: t={t_stat:.4f}  p={p_val:.6f}  {sig}")

        filas.append({
            "Variable":      FEATURE_LABELS[v],
            "mu_Normal":     round(n.mean(), 2),
            "mu_Falla":      round(f.mean(), 2),
            "Delta_mu":      round(f.mean() - n.mean(), 2),
            "Q1_Normal":     round(n.quantile(0.25), 2),
            "Q1_Falla":      round(f.quantile(0.25), 2),
            "Q3_Normal":     round(n.quantile(0.75), 2),
            "Q3_Falla":      round(f.quantile(0.75), 2),
            "p_valor":       round(p_val, 6),
            "Significancia": sig,
        })
    return pd.DataFrame(filas)


# ── Distribución de clases ───────────────────────────────────────────────────
def analisis_clases(df: pd.DataFrame) -> None:
    n_total  = len(df)
    n_falla  = df["falla"].sum()
    n_normal = n_total - n_falla

    print("\n" + "=" * 75)
    print("  DISTRIBUCIÓN DE CLASES")
    print("=" * 75)
    print(f"  Total registros  : {n_total:,}")
    print(f"  Normal (clase 0) : {n_normal:,}  ({n_normal/n_total*100:.2f}%)")
    print(f"  Falla  (clase 1) : {n_falla:,}   ({n_falla/n_total*100:.2f}%)")
    print(f"  Razon desbalance : {n_normal/n_falla:.1f}:1\n")

    print("  MODOS DE FALLA")
    print("  " + "-" * 45)
    for modo in FAILURE_MODES:
        if modo in df.columns:
            n = df[modo].sum()
            print(f"  {modo:<8}: {n:>4} casos  ({n/n_total*100:.3f}%)")


# ── Correlaciones ─────────────────────────────────────────────────────────────
def analisis_correlaciones(df: pd.DataFrame) -> None:
    print("\n" + "=" * 75)
    print("  CORRELACION CON VARIABLE OBJETIVO (Machine Failure)")
    print("=" * 75)
    corr = df[FEATURE_NAMES].corrwith(df["falla"]).abs().sort_values(ascending=False)
    for v, c in corr.items():
        barra = "X" * int(c * 40)
        print(f"  {FEATURE_LABELS[v]:<28}: r={c:.4f}  {barra}")

    print("\n  CORRELACIONES ENTRE VARIABLES (|r| > 0.3)")
    print("  " + "-" * 55)
    corr_matrix = df[FEATURE_NAMES].corr()
    for i, v1 in enumerate(FEATURE_NAMES):
        for v2 in FEATURE_NAMES[i+1:]:
            c = corr_matrix.loc[v1, v2]
            if abs(c) > 0.3:
                nivel = "alta" if abs(c) > 0.7 else "moderada"
                print(f"  {FEATURE_LABELS[v1]:<28} <-> {FEATURE_LABELS[v2]:<28}: "
                      f"r={c:.4f} ({nivel})")


# ── Intervalos críticos ───────────────────────────────────────────────────────
def intervalos_criticos(df: pd.DataFrame) -> None:
    print("\n" + "=" * 75)
    print("  INTERVALOS CRITICOS POR MODO DE FALLA")
    print("  (Sustento empirico de la hipotesis de solucion)")
    print("=" * 75)

    df = df.copy()
    if "diff_temp" not in df.columns:
        df["diff_temp"]  = df["temp_proceso"] - df["temp_aire"]
        df["potencia_w"] = df["torque"] * df["vel_rotacion"] * (2 * np.pi / 60)

    for modo in FAILURE_MODES:
        if modo not in df.columns:
            continue
        sub = df[df[modo] == 1]
        if len(sub) == 0:
            continue
        print(f"\n  [{modo}]  n={len(sub)}")
        for v in FEATURE_NAMES:
            q1  = sub[v].quantile(0.25)
            med = sub[v].median()
            q3  = sub[v].quantile(0.75)
            print(f"    {FEATURE_LABELS[v]:<28}: Q1={q1:.2f}  Med={med:.2f}  "
                  f"Q3={q3:.2f}  [{sub[v].min():.2f} - {sub[v].max():.2f}]")

    falla_df = df[df["falla"] == 1]
    print("\n  UMBRALES HIPOTESIS GENERAL (Q1 de falla):")
    print("  " + "-" * 50)
    for v in FEATURE_NAMES:
        q1      = falla_df[v].quantile(0.25)
        dir_sym = "<" if v == "vel_rotacion" else ">"
        print(f"  {FEATURE_LABELS[v]:<28}: {dir_sym} {q1:.2f}  [zona de riesgo]")


# ── Guardar reporte CSV ───────────────────────────────────────────────────────
def guardar_reporte(df_stats: pd.DataFrame, ruta: str = None) -> None:
    if ruta is None:
        ruta = cfg["paths"]["results_dir"]
    Path(ruta).mkdir(parents=True, exist_ok=True)
    csv_path = Path(ruta) / Path(cfg["paths"]["metrics"]["eda_csv"]).name
    df_stats.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"\n  [OK] Reporte CSV guardado en: {csv_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    args = _parse_args()

    print("\n" + "=" * 75)
    print("  01_exploratory_analysis.py")
    print("  EDA del AI4I 2020 Predictive Maintenance Dataset (Matzka, 2020)")
    print("=" * 75)

    print("\n[1/4] Cargando dataset...")
    df = cargar_datos(args.data)

    print("\n[2/4] Ejecutando analisis estadistico...")
    analisis_clases(df)
    df_stats = estadistica_descriptiva(df)
    analisis_correlaciones(df)
    intervalos_criticos(df)

    guardar_reporte(df_stats, ruta=str(Path(args.output).parent))

    print("\n[3/4] Generando figuras 1-6...")
    Path(args.output).mkdir(parents=True, exist_ok=True)

    graficar_distribucion_clases(df, args.output)
    graficar_histogramas(df, args.output)
    graficar_boxplots(df, args.output)
    graficar_correlacion(df, args.output)
    graficar_dispersion(df, args.output)
    graficar_tipo_producto(df, args.output)

    print("\n[4/4] Completado.")
    print(f"\n  Figuras guardadas en : {args.output}/")
    print(f"  Siguiente paso       : python src/02_preprocessing.py\n")


if __name__ == "__main__":
    main()
