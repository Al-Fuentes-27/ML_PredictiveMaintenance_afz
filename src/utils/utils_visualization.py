"""
utils/visualization.py
======================
Funciones reutilizables para la generación de todas las figuras
del proyecto de mantenimiento predictivo.

Figuras de análisis exploratorio (EDA):
    graficar_distribucion_clases(df, ruta_salida)  → Figura 1
    graficar_histogramas(df, ruta_salida)           → Figura 2
    graficar_boxplots(df, ruta_salida)              → Figura 3
    graficar_correlacion(df, ruta_salida)           → Figura 4
    graficar_dispersion(df, ruta_salida)            → Figura 5
    graficar_tipo_producto(df, ruta_salida)         → Figura 6

Figuras de evaluación de modelos:
    graficar_preparacion(y_train, y_train_b, ruta)  → Figura 7
    graficar_confusion(resultados, ruta_salida)     → Figura 8
    graficar_metricas_comparacion(res, ruta)        → Figura 9
    graficar_roc(resultados, ruta_salida)           → Figura 10
    graficar_importancia_variables(res, ruta)       → Figura 11
    graficar_reporte_clasificacion(res, ruta)       → Figura 12

Dependencias: matplotlib 3.10.8, seaborn 0.13.2, numpy 2.4.4

Referencia del dataset:
    Matzka, S. (2020). AI4I 2020 Predictive Maintenance Dataset.
    UCI Machine Learning Repository. https://doi.org/10.24432/C5HS5C
"""

from pathlib import Path

import numpy  as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats

# ┌─────────────────────────────────────────────────────────────────────────┐
# │  CONFIGURACIÓN CENTRALIZADA                                             │
# │  La ruta a config.json está definida en utils/config.py → CONFIG_PATH  │
from utils.utils_config import cfg                                               #│
# └─────────────────────────────────────────────────────────────────────────┘

# Ruta de figuras leída desde config.json — usada como default en las 12 funciones
_FIGURES_DIR: str = cfg["paths"]["figures_dir"]


# ── Paleta de colores del proyecto ───────────────────────────────────────────
C = {
    "azul":    "#2196F3",
    "rojo":    "#F44336",
    "verde":   "#4CAF50",
    "naranja": "#FF9800",
    "morado":  "#9C27B0",
    "gris":    "#607D8B",
}

MODELOS_COLORES = {
    "Árbol de Decisión":   C["naranja"],
    "Bosque Aleatorio":    C["azul"],
    "Regresión Logística": C["verde"],
}

PALETTE_MODO = [C["gris"], C["naranja"], C["rojo"], C["morado"], C["azul"], C["verde"]]


# ── Configuración global ──────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":         10,
    "axes.titlesize":    12,
    "axes.titleweight": "bold",
    "axes.labelsize":    10,
    "figure.dpi":        150,
    "axes.grid":         True,
    "grid.alpha":        0.3,
})

VARS_OP = ["temp_aire", "temp_proceso", "vel_rotacion", "torque", "desgaste"]
LABELS  = {
    "temp_aire":    "Temp. Aire (K)",
    "temp_proceso": "Temp. Proceso (K)",
    "vel_rotacion": "Vel. Rotación (rpm)",
    "torque":       "Torque (Nm)",
    "desgaste":     "Desgaste (min)",
}

MODOS_FALLA = ["TWF", "HDF", "PWF", "OSF", "RNF"]


def _guardar(fig, ruta: str, nombre: str) -> None:
    """Guarda la figura y cierra para liberar memoria."""
    Path(ruta).parent.mkdir(parents=True, exist_ok=True)
    ruta_completa = str(Path(ruta) / nombre) if Path(ruta).is_dir() else ruta
    fig.savefig(ruta_completa, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  [✓] {nombre} → {ruta_completa}")


# ════════════════════════════════════════════════════════════════
# FIGURAS DE ANÁLISIS EXPLORATORIO (EDA)
# ════════════════════════════════════════════════════════════════

def graficar_distribucion_clases(df, ruta_salida: str = _FIGURES_DIR) -> None:
    """
    Figura 1 — Distribución de Clases y Modos de Falla.

    Parámetros
    ----------
    df          : pd.DataFrame — Dataset cargado con cargar_datos().
    ruta_salida : str          — Carpeta donde se guarda la figura.
                                 Por defecto: config.json → paths.figures_dir
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        "Figura 1 — Distribución de Clases y Modos de Falla\n"
        "AI4I 2020 Predictive Maintenance Dataset (Matzka, 2020)",
        fontweight="bold",
    )

    n_normal = (df["falla"] == 0).sum()
    n_falla  = (df["falla"] == 1).sum()
    axes[0].pie(
        [n_normal, n_falla],
        labels=[f"Normal\n({n_normal/len(df)*100:.2f}%)",
                f"Falla\n({n_falla/len(df)*100:.2f}%)"],
        colors=[C["azul"], C["rojo"]],
        autopct="%1.2f%%", startangle=90, explode=(0, 0.08),
    )
    axes[0].set_title(f"Distribución Global (n={len(df):,})")

    conteos = [df[m].sum() for m in MODOS_FALLA]
    bars    = axes[1].bar(MODOS_FALLA, conteos,
                          color=PALETTE_MODO[1:], edgecolor="white")
    for bar, c in zip(bars, conteos):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{c}\n({c/len(df)*100:.2f}%)",
            ha="center", fontsize=9, fontweight="bold",
        )
    axes[1].set_title("Frecuencia por Modo de Falla")
    axes[1].set_ylabel("Número de Casos")
    axes[1].set_ylim(0, max(conteos) * 1.3)

    plt.tight_layout()
    _guardar(fig, ruta_salida, "fig1_distribucion_clases.png")


def graficar_histogramas(df, ruta_salida: str = _FIGURES_DIR) -> None:
    """Figura 2 — Histogramas de variables operativas: Normal vs. Falla.
    Por defecto: config.json → paths.figures_dir
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle(
        "Figura 2 — Distribución Estadística de Variables Operativas\n"
        "Comparación Normal vs. Falla — AI4I 2020 (Matzka, 2020)",
        fontweight="bold",
    )
    axes_flat = axes.flatten()
    for i, v in enumerate(VARS_OP):
        ax     = axes_flat[i]
        data_n = df[df["falla"] == 0][v]
        data_f = df[df["falla"] == 1][v]
        ax.hist(data_n, bins=40, alpha=0.6, color=C["azul"],
                label="Normal", density=True)
        ax.hist(data_f, bins=40, alpha=0.7, color=C["rojo"],
                label="Falla", density=True)
        ax.axvline(data_n.mean(), color=C["azul"], linestyle="--",
                   linewidth=1.5, label=f"μ_N={data_n.mean():.1f}")
        ax.axvline(data_f.mean(), color=C["rojo"], linestyle="--",
                   linewidth=1.5, label=f"μ_F={data_f.mean():.1f}")
        _, p = stats.ttest_ind(data_n, data_f)
        ax.set_title(f"{LABELS[v]}\n(p<0.001 ***)" if p < 0.001
                     else f"{LABELS[v]}\n(p={p:.3f})")
        ax.set_xlabel(LABELS[v])
        ax.set_ylabel("Densidad")
        ax.legend(fontsize=8)

    axes_flat[5].set_visible(False)
    plt.tight_layout()
    _guardar(fig, ruta_salida, "fig2_histogramas_variables.png")


def graficar_boxplots(df, ruta_salida: str = _FIGURES_DIR) -> None:
    """Figura 3 — Boxplots por estado del equipo.
    Por defecto: config.json → paths.figures_dir
    """
    fig, axes = plt.subplots(1, 5, figsize=(18, 6))
    fig.suptitle(
        "Figura 3 — Comportamiento de Variables por Estado del Equipo\n"
        "AI4I 2020 Predictive Maintenance Dataset (Matzka, 2020)",
        fontweight="bold",
    )
    for i, v in enumerate(VARS_OP):
        ax        = axes[i]
        data_plot = [df[df["falla"] == 0][v].values,
                     df[df["falla"] == 1][v].values]
        bp = ax.boxplot(
            data_plot, patch_artist=True,
            medianprops=dict(color="white", linewidth=2),
            flierprops=dict(marker=".", markersize=2, alpha=0.3),
        )
        bp["boxes"][0].set_facecolor(C["azul"])
        bp["boxes"][1].set_facecolor(C["rojo"])
        ax.set_xticklabels(["Normal", "Falla"], fontweight="bold")
        ax.set_title(LABELS[v], fontsize=9)
        ax.set_ylabel("Valor")
        for j, dataset in enumerate(data_plot):
            med = np.median(dataset)
            ax.text(j + 1, med, f" {med:.1f}", va="center",
                    fontsize=8, color="white", fontweight="bold")

    plt.tight_layout()
    _guardar(fig, ruta_salida, "fig3_boxplots.png")


def graficar_correlacion(df, ruta_salida: str = _FIGURES_DIR) -> None:
    """Figura 4 — Matriz de correlación y correlación con variable objetivo.
    Por defecto: config.json → paths.figures_dir
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "Figura 4 — Matriz de Correlación y Relación con Variable Objetivo\n"
        "AI4I 2020 Predictive Maintenance Dataset (Matzka, 2020)",
        fontweight="bold",
    )
    corr_vars = VARS_OP + ["falla"]
    corr_mat  = df[corr_vars].corr()
    labels_s  = ["T_Aire", "T_Proc.", "Vel.Rot.", "Torque", "Desgaste", "Falla"]
    corr_mat.columns = labels_s
    corr_mat.index   = labels_s

    sns.heatmap(
        corr_mat, ax=axes[0], annot=True, fmt=".3f", cmap="RdBu_r",
        center=0, vmin=-1, vmax=1, square=True, linewidths=0.5,
        cbar_kws={"shrink": 0.8},
    )
    axes[0].set_title("Correlación de Pearson (variables + falla)")

    corr_falla = df[VARS_OP].corrwith(df["falla"]).sort_values()
    colors_bar = [C["rojo"] if c > 0 else C["azul"] for c in corr_falla]
    bars = axes[1].barh(
        [LABELS[v] for v in corr_falla.index],
        corr_falla.values, color=colors_bar, edgecolor="white",
    )
    for bar, val in zip(bars, corr_falla.values):
        axes[1].text(
            val + 0.003 if val >= 0 else val - 0.003,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}", va="center",
            ha="left" if val >= 0 else "right",
            fontsize=9, fontweight="bold",
        )
    axes[1].axvline(0, color="black", linewidth=0.8)
    axes[1].set_xlabel("Coeficiente de Correlación con Machine Failure")
    axes[1].set_title("Correlación con Variable Objetivo")

    plt.tight_layout()
    _guardar(fig, ruta_salida, "fig4_correlacion.png")


def graficar_dispersion(df, ruta_salida: str = _FIGURES_DIR) -> None:
    """Figura 5 — Dispersión de variables clave por estado y modo de falla.
    Por defecto: config.json → paths.figures_dir
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "Figura 5 — Variables Clave y Evidencia de Degradación Detectable\n"
        "AI4I 2020 Predictive Maintenance Dataset (Matzka, 2020)",
        fontweight="bold",
    )
    for estado, color, label, s, a in [
        (0, C["azul"], "Normal",  6,  0.3),
        (1, C["rojo"], "Falla",  12,  0.7),
    ]:
        sub = df[df["falla"] == estado]
        axes[0].scatter(sub["vel_rotacion"], sub["torque"],
                        c=color, alpha=a, s=s, label=label)
    axes[0].set_xlabel("Velocidad de Rotación (rpm)")
    axes[0].set_ylabel("Torque (Nm)")
    axes[0].set_title("Torque vs. Velocidad por Estado")
    axes[0].legend(markerscale=3)

    df = df.copy()
    df["modo_idx"] = 0
    for j, m in enumerate(MODOS_FALLA):
        df.loc[df[m] == 1, "modo_idx"] = j + 1
    etiq_modo = ["Normal"] + MODOS_FALLA

    for idx in range(len(etiq_modo)):
        sub = df[df["modo_idx"] == idx]
        axes[1].scatter(
            sub["desgaste"], sub["temp_proceso"],
            c=PALETTE_MODO[idx],
            alpha=0.2 if idx == 0 else 0.8,
            s=5 if idx == 0 else 18,
            label=etiq_modo[idx],
        )
    axes[1].set_xlabel("Desgaste de Herramienta (min)")
    axes[1].set_ylabel("Temperatura de Proceso (K)")
    axes[1].set_title("Desgaste vs. Temperatura por Modo de Falla")
    axes[1].legend(fontsize=8, markerscale=2)

    plt.tight_layout()
    _guardar(fig, ruta_salida, "fig5_dispersion.png")


def graficar_tipo_producto(df, ruta_salida: str = _FIGURES_DIR) -> None:
    """Figura 6 — Tasa de falla y distribución de desgaste por tipo de producto.
    Por defecto: config.json → paths.figures_dir
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle(
        "Figura 6 — Análisis por Tipo de Producto\n"
        "AI4I 2020 Predictive Maintenance Dataset (Matzka, 2020)",
        fontweight="bold",
    )
    tipo_stats   = df.groupby("Type")["falla"].agg(["sum", "count"])
    tipo_stats["tasa"] = tipo_stats["sum"] / tipo_stats["count"] * 100
    colores_tipo = [C["naranja"], C["verde"], C["rojo"]]

    bars = axes[0].bar(
        tipo_stats.index, tipo_stats["tasa"],
        color=colores_tipo, edgecolor="white",
    )
    for bar, (idx, row) in zip(bars, tipo_stats.iterrows()):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            f"{row['tasa']:.2f}%\n({int(row['sum'])}/{int(row['count'])})",
            ha="center", fontsize=9, fontweight="bold",
        )
    axes[0].set_xlabel("Tipo (H=Alta, M=Media, L=Baja Calidad)")
    axes[0].set_ylabel("Tasa de Falla (%)")
    axes[0].set_title("Tasa de Falla por Tipo")

    for tipo, color in zip(["L", "M", "H"], colores_tipo):
        sub = df[(df["Type"] == tipo) & (df["falla"] == 1)]
        axes[1].hist(sub["desgaste"], bins=30, alpha=0.6,
                     color=color, label=f"Tipo {tipo}", density=True)
    axes[1].set_xlabel("Desgaste de Herramienta en Falla (min)")
    axes[1].set_ylabel("Densidad")
    axes[1].set_title("Desgaste al Momento de Falla")
    axes[1].legend()

    plt.tight_layout()
    _guardar(fig, ruta_salida, "fig6_tipo_producto.png")


# ════════════════════════════════════════════════════════════════
# FIGURAS DE EVALUACIÓN DE MODELOS
# ════════════════════════════════════════════════════════════════

def graficar_preparacion(
    y_train:      np.ndarray,
    y_train_bal:  np.ndarray,
    X_train_size: int,
    X_test_size:  int,
    ruta_salida:  str = _FIGURES_DIR,
) -> None:
    """Figura 7 — División del dataset y efecto de SMOTE.
    Por defecto: config.json → paths.figures_dir
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        "Figura 7 — Preparación: División y Balance de Clases\n"
        "Metodología CRISP-DM — AI4I 2020 (Matzka, 2020)",
        fontweight="bold",
    )
    axes[0].pie(
        [X_train_size, X_test_size],
        labels=[f"Entrenamiento 80%\nn={X_train_size:,}",
                f"Prueba 20%\nn={X_test_size:,}"],
        colors=[C["azul"], C["naranja"]],
        autopct="%1.1f%%", startangle=90, explode=(0, 0.05),
    )
    axes[0].set_title("División Estratificada")

    antes = [sum(y_train == 0), sum(y_train == 1)]
    axes[1].bar(["Normal", "Falla"], antes, color=[C["azul"], C["rojo"]], edgecolor="white")
    for i, v in enumerate(antes):
        axes[1].text(i, v + 20, f"{v:,}\n({v/len(y_train)*100:.1f}%)",
                     ha="center", fontsize=9, fontweight="bold")
    axes[1].set_title("ANTES de SMOTE")
    axes[1].set_ylabel("Registros")

    despues = [sum(y_train_bal == 0), sum(y_train_bal == 1)]
    axes[2].bar(["Normal", "Falla"], despues, color=[C["azul"], C["verde"]], edgecolor="white")
    for i, v in enumerate(despues):
        axes[2].text(i, v + 20, f"{v:,}\n(50.0%)",
                     ha="center", fontsize=9, fontweight="bold")
    axes[2].set_title("DESPUÉS de SMOTE")
    axes[2].set_ylabel("Registros")

    plt.tight_layout()
    _guardar(fig, ruta_salida, "fig7_preparacion_dataset.png")


def graficar_confusion(
    resultados:  dict,
    ruta_salida: str = _FIGURES_DIR,
) -> None:
    """Figura 8 — Matrices de confusión para cada modelo.
    Por defecto: config.json → paths.figures_dir
    """
    fig, axes = plt.subplots(1, len(resultados), figsize=(5 * len(resultados), 5))
    if len(resultados) == 1:
        axes = [axes]
    fig.suptitle(
        "Figura 8 — Matrices de Confusión por Modelo\n"
        "AI4I 2020 Predictive Maintenance Dataset (Matzka, 2020)",
        fontweight="bold",
    )
    for ax, (nombre, m) in zip(axes, resultados.items()):
        cm = m["confusion_matrix"]["matrix"]
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["Pred. Normal", "Pred. Falla"],
            yticklabels=["Real Normal", "Real Falla"],
            linewidths=0.5, cbar=False,
            annot_kws={"size": 14, "weight": "bold"},
        )
        tn = m["confusion_matrix"]["tn"]; fp = m["confusion_matrix"]["fp"]
        fn = m["confusion_matrix"]["fn"]; tp = m["confusion_matrix"]["tp"]
        ax.set_title(f"{nombre}\nF1={m['f1']:.3f} | AUC={m['auc_roc']:.3f}")
        ax.set_xlabel("Predicción"); ax.set_ylabel("Valor Real")
        ax.text(0.5, -0.22, f"VP={tp} | FP={fp} | VN={tn} | FN={fn}",
                ha="center", transform=ax.transAxes, fontsize=8, color=C["gris"])

    plt.tight_layout()
    _guardar(fig, ruta_salida, "fig8_matrices_confusion.png")


def graficar_metricas_comparacion(
    resultados:  dict,
    ruta_salida: str = _FIGURES_DIR,
) -> None:
    """Figura 9 — Comparación de métricas y validación cruzada.
    Por defecto: config.json → paths.figures_dir
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "Figura 9 — Comparación de Métricas de Desempeño\n"
        "AI4I 2020 Predictive Maintenance Dataset (Matzka, 2020)",
        fontweight="bold",
    )
    nombres  = list(resultados.keys())
    metricas = ["accuracy", "precision", "recall", "f1"]
    etiq_met = ["Accuracy", "Precision", "Recall", "F1-Score"]
    x        = np.arange(len(nombres))
    width    = 0.2
    colores  = [C["azul"], C["verde"], C["naranja"], C["rojo"]]

    for i, (met, et, col) in enumerate(zip(metricas, etiq_met, colores)):
        vals = [resultados[n][met] for n in nombres]
        bars = axes[0].bar(x + i * width, vals, width, label=et,
                           color=col, alpha=0.85, edgecolor="white")
        for bar, v in zip(bars, vals):
            axes[0].text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() + 0.005, f"{v:.3f}",
                         ha="center", fontsize=7.5, fontweight="bold")
    axes[0].set_xticks(x + width * 1.5)
    axes[0].set_xticklabels(nombres, fontsize=9)
    axes[0].set_ylim(0, 1.12)
    axes[0].set_ylabel("Valor de la Métrica")
    axes[0].set_title("Accuracy / Precision / Recall / F1-Score")
    axes[0].legend(loc="upper right", fontsize=9)

    cv_means = [resultados[n]["cv_f1_mean"] or 0 for n in nombres]
    cv_stds  = [resultados[n]["cv_f1_std"]  or 0 for n in nombres]
    cols_cv  = [MODELOS_COLORES.get(n, C["gris"]) for n in nombres]
    bars = axes[1].bar(nombres, cv_means, color=cols_cv, alpha=0.85,
                       edgecolor="white", yerr=cv_stds, capsize=8,
                       error_kw={"linewidth": 2, "color": "black"})
    for bar, v, s in zip(bars, cv_means, cv_stds):
        axes[1].text(bar.get_x() + bar.get_width() / 2, v + s + 0.01,
                     f"{v:.3f}\n±{s:.3f}", ha="center", fontsize=9, fontweight="bold")
    axes[1].set_ylabel("F1-Score Promedio")
    axes[1].set_ylim(0, 1.1)
    axes[1].set_title("Validación Cruzada 5-Fold\nF1-Score ± Desviación Estándar")

    plt.tight_layout()
    _guardar(fig, ruta_salida, "fig9_comparacion_metricas.png")


def graficar_roc(
    resultados:  dict,
    ruta_salida: str = _FIGURES_DIR,
) -> None:
    """Figura 10 — Curvas ROC superpuestas por modelo.
    Por defecto: config.json → paths.figures_dir
    """
    fig, ax = plt.subplots(figsize=(8, 7))
    fig.suptitle(
        "Figura 10 — Curvas ROC: Capacidad Discriminante por Modelo\n"
        "AI4I 2020 Predictive Maintenance Dataset (Matzka, 2020)",
        fontweight="bold",
    )
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.5,
            label="Clasificador Aleatorio (AUC = 0.50)", alpha=0.6)

    for nombre, m in resultados.items():
        fpr = m["curva_roc"]["fpr"]
        tpr = m["curva_roc"]["tpr"]
        ax.plot(fpr, tpr, linewidth=2.5,
                color=MODELOS_COLORES.get(nombre, C["gris"]),
                label=f"{nombre} (AUC = {m['auc_roc']:.4f})")

    mejor = max(resultados, key=lambda k: resultados[k]["auc_roc"])
    ax.fill_between(
        resultados[mejor]["curva_roc"]["fpr"],
        resultados[mejor]["curva_roc"]["tpr"],
        alpha=0.07, color=C["azul"],
    )
    ax.set_xlabel("Tasa de Falsos Positivos (1 − Especificidad)")
    ax.set_ylabel("Tasa de Verdaderos Positivos (Recall)")
    ax.set_title("Curvas ROC — Comparación de Modelos")
    ax.legend(loc="lower right", fontsize=10)
    ax.set_xlim([-0.01, 1.01]); ax.set_ylim([-0.01, 1.01])

    plt.tight_layout()
    _guardar(fig, ruta_salida, "fig10_curvas_roc.png")


def graficar_importancia_variables(
    resultados:    dict,
    nombre_modelo: str,
    feature_names: list,
    ruta_salida:   str = _FIGURES_DIR,
) -> None:
    """Figura 11 — Importancia de variables del modelo seleccionado.
    Por defecto: config.json → paths.figures_dir
    """
    if nombre_modelo not in resultados:
        print(f"  [!] Modelo '{nombre_modelo}' no encontrado en resultados.")
        return

    importancias_dict = resultados[nombre_modelo].get("feature_importance", {})
    if not importancias_dict:
        print(f"  [!] '{nombre_modelo}' no tiene feature_importance.")
        return

    labels       = list(importancias_dict.keys())
    importancias = np.array(list(importancias_dict.values()))
    idx_sorted   = np.argsort(importancias)
    max_idx      = np.argmax(importancias)
    colores_sort = [C["rojo"] if idx_sorted[i] == max_idx else C["azul"]
                    for i in range(len(idx_sorted))]

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(
        f"Figura 11 — Importancia de Variables: {nombre_modelo}\n"
        "AI4I 2020 Predictive Maintenance Dataset (Matzka, 2020)",
        fontweight="bold",
    )
    ax.barh([labels[i] for i in idx_sorted],
            importancias[idx_sorted],
            color=colores_sort, edgecolor="white")
    for i, (v, idx) in enumerate(zip(importancias[idx_sorted], idx_sorted)):
        ax.text(v + 0.002, i, f"{v:.4f}", va="center", fontsize=9,
                fontweight="bold" if idx == max_idx else "normal")
    ax.set_xlabel("Importancia (Gini Impurity Reduction)")
    ax.set_title(f"Importancia de Variables — {nombre_modelo}")
    ax.set_xlim(0, max(importancias) * 1.2)

    plt.tight_layout()
    _guardar(fig, ruta_salida, "fig11_importancia_variables.png")


def graficar_reporte_clasificacion(
    resultados:  dict,
    ruta_salida: str = _FIGURES_DIR,
) -> None:
    """Figura 12 — Heatmap de Precision/Recall/F1 por clase y modelo.
    Por defecto: config.json → paths.figures_dir
    """
    fig, axes = plt.subplots(1, len(resultados), figsize=(5 * len(resultados) + 1, 5))
    if len(resultados) == 1:
        axes = [axes]
    fig.suptitle(
        "Figura 12 — Reporte de Clasificación por Modelo\n"
        "Precision, Recall y F1-Score por Clase (Matzka, 2020)",
        fontweight="bold",
    )
    clases = ["Normal", "Falla"]
    mets   = ["precision", "recall", "f1"]
    etq    = ["Precision", "Recall", "F1-Score"]

    for ax, (nombre, m) in zip(axes, resultados.items()):
        data = np.array([
            [m["por_clase"][c][met] for met in mets]
            for c in clases
        ])
        im = ax.imshow(data, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(etq))); ax.set_xticklabels(etq, fontsize=10)
        ax.set_yticks(range(len(clases)))
        ax.set_yticklabels(clases, fontsize=10, fontweight="bold")
        for i in range(len(clases)):
            for j in range(len(mets)):
                val = data[i, j]
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=13, fontweight="bold",
                        color="white" if val > 0.6 else "black")

        # ==================
        cv_val = m.get('cv_f1_mean')
        if cv_val is None:
            cv_val = 0.0
        ax.set_title(f"{nombre}\nAUC={m['auc_roc']:.3f} | "
                     f"CV-F1={cv_val:.3f}")
        # ==================



    plt.tight_layout()
    _guardar(fig, ruta_salida, "fig12_reporte_clasificacion.png")
