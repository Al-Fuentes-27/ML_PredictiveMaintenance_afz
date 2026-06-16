"""
06_html_dashboard.py
Generador de Dashboard HTML Interactivo para Portafolio CV.

Lee dashboard_data.json (generado por 05_save_results.py) y construye
un dashboard web autocontenido en un solo archivo HTML, sin necesidad
de frameworks frontend (React/Vue) ni servidores backend (Streamlit/Flask).

Tecnologías: Plotly.js (gráficos interactivos) + Tailwind CSS (estilos).

Mejoras implementadas:
    ✓ 1. Business Insights (Hallazgos de Negocio)
    ✓ 2. Selector dinámico de Matriz de Confusión (JavaScript)
    ✓ 3. Validación Cruzada con barras de error (CV-F1)
    ✓ 4. Botón de Descarga del Informe Técnico (PDF)
    ✓ 5. Footer con enlaces a GitHub, UCI ML, LinkedIn
    ✓ 6. Gráfico de Radar (Perfil Multidimensional)
    ✓ 7. Animaciones Fade-In al hacer Scroll (Intersection Observer)
    ✓ 8. Header Sticky con Navegación por Anclas
    ✓ 9. Tooltips Educativos en Métricas (Contexto de Negocio)
    ✓ 10. Toggle Modo Claro/Oscuro con persistencia (localStorage)

Uso:
    python src/06_html_dashboard.py

Entradas:
    data/dashboard_metrics/dashboard_data.json

Salidas:
    dashboard/dashboard.html

Dependencias: plotly
Autor: Aldo Fuentes Zaldivar — 2025-2026
"""

import sys
import json
import plotly.graph_objects as go
from pathlib import Path

# ── RESOLUCIÓN DE RUTAS Y CONFIGURACIÓN CENTRALIZADA ─────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from utils.utils_config import cfg

DASHBOARD_DATA_DIR = (SCRIPT_DIR / cfg["paths"]["metrics"]["dashboard_data_dir"]).resolve()
DASHBOARD_JSON = DASHBOARD_DATA_DIR / "dashboard_data.json"
OUTPUT_HTML = (SCRIPT_DIR / cfg["paths"]["dashboard_html"]).resolve()


# ── HELPER: Limpiar espacios en blanco en claves JSON ────────────────────────
def clean_keys(obj):
    """Recorre el JSON y elimina espacios en claves y valores string."""
    if isinstance(obj, dict):
        return {k.strip(): clean_keys(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean_keys(i) for i in obj]
    if isinstance(obj, str):
        return obj.strip()
    return obj


# ── HELPER: Convertir Hex a RGBA para Plotly ─────────────────────────────────
def hex_to_rgba(hex_color: str, alpha: float = 0.2) -> str:
    """Convierte un color hex (#rrggbb) a formato rgba() para Plotly."""
    hex_color = hex_color.lstrip('#')
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return f'rgba({r},{g},{b},{alpha})'


# ── GENERACIÓN DE GRÁFICOS PLOTLY ────────────────────────────────────────────
def generate_plots(data: dict, best_model: str) -> dict:
    """Genera los 6 gráficos interactivos y retorna sus divs HTML."""
    plots = {}
    
    # Layout con fondo transparente para adaptarse al tema claro/oscuro
    dark_layout = dict(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e2e8f0'),
        margin=dict(l=20, r=20, t=40, b=20),
        height=380
    )
    colors = ['#3b82f6', '#10b981', '#f59e0b']

    # 1. Curvas ROC (Todos los modelos)
    fig_roc = go.Figure()
    for i, (name, roc) in enumerate(data.get('roc_curves', {}).items()):
        fig_roc.add_trace(go.Scatter(
            x=roc['fpr'], y=roc['tpr'],
            name=f"{name} (AUC={roc['auc_roc']:.3f})",
            line=dict(color=colors[i % 3], width=2)
        ))
    fig_roc.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode='lines',
        line=dict(dash='dash', color='gray'), showlegend=False
    ))
    fig_roc.update_layout(
        **dark_layout,
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        title="Curvas ROC Comparativas"
    )
    plots['roc'] = fig_roc.to_html(
        full_html=False, include_plotlyjs=False, config={'displayModeBar': False}
    )

    # 2. Comparación de Métricas (Bar Chart)
    comp = data.get('comparacion', {})
    fig_comp = go.Figure()
    metrics = ['recall', 'precision', 'f1', 'auc_roc']
    for i, metric in enumerate(metrics):
        fig_comp.add_trace(go.Bar(
            name=metric.upper(),
            x=comp.get('nombres', []),
            y=comp.get(metric, []),
            marker_color=colors[i % 3]
        ))
    fig_comp.update_layout(
        **dark_layout, barmode='group',
        title="Comparación de Métricas por Modelo",
        yaxis_title="Score"
    )
    plots['comp'] = fig_comp.to_html(
        full_html=False, include_plotlyjs=False, config={'displayModeBar': False}
    )

    # 3. Importancia de Variables (Solo mejor modelo)
    fi = data.get('feature_importance', [])
    sorted_fi = sorted(fi, key=lambda x: x['importancia'])
    fig_imp = go.Figure(go.Bar(
        x=[v['importancia'] for v in sorted_fi],
        y=[v['variable'].replace('_', ' ').title() for v in sorted_fi],
        orientation='h', marker_color='#3b82f6'
    ))
    fig_imp.update_layout(
        **dark_layout,
        xaxis_title="Gini Impurity Reduction",
        title=f"Importancia de Variables ({best_model})"
    )
    plots['imp'] = fig_imp.to_html(
        full_html=False, include_plotlyjs=False, config={'displayModeBar': False}
    )

    # 4. Distribución de Modos de Falla
    modes = data.get('dataset', {}).get('frecuencias_modo', {})
    fig_modes = go.Figure(go.Bar(
        x=list(modes.keys()), y=list(modes.values()),
        marker_color='#ef4444'
    ))
    fig_modes.update_layout(
        **dark_layout,
        title="Frecuencia por Modo de Falla (Dataset Original)",
        yaxis_title="Casos"
    )
    plots['modes'] = fig_modes.to_html(
        full_html=False, include_plotlyjs=False, config={'displayModeBar': False}
    )

    # 5. Validación Cruzada (CV-F1) con barras de error
    modelos_data = data.get('modelos', {})
    nombres_cv = list(modelos_data.keys())
    cv_means = [modelos_data[n].get('cv_f1_mean', 0) or 0 for n in nombres_cv]
    cv_stds = [modelos_data[n].get('cv_f1_std', 0) or 0 for n in nombres_cv]

    fig_cv = go.Figure()
    fig_cv.add_trace(go.Bar(
        x=nombres_cv,
        y=cv_means,
        error_y=dict(
            type='data', array=cv_stds, visible=True,
            color='#f59e0b', thickness=2, width=8
        ),
        marker_color=[
            '#3b82f6' if n == best_model else '#475569'
            for n in nombres_cv
        ],
        text=[f"{m:.4f}" for m in cv_means],
        textposition='outside',
        textfont=dict(color='#e2e8f0', size=12)
    ))
    fig_cv.update_layout(
        **dark_layout,
        title="Validación Cruzada 5-Fold (F1-Score)",
        yaxis_title="F1-Score Promedio",
        yaxis_range=[0, 1.15]
    )
    plots['cv'] = fig_cv.to_html(
        full_html=False, include_plotlyjs=False, config={'displayModeBar': False}
    )

    # 6. Gráfico de Radar (Spider Chart)
    modelos_info = data.get('modelos', {})
    nombres_modelos = list(modelos_info.keys())
    metricas_radar = ['accuracy', 'recall', 'precision', 'f1', 'auc_roc']
    metricas_labels = ['Accuracy', 'Recall', 'Precision', 'F1-Score', 'AUC-ROC']

    fig_radar = go.Figure()
    for i, nombre in enumerate(nombres_modelos):
        m = modelos_info[nombre]
        valores = [m.get(metric, 0) for metric in metricas_radar]
        valores.append(valores[0])
        theta_cerrado = metricas_labels + [metricas_labels[0]]

        fig_radar.add_trace(go.Scatterpolar(
            r=valores,
            theta=theta_cerrado,
            fill='toself',
            name=nombre,
            line=dict(color=colors[i % 3], width=2),
            fillcolor=hex_to_rgba(colors[i % 3], 0.2)
        ))

    fig_radar.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e2e8f0'),
        polar=dict(
            radialaxis=dict(
                visible=True, range=[0, 1],
                tickfont=dict(size=10), gridcolor='#334155'
            ),
            angularaxis=dict(
                tickfont=dict(size=11, color='#e2e8f0'),
                gridcolor='#334155'
            ),
            bgcolor='rgba(0,0,0,0)'
        ),
        title="Perfil Multidimensional de Modelos",
        height=420,
        margin=dict(l=60, r=60, t=50, b=40),
        showlegend=True,
        legend=dict(
            orientation="h", yanchor="bottom",
            y=-0.15, xanchor="center", x=0.5
        )
    )
    plots['radar'] = fig_radar.to_html(
        full_html=False, include_plotlyjs=False, config={'displayModeBar': False}
    )

    return plots


# ── CONSTRUCCIÓN DEL HTML ────────────────────────────────────────────────────
def build_html(data: dict, plots: dict) -> str:
    """Construye la plantilla HTML final con todas las secciones."""
    ds = data.get('dataset', {})
    hip = data.get('hipotesis', {})
    is_verified = hip.get('verificada', False)
    best_model = data.get('mejor_modelo', 'N/A')
    best_recall = hip.get('mejor_recall', 0)

    badge_text = "✅ HIPÓTESIS NO FALSADA" if is_verified else "❌ HIPÓTESIS FALSADA"
    badge_color = "bg-green-600" if is_verified else "bg-red-600"

    # ── Intervalos de Riesgo ─────────────────────────────────────────────────
    intervalos = hip.get('intervalos', {})
    intervalos_html = "".join([
        f"<div class='interval-box p-2 rounded text-center'>"
        f"<span class='interval-label text-xs'>{k.replace('_', ' ').title()}</span><br>"
        f"<span class='interval-value font-bold'>{v}</span></div>"
        for k, v in intervalos.items() if k != 'recall_minimo'
    ])

    # ── Tooltips Educativos ──────────────────────────────────────────────────
    tooltips = {
        'accuracy': 'Proporción total de predicciones correctas. Puede ser engañosa cuando hay desbalance de clases (96% Normal vs 4% Falla).',
        'recall': 'De todas las fallas reales, ¿cuántas logró detectar el modelo? Es la métrica más crítica en mantenimiento predictivo para evitar paros no planificados.',
        'f1': 'Media armónica entre Precision y Recall. Busca un equilibrio entre detectar fallas reales y no generar falsas alarmas innecesarias.',
        'auc': 'Capacidad general del modelo para distinguir entre una operación normal y una falla inminente. 1.0 es perfecto, 0.5 es aleatorio.'
    }

    # ── Tarjetas de Métricas por Modelo ──────────────────────────────────────
    models_html = ""
    for idx, (name, metrics) in enumerate(data.get('modelos', {}).items()):
        is_best = (name == best_model)
        border_class = "card-best" if is_best else "card-normal"
        best_badge = (
            "<span class='text-xs best-badge px-2 py-0.5 rounded-full ml-2'>"
            "MEJOR</span>" if is_best else ""
        )
        models_html += f"""
        <div class="card fade-in {border_class}" style="--delay: {idx * 0.1}s">
            <h3 class="text-lg font-bold mb-3" style="color: var(--text-primary)">{name}{best_badge}</h3>
            <div class="grid grid-cols-2 gap-3">
                <div class="metric-box p-2 rounded text-center">
                    <p class="metric-label text-xs tooltip" data-tip="{tooltips['accuracy']}">Accuracy</p>
                    <p class="text-xl font-bold metric-white">{metrics.get('accuracy', 0):.2%}</p>
                </div>
                <div class="metric-box p-2 rounded text-center">
                    <p class="metric-label text-xs tooltip" data-tip="{tooltips['recall']}">Recall</p>
                    <p class="text-xl font-bold metric-green">{metrics.get('recall', 0):.2%}</p>
                </div>
                <div class="metric-box p-2 rounded text-center">
                    <p class="metric-label text-xs tooltip" data-tip="{tooltips['f1']}">F1-Score</p>
                    <p class="text-xl font-bold metric-amber">{metrics.get('f1', 0):.2%}</p>
                </div>
                <div class="metric-box p-2 rounded text-center">
                    <p class="metric-label text-xs tooltip" data-tip="{tooltips['auc']}">AUC-ROC</p>
                    <p class="text-xl font-bold metric-blue">{metrics.get('auc_roc', 0):.3f}</p>
                </div>
            </div>
        </div>
        """

    # ── Business Insights ────────────────────────────────────────────────────
    tasa_falla = ds.get('tasa_falla_pct', 0)
    total_registros = ds.get('total_registros', 0)

    insights_html = f"""
    <section id="insights" class="card fade-in border-l-4 border-blue-500" style="--delay: 0.1s">
        <h2 class="text-2xl font-bold mb-4" style="color: var(--text-primary)">💡 Hallazgos Clave para el Negocio</h2>
        <div class="grid md:grid-cols-3 gap-4">
            <div class="insight-card p-4 rounded">
                <p class="insight-title-green font-bold mb-2">🎯 Detección Temprana</p>
                <p class="insight-text text-sm">
                    El modelo <strong>{best_model}</strong> identifica correctamente el
                    <strong class="insight-value-green">{best_recall:.1%}</strong> de las fallas reales antes de
                    que ocurran, permitiendo programar mantenimiento preventivo y reducir el tiempo de
                    inactividad no planificado.
                </p>
            </div>
            <div class="insight-card p-4 rounded">
                <p class="insight-title-amber font-bold mb-2">⚠️ Predictores Críticos</p>
                <p class="insight-text text-sm">
                    El <strong>Torque elevado (&gt; 45.95 Nm)</strong> y la
                    <strong>alta temperatura de proceso (&gt; 309.5 K)</strong>
                    son las señales físicas más fuertes de falla inminente. Monitorear estas variables
                    en tiempo real puede prevenir hasta el 88% de los incidentes.
                </p>
            </div>
            <div class="insight-card p-4 rounded">
                <p class="insight-title-red font-bold mb-2">📊 Contexto del Problema</p>
                <p class="insight-text text-sm">
                    Solo el <strong>{tasa_falla}%</strong> de las
                    {total_registros:,} operaciones registran falla. Sin técnicas de balanceo
                    como <strong>SMOTE</strong>, un modelo convencional habría
                    ignorado estas fallas críticas, alcanzando un Recall cercano al 0%.
                </p>
            </div>
        </div>
    </section>
    """

    # ── Selector Dinámico + Datos JS para Matriz de Confusión ────────────────
    cm_data = data.get('confusion_matrices', {})
    cm_js_data = {}
    for model_name, cm_info in cm_data.items():
        cm_js_data[model_name] = {
            'matrix': cm_info.get('matrix', [[0, 0], [0, 0]]),
            'tp': cm_info.get('tp', 0),
            'fp': cm_info.get('fp', 0),
            'tn': cm_info.get('tn', 0),
            'fn': cm_info.get('fn', 0)
        }

    selector_options = ""
    for model_name in cm_data.keys():
        selected = "selected" if model_name == best_model else ""
        selector_options += f'<option value="{model_name}" {selected}>{model_name}</option>\n'

    # ── Plantilla HTML Principal ─────────────────────────────────────────────
    return f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dashboard | Mantenimiento Predictivo AI4I 2020</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        /* ══════════════════════════════════════════════════════════════════
           VARIABLES CSS PARA TEMA CLARO/OSCURO
           ══════════════════════════════════════════════════════════════════ */
        :root {{
            /* Tema Oscuro (default) */
            --bg-primary: #0f172a;
            --bg-secondary: #1e293b;
            --bg-tertiary: #334155;
            --bg-card: #1e293b;
            --bg-card-hover: #273549;
            --text-primary: #e2e8f0;
            --text-secondary: #94a3b8;
            --text-muted: #64748b;
            --border-color: #334155;
            --border-hover: #475569;
            --nav-bg: rgba(15, 23, 42, 0.85);
            --shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
            --tooltip-bg: #020617;
            --tooltip-border: #334155;
        }}

        /* Tema Claro */
        :root.light-theme {{
            --bg-primary: #f8fafc;
            --bg-secondary: #ffffff;
            --bg-tertiary: #e2e8f0;
            --bg-card: #ffffff;
            --bg-card-hover: #f1f5f9;
            --text-primary: #0f172a;
            --text-secondary: #475569;
            --text-muted: #64748b;
            --border-color: #e2e8f0;
            --border-hover: #cbd5e1;
            --nav-bg: rgba(255, 255, 255, 0.9);
            --shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.08);
            --tooltip-bg: #1e293b;
            --tooltip-border: #475569;
        }}

        html {{
            scroll-behavior: smooth;
            scroll-padding-top: 4rem;
        }}
        body {{
            background-color: var(--bg-primary);
            color: var(--text-primary);
            font-family: 'Inter', system-ui, sans-serif;
            transition: background-color 0.3s ease, color 0.3s ease;
        }}
        .card {{
            background-color: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 0.5rem;
            padding: 1.5rem;
            box-shadow: var(--shadow);
            transition: background-color 0.3s ease, border-color 0.3s ease;
        }}
        .grid-charts {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(450px, 1fr));
            gap: 1.5rem;
        }}

        /* ══════════════════════════════════════════════════════════════════
           COLORES POR TEMA PARA INSIGHTS Y MÉTRICAS
           ══════════════════════════════════════════════════════════════════ */
        
        /* --- Insight Cards (Business Insights) --- */
        .insight-card {{
            background-color: #1e293b;  /* slate-800 (dark) */
            transition: background-color 0.3s ease;
        }}
        .light-theme .insight-card {{
            background-color: #f1f5f9;  /* slate-100 (light) */
        }}
        
        .insight-title-green {{ color: #4ade80; }}  /* green-400 */
        .light-theme .insight-title-green {{ color: #16a34a; }}  /* green-600 */
        
        .insight-title-amber {{ color: #fbbf24; }}  /* amber-400 */
        .light-theme .insight-title-amber {{ color: #d97706; }}  /* amber-600 */
        
        .insight-title-red {{ color: #f87171; }}  /* red-400 */
        .light-theme .insight-title-red {{ color: #dc2626; }}  /* red-600 */
        
        .insight-value-green {{ color: #4ade80; }}  /* green-400 */
        .light-theme .insight-value-green {{ color: #16a34a; }}  /* green-600 */
        
        .insight-text {{
            color: #cbd5e1;  /* slate-300 */
        }}
        .light-theme .insight-text {{
            color: #475569;  /* slate-600 */
        }}
        
        .insight-text strong {{
            color: #f8fafc;  /* slate-50 */
        }}
        .light-theme .insight-text strong {{
            color: #0f172a;  /* slate-900 */
        }}
        
        /* --- Metric Cards (Desempeño por Modelo) --- */
        .metric-box {{
            background-color: #0f172a;  /* slate-900 (dark) */
            transition: background-color 0.3s ease;
        }}
        .light-theme .metric-box {{
            background-color: #e2e8f0;  /* slate-200 (light) */
        }}
        
        .metric-label {{
            color: #94a3b8;  /* slate-400 */
        }}
        .light-theme .metric-label {{
            color: #64748b;  /* slate-500 */
        }}
        
        .metric-white {{ color: #f8fafc; }}  /* slate-50 */
        .light-theme .metric-white {{ color: #0f172a; }}  /* slate-900 */
        
        .metric-green {{ color: #4ade80; }}  /* green-400 */
        .light-theme .metric-green {{ color: #16a34a; }}  /* green-600 */
        
        .metric-amber {{ color: #fbbf24; }}  /* amber-400 */
        .light-theme .metric-amber {{ color: #d97706; }}  /* amber-600 */
        
        .metric-blue {{ color: #60a5fa; }}  /* blue-400 */
        .light-theme .metric-blue {{ color: #2563eb; }}  /* blue-600 */
        
        .metric-red {{ color: #f87171; }}  /* red-400 */
        .light-theme .metric-red {{ color: #dc2626; }}  /* red-600 */
        
        /* Best model badge */
        .best-badge {{
            background-color: #3b82f6;  /* blue-500 */
            color: #ffffff;
        }}
        .light-theme .best-badge {{
            background-color: #2563eb;  /* blue-600 */
            color: #ffffff;
        }}
        
        /* Best model border */
        .card-best {{
            border: 2px solid #3b82f6;  /* blue-500 */
        }}
        .light-theme .card-best {{
            border: 2px solid #2563eb;  /* blue-600 */
        }}
        .card-normal {{
            border: 1px solid #334155;  /* slate-700 */
        }}
        .light-theme .card-normal {{
            border: 1px solid #cbd5e1;  /* slate-300 */
        }}
        
        /* KPI cards specific colors */
        .kpi-red {{ color: #f87171; }}
        .light-theme .kpi-red {{ color: #dc2626; }}
        
        .kpi-blue {{ color: #60a5fa; }}
        .light-theme .kpi-blue {{ color: #2563eb; }}
        
        .kpi-green {{ color: #4ade80; }}
        .light-theme .kpi-green {{ color: #16a34a; }}

        /* Interval boxes */
        .interval-box {{
            background-color: #1e293b;
            transition: background-color 0.3s ease;
        }}
        .light-theme .interval-box {{
            background-color: #f1f5f9;
        }}
        .interval-label {{
            color: #94a3b8;
        }}
        .light-theme .interval-label {{
            color: #64748b;
        }}
        .interval-value {{
            color: #fbbf24;
        }}
        .light-theme .interval-value {{
            color: #d97706;
        }}

        /* ══════════════════════════════════════════════════════════════════
           TOGGLE SWITCH MODO CLARO/OSCURO
           ══════════════════════════════════════════════════════════════════ */
        .theme-toggle {{
            position: relative;
            width: 60px;
            height: 30px;
            background-color: var(--bg-tertiary);
            border-radius: 15px;
            cursor: pointer;
            border: 1px solid var(--border-color);
            transition: background-color 0.3s ease;
            display: flex;
            align-items: center;
            padding: 0 4px;
        }}
        .theme-toggle::before {{
            content: '🌙';
            position: absolute;
            left: 6px;
            font-size: 14px;
            z-index: 1;
        }}
        .theme-toggle::after {{
            content: '☀️';
            position: absolute;
            right: 6px;
            font-size: 14px;
            z-index: 1;
        }}
        .theme-toggle-slider {{
            position: absolute;
            width: 24px;
            height: 24px;
            background-color: var(--text-primary);
            border-radius: 50%;
            transition: transform 0.3s ease;
            transform: translateX(0);
            z-index: 2;
        }}
        .light-theme .theme-toggle-slider {{
            transform: translateX(30px);
        }}

        /* ══════════════════════════════════════════════════════════════════
           TOOLTIPS EDUCATIVOS
           ══════════════════════════════════════════════════════════════════ */
        .tooltip {{
            position: relative;
            cursor: help;
            border-bottom: 1px dashed var(--text-muted);
            display: inline-block;
        }}
        .tooltip::after {{
            content: attr(data-tip);
            position: absolute;
            bottom: 125%;
            left: 50%;
            transform: translateX(-50%);
            background-color: var(--tooltip-bg);
            color: #e2e8f0;
            padding: 0.5rem 0.75rem;
            border-radius: 0.375rem;
            font-size: 0.75rem;
            font-weight: normal;
            width: 240px;
            text-align: center;
            opacity: 0;
            visibility: hidden;
            transition: opacity 0.2s, visibility 0.2s;
            z-index: 100;
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.5);
            border: 1px solid var(--tooltip-border);
            line-height: 1.4;
            pointer-events: none;
        }}
        .tooltip:hover::after {{
            opacity: 1;
            visibility: visible;
        }}

        /* ══════════════════════════════════════════════════════════════════
           ANIMACIONES FADE-IN AL HACER SCROLL
           ══════════════════════════════════════════════════════════════════ */
        .fade-in {{
            opacity: 0;
            transform: translateY(30px);
            transition: opacity 0.6s ease-out, transform 0.6s ease-out;
            transition-delay: var(--delay, 0s);
        }}
        .fade-in.visible {{
            opacity: 1;
            transform: translateY(0);
        }}

        .header-animate {{
            opacity: 0;
            transform: translateY(-20px);
            animation: slideDown 0.8s ease-out 0.2s forwards;
        }}
        @keyframes slideDown {{
            to {{
                opacity: 1;
                transform: translateY(0);
            }}
        }}

        .pulse-badge {{
            animation: pulse 2s ease-in-out infinite;
        }}
        @keyframes pulse {{
            0%, 100% {{ box-shadow: 0 0 0 0 rgba(59, 130, 246, 0.4); }}
            50% {{ box-shadow: 0 0 0 8px rgba(59, 130, 246, 0); }}
        }}

        /* Nav sticky */
        .nav-sticky {{
            position: sticky;
            top: 0;
            z-index: 50;
            background-color: var(--nav-bg);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border-bottom: 1px solid var(--border-color);
            transition: background-color 0.3s ease, border-color 0.3s ease;
        }}
        .nav-link {{
            color: var(--text-secondary);
            transition: color 0.2s ease;
        }}
        .nav-link:hover {{
            color: #3b82f6;
        }}

        /* Selector dinámico */
        .model-selector {{
            background-color: var(--bg-tertiary);
            color: var(--text-primary);
            border: 1px solid var(--border-color);
            border-radius: 0.375rem;
            padding: 0.5rem 1rem;
            font-size: 0.875rem;
            cursor: pointer;
            outline: none;
            transition: border-color 0.2s, background-color 0.3s ease;
        }}
        .model-selector:hover {{ border-color: #3b82f6; }}
        .model-selector:focus {{
            border-color: #3b82f6;
            box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.3);
        }}
        .cm-stat {{
            background-color: var(--bg-primary);
            border-radius: 0.375rem;
            padding: 0.5rem;
            text-align: center;
            transition: background-color 0.3s ease;
        }}

        /* Footer */
        footer {{
            border-top: 1px solid var(--border-color);
            transition: border-color 0.3s ease;
        }}
        .footer-btn {{
            background-color: var(--bg-tertiary);
            color: var(--text-secondary);
            transition: background-color 0.3s ease, color 0.3s ease;
        }}
        .footer-btn:hover {{
            background-color: var(--bg-card-hover);
            color: var(--text-primary);
        }}
    </style>
</head>
<body class="min-h-screen">
    <!-- HEADER STICKY CON NAVEGACIÓN + TOGGLE TEMA -->
    <nav class="nav-sticky">
        <div class="max-w-7xl mx-auto px-4 py-3 flex flex-wrap items-center justify-between gap-4 text-sm font-medium">
            <div class="flex flex-wrap items-center gap-4 md:gap-8">
                <a href="#kpis" class="nav-link">📊 KPIs</a>
                <a href="#insights" class="nav-link">💡 Insights</a>
                <a href="#modelos" class="nav-link">🤖 Modelos</a>
                <a href="#intervalos" class="nav-link">⚠️ Riesgo</a>
                <a href="#matriz" class="nav-link">🔲 Matriz</a>
                <a href="#graficos" class="nav-link">📈 Gráficos</a>
            </div>
            <!-- Toggle Modo Claro/Oscuro -->
            <div class="flex items-center gap-2">
                <span class="text-xs" style="color: var(--text-muted)">Tema</span>
                <div class="theme-toggle" id="theme-toggle" onclick="toggleTheme()" title="Cambiar tema">
                    <div class="theme-toggle-slider"></div>
                </div>
            </div>
        </div>
    </nav>

    <header class="max-w-7xl mx-auto mb-8 pt-8 px-8 text-center header-animate">
        <h1 class="text-4xl font-bold mb-2" style="color: var(--text-primary)">
            Mantenimiento Predictivo Industrial
        </h1>
        <p style="color: var(--text-secondary)">
            AI4I 2020 Predictive Maintenance Dataset (Matzka, 2020) |
            Metodología CRISP-DM
        </p>
        <span class="inline-block mt-4 px-4 py-1 rounded-full text-white font-semibold {badge_color} pulse-badge">
            {badge_text}
        </span>
        <div class="mt-4">
        
            <a href="../docs/aldo fuentes zaldivar_informe técnico_completo final.pdf" download
               class="inline-block bg-blue-600 hover:bg-blue-700 transition-colors px-6 py-2 rounded-lg text-white font-semibold shadow-lg">
                📥 Descargar Informe Técnico (PDF)
            </a>
            
        </div>
    </header>

    <main class="max-w-7xl mx-auto space-y-8 px-8 pb-8">
        <!-- KPIs Globales -->
        <section id="kpis" class="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div class="card fade-in text-center" style="--delay: 0s">
                <p class="text-sm" style="color: var(--text-secondary)">Total Registros</p>
                <p class="text-2xl font-bold" style="color: var(--text-primary)">
                    {ds.get('total_registros', 0):,}
                </p>
            </div>
            <div class="card fade-in text-center" style="--delay: 0.1s">
                <p class="text-sm" style="color: var(--text-secondary)">Tasa de Falla Real</p>
                <p class="text-2xl font-bold kpi-red">
                    {ds.get('tasa_falla_pct', 0)}%
                </p>
            </div>
            <div class="card fade-in text-center" style="--delay: 0.2s">
                <p class="text-sm" style="color: var(--text-secondary)">Mejor Modelo</p>
                <p class="text-2xl font-bold kpi-blue">{best_model}</p>
            </div>
            <div class="card fade-in text-center" style="--delay: 0.3s">
                <p class="text-sm" style="color: var(--text-secondary)">Recall Obtenido</p>
                <p class="text-2xl font-bold kpi-green">
                    {best_recall:.2%}
                    <span class="text-sm" style="color: var(--text-muted)">
                        (Meta ≥ {hip.get('recall_minimo', 0.8):.0%})
                    </span>
                </p>
            </div>
        </section>

        <!-- Business Insights -->
        {insights_html}

        <!-- Desempeño por Modelo -->
        <section id="modelos" class="fade-in" style="--delay: 0.2s">
            <h2 class="text-2xl font-bold mb-4" style="color: var(--text-primary)">
                📊 Desempeño por Modelo
            </h2>
            <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
                {models_html}
            </div>
        </section>

        <!-- Intervalos de Riesgo -->
        <section id="intervalos" class="card fade-in" style="--delay: 0.1s">
            <h2 class="text-xl font-bold mb-4" style="color: #fbbf24">
                ⚠️ Intervalos de Riesgo Físico (Solución)
            </h2>
            <div class="grid grid-cols-2 md:grid-cols-5 gap-3">
                {intervalos_html}
            </div>
        </section>

        <!-- Matriz de Confusión con Selector Dinámico -->
        <section id="matriz" class="card fade-in" style="--delay: 0.2s">
            <div class="flex flex-col sm:flex-row items-start sm:items-center justify-between mb-4 gap-3">
                <h2 class="text-xl font-bold" style="color: var(--text-primary)">
                    🔲 Matriz de Confusión Interactiva
                </h2>
                <div class="flex items-center gap-3">
                    <label for="model-select" class="text-sm" style="color: var(--text-secondary)">Modelo:</label>
                    <select id="model-select" class="model-selector" onchange="updateConfusionMatrix(this.value)">
                        {selector_options}
                    </select>
                </div>
            </div>
            <div id="cm-stats" class="grid grid-cols-4 gap-3 mb-4">
                <div class="cm-stat">
                    <p class="text-xs" style="color: var(--text-secondary)">Verdaderos Positivos</p>
                    <p id="stat-tp" class="text-lg font-bold metric-green">--</p>
                </div>
                <div class="cm-stat">
                    <p class="text-xs" style="color: var(--text-secondary)">Falsos Positivos</p>
                    <p id="stat-fp" class="text-lg font-bold metric-red">--</p>
                </div>
                <div class="cm-stat">
                    <p class="text-xs" style="color: var(--text-secondary)">Verdaderos Negativos</p>
                    <p id="stat-tn" class="text-lg font-bold metric-blue">--</p>
                </div>
                <div class="cm-stat">
                    <p class="text-xs" style="color: var(--text-secondary)">Falsos Negativos</p>
                    <p id="stat-fn" class="text-lg font-bold metric-amber">--</p>
                </div>
            </div>
            <div id="confusion-matrix-container"></div>
        </section>

        <!-- Gráficos Interactivos -->
        <section id="graficos" class="grid-charts">
            <div class="card fade-in" style="--delay: 0s">
                📊 Distribución Modos Falla
                {plots['modes']}
            </div>
            <div class="card fade-in" style="--delay: 0.1s">
                📈 Comparación de Métricas
                {plots['comp']}
            </div>
            <div class="card fade-in" style="--delay: 0.2s">
                🕸️ Perfil Multidimensional (Radar)
                {plots['radar']}
            </div>
            <div class="card fade-in" style="--delay: 0.3s">
                🎯 Curvas ROC
                {plots['roc']}
            </div>
            <div class="card fade-in" style="--delay: 0.4s">
                🧠 Importancia de Variables
                {plots['imp']}
            </div>
            <div class="card fade-in" style="--delay: 0.5s">
                📊 Validación Cruzada (5-Fold)
                {plots['cv']}
            </div>
        </section>
    </main>

    <!-- Footer con Enlaces -->
    <footer class="max-w-7xl mx-auto mt-12 text-center pt-6 fade-in px-8" style="--delay: 0s">
        <p class="text-sm mb-4" style="color: var(--text-muted)">
            Desarrollado por Aldo Fuentes Zaldivar | Ingeniería Mecánica UAEMex
        </p>
        <p class="text-xs mb-6" style="color: var(--text-muted)">
            Pipeline de Machine Learning con Python, Scikit-Learn y Plotly | 2025-2026
        </p>
        <div class="flex flex-wrap justify-center gap-4 mb-6">
        
            <a href="https://github.com/Al-Fuentes-27/ML_PredictiveMaintenance_afz" target="_blank"
               class="flex items-center gap-2 footer-btn px-4 py-2 rounded-lg text-sm">
                <svg class="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/>
                </svg>
                Código Fuente
            </a>
            
            <a href="https://doi.org/10.24432/C5HS5C" target="_blank"
               class="flex items-center gap-2 footer-btn px-4 py-2 rounded-lg text-sm">
                <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4m0 5c0 2.21-3.582 4-8 4s-8-1.79-8-4"/>
                </svg>
                Dataset UCI ML
            </a>
            
            <!--<a href="https://www.linkedin.com/in/aldo-fuentes-zaldivar/" target="_blank"
               class="flex items-center gap-2 footer-btn px-4 py-2 rounded-lg text-sm">
                <svg class="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z"/>
                </svg>
                LinkedIn
            </a>-->
            
            <a href="../results/reports/metrics_summary.txt" download
               class="flex items-center gap-2 footer-btn px-4 py-2 rounded-lg text-sm">
                <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"/>
                </svg>
                Métricas Raw (TXT)
            </a>
            
        </div>
        <p class="text-xs pb-4" style="color: var(--text-muted)">
            Metodología CRISP-DM | Método Científico de Bunge | Taxonomía de Bloom
        </p>
    </footer>

    <!-- JavaScript -->
    <script>
        // ═══════════════════════════════════════════════════════════════════
        // DATOS DE MATRICES DE CONFUSIÓN
        // ═══════════════════════════════════════════════════════════════════
        const confusionMatrices = {json.dumps(cm_js_data, ensure_ascii=False)};

        function getPlotlyColors() {{
            const isLight = document.documentElement.classList.contains('light-theme');
            return {{
                font: isLight ? '#0f172a' : '#e2e8f0',
                grid: isLight ? '#e2e8f0' : '#334155',
                paper: 'rgba(0,0,0,0)',
                plot: 'rgba(0,0,0,0)'
            }};
        }}

        function updateConfusionMatrix(modelName) {{
            const cm = confusionMatrices[modelName];
            if (!cm) return;

            document.getElementById('stat-tp').textContent = cm.tp;
            document.getElementById('stat-fp').textContent = cm.fp;
            document.getElementById('stat-tn').textContent = cm.tn;
            document.getElementById('stat-fn').textContent = cm.fn;

            const colors = getPlotlyColors();

            const data = [{{
                type: 'heatmap',
                z: cm.matrix,
                x: ['Normal', 'Falla'],
                y: ['Normal', 'Falla'],
                colorscale: [
                    [0, '#064e3b'],
                    [0.5, '#065f46'],
                    [1, '#10b981']
                ],
                text: cm.matrix,
                texttemplate: '%{{text}}',
                textfont: {{ size: 28, color: 'white' }},
                showscale: false,
                hovertemplate:
                    '<b>Real:</b> %{{y}}<br>' +
                    '<b>Pred:</b> %{{x}}<br>' +
                    '<b>Casos:</b> %{{z}}<extra></extra>'
            }}];

            const layout = {{
                template: 'plotly_dark',
                paper_bgcolor: colors.paper,
                plot_bgcolor: colors.plot,
                font: {{ color: colors.font }},
                margin: {{ l: 40, r: 20, t: 30, b: 40 }},
                height: 320,
                xaxis: {{ title: 'Predicción', side: 'bottom' }},
                yaxis: {{ title: 'Valor Real', autorange: 'reversed' }},
                title: {{
                    text: modelName,
                    font: {{ size: 16, color: colors.font }}
                }}
            }};

            Plotly.react('confusion-matrix-container', data, layout, {{
                displayModeBar: false,
                responsive: true
            }});
        }}

        // ═══════════════════════════════════════════════════════════════════
        // TOGGLE MODO CLARO/OSCURO CON PERSISTENCIA
        // ═══════════════════════════════════════════════════════════════════
        function toggleTheme() {{
            const html = document.documentElement;
            html.classList.toggle('light-theme');
            
            const isLight = html.classList.contains('light-theme');
            localStorage.setItem('dashboard-theme', isLight ? 'light' : 'dark');
            
            updateAllPlotlyCharts();
            
            const selector = document.getElementById('model-select');
            if (selector) {{
                updateConfusionMatrix(selector.value);
            }}
        }}

        function updateAllPlotlyCharts() {{
            const colors = getPlotlyColors();
            const update = {{
                'paper_bgcolor': colors.paper,
                'plot_bgcolor': colors.plot,
                'font.color': colors.font
            }};
            
            const plotlyDivs = document.querySelectorAll('.js-plotly-plot');
            plotlyDivs.forEach(div => {{
                try {{
                    Plotly.relayout(div, update);
                }} catch(e) {{
                    // Ignorar errores en gráficos que no soportan relayout
                }}
            }});
        }}

        function initTheme() {{
            const saved = localStorage.getItem('dashboard-theme');
            
            if (!saved) {{
                const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
                if (!prefersDark) {{
                    document.documentElement.classList.add('light-theme');
                }}
            }} else if (saved === 'light') {{
                document.documentElement.classList.add('light-theme');
            }}
        }}

        // ═══════════════════════════════════════════════════════════════════
        // INTERSECTION OBSERVER — Animaciones Fade-In
        // ═══════════════════════════════════════════════════════════════════
        document.addEventListener('DOMContentLoaded', function() {{
            initTheme();

            const selector = document.getElementById('model-select');
            if (selector) {{
                updateConfusionMatrix(selector.value);
            }}

            const observerOptions = {{
                root: null,
                rootMargin: '0px',
                threshold: 0.1
            }};

            const observer = new IntersectionObserver((entries) => {{
                entries.forEach(entry => {{
                    if (entry.isIntersecting) {{
                        entry.target.classList.add('visible');
                        observer.unobserve(entry.target);
                    }}
                }});
            }}, observerOptions);

            document.querySelectorAll('.fade-in').forEach(el => {{
                observer.observe(el);
            }});
        }});
    </script>
</body>
</html>"""





# ── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "=" * 60)
    print(" 06_html_dashboard.py — Generador de Dashboard Interactivo")
    print("=" * 60)

    if not DASHBOARD_JSON.exists():
        print(f"\n❌ Error: No se encontró {DASHBOARD_JSON}")
        print(" Asegúrate de haber ejecutado primero: python src/05_save_results.py\n")
        sys.exit(1)

    print(f"\n[1/3] Cargando datos desde: {DASHBOARD_JSON.name} (vía utils_paths.json)")
    with open(DASHBOARD_JSON, "r", encoding="utf-8") as f:
        data = clean_keys(json.load(f))

    best_model = data.get("mejor_modelo", "Bosque Aleatorio")
    print(f" Mejor modelo detectado: {best_model}")

    print("\n[2/3] Generando gráficos interactivos con Plotly...")
    print("    ✓ Curvas ROC")
    print("    ✓ Comparación de Métricas")
    print("    ✓ Importancia de Variables")
    print("    ✓ Distribución de Modos de Falla")
    print("    ✓ Validación Cruzada (CV-F1)")
    print("    ✓ Gráfico de Radar (Spider Chart)")
    plots = generate_plots(data, best_model)

    print("\n[3/3] Construyendo y guardando archivo HTML...")
    html_content = build_html(data, plots)

    OUTPUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"\n✅ Dashboard HTML generado exitosamente!")
    print(f"📂 Ubicación: {OUTPUT_HTML}")
    print(f"👉 Abre este archivo en tu navegador para visualizar los resultados.\n")
    print(" Las 10 mejoras implementadas:")
    print("    ✓ 1. Business Insights (Hallazgos de Negocio)")
    print("    ✓ 2. Selector dinámico de Matriz de Confusión")
    print("    ✓ 3. Validación Cruzada con barras de error")
    print("    ✓ 4. Botón de Descarga del Informe (PDF)")
    print("    ✓ 5. Footer con enlaces a GitHub, UCI ML y LinkedIn")
    print("    ✓ 6. Gráfico de Radar (Perfil Multidimensional)")
    print("    ✓ 7. Animaciones Fade-In al hacer Scroll")
    print("    ✓ 8. Header Sticky con Navegación por Anclas")
    print("    ✓ 9. Tooltips Educativos en Métricas")
    print("    ✓ 10. Toggle Modo Claro/Oscuro con persistencia")




if __name__ == "__main__":
    main()
    
    
    
    