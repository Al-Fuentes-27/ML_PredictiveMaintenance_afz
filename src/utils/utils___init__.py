"""
utils/
======
Funciones reutilizables del proyecto de mantenimiento predictivo.

Módulos disponibles:
    - config        : carga centralizada de config.json (CONFIG_PATH definida aquí)
    - metrics       : cálculo y serialización de métricas de clasificación
    - preprocessing : codificación, normalización y balanceo de datos
    - visualization : generación de figuras estadísticas y de evaluación

Uso:
    from utils.config        import cfg
    from utils.metrics       import calcular_metricas, guardar_metricas_json
    from utils.preprocessing import cargar_datos, construir_pipeline
    from utils.visualization import graficar_confusion, graficar_roc

Autor  : Aldo Fuentes Zaldívar
Dataset: AI4I 2020 Predictive Maintenance (Matzka, 2020)
         DOI: https://doi.org/10.24432/C5HS5C
"""

# ┌─────────────────────────────────────────────────────────────────────────┐
# │  CONFIGURACIÓN CENTRALIZADA                                             │
# │  La ruta a config.json está definida en utils/config.py → CONFIG_PATH  │
from utils.utils_config             import cfg, cargar_config                                #│
# └─────────────────────────────────────────────────────────────────────────┘

from utils.utils_metrics            import calcular_metricas, guardar_metricas_json, imprimir_reporte
from utils.utils_preprocessing      import (
    cargar_datos, 
    preparar_features, 
    dividir_datos, 
    balancear_smote,
    normalizar,
    guardar_datos_procesados,
    cargar_datos_procesados,
)
from utils.utils_visualization      import (
    graficar_distribucion_clases,
    graficar_histogramas,
    graficar_boxplots,
    graficar_correlacion,
    graficar_dispersion,
    graficar_tipo_producto,
    graficar_confusion,
    graficar_metricas_comparacion,
    graficar_roc,
    graficar_importancia_variables,
    graficar_reporte_clasificacion,
)

__all__ = [
    # config
    "cfg",
    "cargar_config",
    # metrics
    "calcular_metricas",
    "guardar_metricas_json",
    "imprimir_reporte",
    # preprocessing
    "cargar_datos",
    "preparar_features",
    "dividir_datos",
    "balancear_smote",
    "normalizar",
    "guardar_datos_procesados",
    "cargar_datos_procesados",
    # visualization
    "graficar_distribucion_clases",
    "graficar_histogramas",
    "graficar_boxplots",
    "graficar_correlacion",
    "graficar_dispersion",
    "graficar_tipo_producto",
    "graficar_confusion",
    "graficar_metricas_comparacion",
    "graficar_roc",
    "graficar_importancia_variables",
    "graficar_reporte_clasificacion",
]
