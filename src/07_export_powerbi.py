"""
07_export_powerbi.py
====================
Exportador de datos para Power BI (Star Schema / Flat CSV).
Toma las predicciones del mejor modelo (Bosque Aleatorio) y las mapea
de vuelta a los datos crudos originales para que los analistas de BI
puedan filtrar por Product ID, Type (L/M/H) y umbrales físicos.

Uso:
python src/07_export_powerbi.py

Salidas:
data/powerbi_datasets/fact_predictions.csv

Dependencias: pandas 3.0.2, numpy 2.4.4, scikit-learn 1.8.0
Autor: Aldo Fuentes Zaldivar — 2025-2026
"""
import sys
import pickle
import argparse
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

# ┌─────────────────────────────────────────────────────────────────────────┐
# │  CONFIGURACIÓN CENTRALIZADA                                             │
# │  La ruta a config.json está definida en utils/config.py → CONFIG_PATH  │
from utils.utils_config import cfg                                               #│
# └─────────────────────────────────────────────────────────────────────────┘
from utils.utils_preprocessing import cargar_datos, preparar_features

# ── CLI ───────────────────────────────────────────────────────────────────────
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exportación de predicciones a formato plano para Power BI."
    )
    parser.add_argument(
        "--data",
        default=cfg["paths"]["data"],
        help=f"Ruta al dataset CSV original (default: {cfg['paths']['data']})",
    )
    parser.add_argument(
        "--output-dir",
        default=cfg["paths"]["powerbi_dir"],
        help=f"Carpeta de salida para datasets de Power BI (default: {cfg['paths']['powerbi_dir']})",
    )
    return parser.parse_args()

# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    args = _parse_args()
    
    print("\n" + "=" * 65)
    print("  07_export_powerbi.py")
    print("  Exportación de Datos para Enterprise BI (Power BI)")
    print("=" * 65)

    # 1. Cargar datos originales y recrear el split exacto
    print("\n[1/5] Cargando dataset original y recreando índices de prueba...")
    df_raw = cargar_datos(args.data)
    y_full = df_raw["falla"].values
    
    # Recreamos la división estratificada con la misma semilla para obtener los índices exactos
    indices = np.arange(len(df_raw))
    _, test_indices, _, _ = train_test_split(
        indices, y_full,
        test_size=cfg["params"]["test_size"],
        random_state=cfg["params"]["random_state"],
        stratify=y_full
    )
    
    # Filtramos el dataframe original para quedarnos solo con las filas del conjunto de prueba
    df_test = df_raw.iloc[test_indices].copy()
    print(f"  [OK] Aisladas {len(df_test):,} filas del conjunto de prueba con datos de negocio.")

    # 2. Cargar el conjunto de prueba escalado (X_test_s) para las predicciones
    print("\n[2/5] Cargando matrices procesadas (X_test_s, y_test)...")
    proc_dir = Path(cfg["paths"]["processed_dir"])
    X_test_s = np.load(proc_dir / Path(cfg["paths"]["processed"]["X_test_s"]).name)
    y_test   = np.load(proc_dir / Path(cfg["paths"]["processed"]["y_test"]).name)
    
    # 3. Cargar el mejor modelo (Bosque Aleatorio)
    print("\n[3/5] Cargando modelo Bosque Aleatorio (random_forest_v1.pkl)...")
    model_path = Path(cfg["paths"]["models"]["random_forest"])
    if not model_path.exists():
        raise FileNotFoundError(f"Modelo no encontrado en {model_path}. Ejecuta primero 03_train_random_forest.py")
        
    with open(model_path, "rb") as f:
        rf_model = pickle.load(f)
        
    # 4. Generar predicciones y probabilidades
    print("\n[4/5] Generando predicciones y probabilidades...")
    y_pred = rf_model.predict(X_test_s)
    y_prob = rf_model.predict_proba(X_test_s)[:, 1]
    
    # 5. Construir el DataFrame final para Power BI (Flat Table)
    print("\n[5/5] Construyendo tabla de hechos (Fact Table) para Power BI...")
    
    # Extraer columnas de negocio legibles (usando los nombres ya renombrados por utils_preprocessing)
    pbi_df = pd.DataFrame({
        "Product_ID":           df_test["Product ID"].values,
        "Type":                 df_test["Type"].values,
        "Air_Temp_K":           df_test["temp_aire"].values,
        "Process_Temp_K":       df_test["temp_proceso"].values,
        "Rotational_Speed_rpm": df_test["vel_rotacion"].values,
        "Torque_Nm":            df_test["torque"].values,
        "Tool_Wear_min":        df_test["desgaste"].values,
        "True_Label":           y_test,
        "Predicted_Label":      y_pred,
        "Failure_Probability":  np.round(y_prob, 4),
    })
    
    # Añadir lógica de negocio (Métricas de error y Zonas de Riesgo)
    pbi_df["Is_True_Positive"]  = (pbi_df["True_Label"] == 1) & (pbi_df["Predicted_Label"] == 1)
    pbi_df["Is_False_Positive"] = (pbi_df["True_Label"] == 0) & (pbi_df["Predicted_Label"] == 1)
    pbi_df["Is_False_Negative"] = (pbi_df["True_Label"] == 1) & (pbi_df["Predicted_Label"] == 0)
    
    # Banderas basadas en la Hipótesis de Solución (Intervalos Críticos)
    pbi_df["Risk_Torque_High"]  = pbi_df["Torque_Nm"] > 45.95
    pbi_df["Risk_Speed_Low"]    = pbi_df["Rotational_Speed_rpm"] < 1421.50
    pbi_df["Risk_Wear_High"]    = pbi_df["Tool_Wear_min"] > 84.50
    pbi_df["Risk_TempAir_High"] = pbi_df["Air_Temp_K"] > 299.10
    pbi_df["Risk_TempProc_High"]= pbi_df["Process_Temp_K"] > 309.50

    # Guardar CSV
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    out_file = out_dir / Path(cfg["paths"]["powerbi"]["fact_predictions"]).name
    pbi_df.to_csv(out_file, index=False, encoding="utf-8")
    
    print("\n" + "=" * 65)
    print("  EXPORTACIÓN COMPLETADA")
    print("=" * 65)
    print(f"  Archivo generado : {out_file}")
    print(f"  Total de filas   : {len(pbi_df):,}")
    print(f"  Fallas Reales    : {pbi_df['True_Label'].sum()}")
    print(f"  Falsos Negativos : {pbi_df['Is_False_Negative'].sum()} (Riesgo no detectado)")
    print(f"  Falsos Positivos : {pbi_df['Is_False_Positive'].sum()} (Falsas alarmas)")
    print(f"\n🚀 Siguiente paso: Abrir Power BI Desktop e importar este CSV.\n")

if __name__ == "__main__":
    main()





