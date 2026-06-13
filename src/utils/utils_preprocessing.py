"""
utils/preprocessing.py
======================
Funciones reutilizables para la carga, codificación, normalización
y balanceo del AI4I 2020 Predictive Maintenance Dataset.

Funciones:
    cargar_datos(ruta)
        → DataFrame limpio con columnas renombradas

    preparar_features(df)
        → X (np.ndarray), y (np.ndarray), feature_names (list)

    dividir_datos(X, y, test_size, random_state)
        → X_train, X_test, y_train, y_test (estratificado)

    balancear_smote(X_train, y_train, random_state)
        → X_train_bal, y_train_bal (balanceado 50/50)

    construir_pipeline(X_train, X_test, X_train_bal)
        → X_train_s, X_test_s, X_train_bal_s, scaler (normalizado)

Dependencias: pandas 3.0.2, numpy 2.4.4, scikit-learn 1.8.0,
              imbalanced-learn 0.14.1

Referencia del dataset:
    Matzka, S. (2020). AI4I 2020 Predictive Maintenance Dataset.
    UCI Machine Learning Repository. https://doi.org/10.24432/C5HS5C
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler, LabelEncoder
from imblearn.over_sampling  import SMOTE

# ┌─────────────────────────────────────────────────────────────────────────┐
# │  CONFIGURACIÓN CENTRALIZADA                                             │
# │  La ruta a config.json está definida en utils/config.py → CONFIG_PATH  │
from utils.utils_config import cfg                                               #│
# └─────────────────────────────────────────────────────────────────────────┘


# ── Constantes ────────────────────────────────────────────────────────────────
FAILURE_MODES = ["TWF", "HDF", "PWF", "OSF", "RNF"]

COLUMN_RENAME = {
    "Air temperature [K]":     "temp_aire",
    "Process temperature [K]": "temp_proceso",
    "Rotational speed [rpm]":  "vel_rotacion",
    "Torque [Nm]":             "torque",
    "Tool wear [min]":         "desgaste",
    "Machine failure":         "falla",
}

FEATURE_NAMES = [
    "temp_aire",
    "temp_proceso",
    "vel_rotacion",
    "torque",
    "desgaste",
    "tipo_cod",
]

FEATURE_LABELS = {
    "temp_aire":    "Temp. Aire (K)",
    "temp_proceso": "Temp. Proceso (K)",
    "vel_rotacion": "Vel. Rotación (rpm)",
    "torque":       "Torque (Nm)",
    "desgaste":     "Desgaste Herramienta (min)",
    "tipo_cod":     "Tipo Producto",
}

TARGET = "falla"

# Parámetros leídos desde config.json
TEST_SIZE    = cfg["params"]["test_size"]
RANDOM_STATE = cfg["params"]["random_state"]

# Intervalos críticos documentados en la hipótesis de solución
INTERVALOS_RIESGO = {
    "temp_aire":    {"umbral": 299.10,  "direccion": ">", "unidad": "K"},
    "temp_proceso": {"umbral": 309.50,  "direccion": ">", "unidad": "K"},
    "vel_rotacion": {"umbral": 1421.50, "direccion": "<", "unidad": "rpm"},
    "torque":       {"umbral": 45.95,   "direccion": ">", "unidad": "Nm"},
    "desgaste":     {"umbral": 84.50,   "direccion": ">", "unidad": "min"},
}


# ── Cargar y limpiar datos ────────────────────────────────────────────────────
def cargar_datos(ruta: str = None) -> pd.DataFrame:
    """
    Carga el dataset AI4I 2020 y aplica limpieza básica de nombres de columnas.

    Parámetros
    ----------
    ruta : str
        Ruta al archivo CSV. Si es None, usa config.json → paths.data.

    Retorna
    -------
    pd.DataFrame con columnas renombradas y columna 'tipo_cod' codificada.
    """
    # Ruta por defecto tomada de config.json
    if ruta is None:
        ruta = cfg["paths"]["data"]

    ruta = Path(ruta)
    if not ruta.exists():
        raise FileNotFoundError(
            f"Dataset no encontrado en: {ruta}\n"
            "Descárgalo desde: https://doi.org/10.24432/C5HS5C"
        )

    df = pd.read_csv(ruta)
    df.columns = [c.strip() for c in df.columns]
    df.rename(columns=COLUMN_RENAME, inplace=True)

    le = LabelEncoder()
    df["tipo_cod"] = le.fit_transform(df["Type"])

    df["diff_temp"]  = df["temp_proceso"] - df["temp_aire"]
    df["potencia_w"] = df["torque"] * df["vel_rotacion"] * (2 * np.pi / 60)

    _validar_columnas(df)
    _imprimir_resumen(df)

    return df


def _validar_columnas(df: pd.DataFrame) -> None:
    requeridas = set(FEATURE_NAMES + [TARGET])
    faltantes  = requeridas - set(df.columns)
    if faltantes:
        raise ValueError(f"Columnas faltantes en el dataset: {faltantes}")


def _imprimir_resumen(df: pd.DataFrame) -> None:
    n_fallas = df[TARGET].sum()
    print(f"\n  [Dataset] {len(df):,} registros | "
          f"{n_fallas:,} fallas ({n_fallas/len(df)*100:.2f}%) | "
          f"{df.isnull().sum().sum()} valores nulos")


# ── Preparar features ─────────────────────────────────────────────────────────
def preparar_features(
    df:       pd.DataFrame,
    features: list = None,
    target:   str  = TARGET,
) -> tuple:
    """
    Extrae X e y del DataFrame.
    """
    if features is None:
        features = FEATURE_NAMES

    X = df[features].values
    y = df[target].values

    print(f"  [Features] {len(features)} variables: {features}")
    print(f"  [Target]   '{target}' — Clases: {np.unique(y, return_counts=True)}")

    return X, y, features


# ── División entrenamiento / prueba ───────────────────────────────────────────
def dividir_datos(
    X:            np.ndarray,
    y:            np.ndarray,
    test_size:    float = None,
    random_state: int   = None,
) -> tuple:
    """
    Divide X e y en conjuntos de entrenamiento y prueba de forma estratificada.
    Los valores por defecto provienen de config.json → params.
    """
    if test_size    is None: test_size    = TEST_SIZE
    if random_state is None: random_state = RANDOM_STATE

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    print(f"\n  [División] Entrenamiento: {len(X_train):,} | Prueba: {len(X_test):,}")
    print(f"  [Entrenamiento] Normal: {sum(y_train==0):,} | "
          f"Falla: {sum(y_train==1):,} ({sum(y_train==1)/len(y_train)*100:.2f}%)")
    print(f"  [Prueba]        Normal: {sum(y_test==0):,}  | "
          f"Falla: {sum(y_test==1):,} ({sum(y_test==1)/len(y_test)*100:.2f}%)")

    return X_train, X_test, y_train, y_test


# ── Normalización ─────────────────────────────────────────────────────────────
def normalizar(
    X_train: np.ndarray,
    X_test:  np.ndarray,
    X_extra: np.ndarray = None,
) -> tuple:
    """
    Aplica StandardScaler ajustado SOLO sobre X_train.

    ⚠️ El scaler.fit() se realiza únicamente sobre los datos de entrenamiento
    para evitar fuga de información (data leakage).
    """
    scaler    = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)
    X_extra_s = scaler.transform(X_extra) if X_extra is not None else None

    print(f"\n  [Normalización] StandardScaler aplicado.")
    print(f"  Media (train): {scaler.mean_[:3].round(3)} ...")
    print(f"  Std  (train):  {scaler.scale_[:3].round(3)} ...")

    return X_train_s, X_test_s, scaler, X_extra_s


# ── Balanceo SMOTE ────────────────────────────────────────────────────────────
def balancear_smote(
    X_train:      np.ndarray,
    y_train:      np.ndarray,
    random_state: int = None,
) -> tuple:
    """
    Aplica SMOTE para balancear el conjunto de entrenamiento al 50/50.

    ⚠️ SMOTE se aplica ÚNICAMENTE sobre el conjunto de entrenamiento.
    El valor por defecto de random_state proviene de config.json → params.
    """
    if random_state is None:
        random_state = RANDOM_STATE

    smote = SMOTE(random_state=random_state)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

    print(f"\n  [SMOTE] Antes  → Normal: {sum(y_train==0):,} | "
          f"Falla: {sum(y_train==1):,} ({sum(y_train==1)/len(y_train)*100:.2f}%)")
    print(f"  [SMOTE] Después → Normal: {sum(y_train_bal==0):,} | "
          f"Falla: {sum(y_train_bal==1):,} (50.00%)")

    return X_train_bal, y_train_bal


# ── Guardar / Cargar datos procesados (data/processed/) ──────────────────────
def guardar_datos_procesados(
    X_train_bal: np.ndarray,
    y_train_bal: np.ndarray,
    X_test_s:    np.ndarray,
    y_test:      np.ndarray,
    feature_names: list,
) -> None:
    """
    Guarda los arrays preprocesados en la carpeta data/processed/
    usando las rutas definidas en config.json → paths.processed.
    """

    proc = cfg["paths"]["processed"]
    proc_dir = Path(cfg["paths"]["processed_dir"])
    proc_dir.mkdir(parents=True, exist_ok=True)

    np.save(proc_dir / Path(proc["X_train_bal"]).name, X_train_bal)
    np.save(proc_dir / Path(proc["y_train_bal"]).name, y_train_bal)
    np.save(proc_dir / Path(proc["X_test_s"]).name, X_test_s)
    np.save(proc_dir / Path(proc["y_test"]).name, y_test)

    with open(proc_dir / Path(proc["feature_names"]).name, "w", encoding="utf-8") as f:
        json.dump(feature_names, f, indent=2)

    print(f"\n  [Procesados] Datos guardados en: {proc_dir}")
    print(f"    - X_train_bal.npy: {X_train_bal.shape}")
    print(f"    - y_train_bal.npy: {y_train_bal.shape}")
    print(f"    - X_test_s.npy:    {X_test_s.shape}")
    print(f"    - y_test.npy:      {y_test.shape}")
    print(f"    - feature_names.json")


def cargar_datos_procesados() -> tuple:
    """
    Carga los arrays preprocesados desde data/processed/
    Retorna (X_train_bal, y_train_bal, X_test_s, y_test, feature_names)
    """

    proc = cfg["paths"]["processed"]
    proc_dir = Path(cfg["paths"]["processed_dir"])

    X_train_bal = np.load(proc_dir / Path(proc["X_train_bal"]).name)
    y_train_bal = np.load(proc_dir / Path(proc["y_train_bal"]).name)
    X_test_s    = np.load(proc_dir / Path(proc["X_test_s"]).name)
    y_test      = np.load(proc_dir / Path(proc["y_test"]).name)

    with open(proc_dir / Path(proc["feature_names"]).name, "r", encoding="utf-8") as f:
        feature_names = json.load(f)

    print(f"\n  [Procesados] Datos cargados desde: {proc_dir}")
    print(f"    - X_train_bal: {X_train_bal.shape}")
    print(f"    - y_train_bal: {y_train_bal.shape}")
    print(f"    - X_test_s:    {X_test_s.shape}")
    print(f"    - y_test:      {y_test.shape}")
    print(f"    - feature_names: {len(feature_names)} variables")

    return X_train_bal, y_train_bal, X_test_s, y_test, feature_names



