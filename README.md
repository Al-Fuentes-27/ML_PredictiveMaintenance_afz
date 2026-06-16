# 🏭 AI-Driven Predictive Maintenance Pipeline (AI4I 2020)

<div align="center">

# 🏭 AI-Driven Predictive Maintenance Pipeline

[![🚀 LIVE INTERACTIVE DASHBOARD](https://img.shields.io/badge/🚀_LIVE_DEMO-Interactive_Dashboard-0078D4?style=for-the-badge&logo=google-chrome&logoColor=white)](https://al-fuentes-27.github.io/ML_PredictiveMaintenance_afz/dashboard/dashboard.html)

*Click the button above to explore the live dashboard (Dark/Light mode, dynamic charts, and business insights).*

</div>

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.8-orange?logo=scikit-learn&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-Interactive-3F4F75?logo=plotly&logoColor=white)
![Methodology](https://img.shields.io/badge/Methodology-CRISP--DM-green)
![Status](https://img.shields.io/badge/Status-Production_Ready-success)

An end-to-end Machine Learning pipeline designed to anticipate industrial machine failures before they occur. Built upon the **AI4I 2020 Predictive Maintenance Dataset** (UCI), this project translates raw sensor telemetry into actionable maintenance insights, achieving an **88.2% Recall** in failure detection.

> 📊 **[View the Live Interactive Dashboard](./dashboard/dashboard.html)** | 📄 **[Read the Full Technical Report (PDF)](./docs/aldo%20fuentes%20zaldivar_informe%20t%C3%A9cnico_completo%20final.pdf)**

---

## 💡 Business Insights & Key Findings

- 🎯 **Early Detection:** The Random Forest model correctly identifies **88.2%** of actual machine failures before they happen, allowing for scheduled preventive maintenance and drastically reducing unplanned downtime.
- ⚠️ **Critical Predictors:** **High Torque (> 45.95 Nm)** and **Process Temperature (> 309.5 K)** are the strongest physical indicators of imminent mechanical failure.
- 📉 **The Imbalance Challenge:** Only **3.39%** of operations result in failure. Without advanced resampling techniques like **SMOTE**, traditional models would ignore these critical minority cases, yielding a near 0% Recall.

---

## 🧪 Methodology

This project strictly adheres to industry and scientific standards:
1. **CRISP-DM Framework:** Structured across all 6 phases (Business Understanding, Data Understanding, Data Preparation, Modeling, Evaluation, Deployment).
2. **Mario Bunge's Scientific Method:** The model is treated as a hypothesis ("*The model will achieve a Recall ≥ 80%*") which is rigorously tested and **not falsified** against unseen, imbalanced real-world test data.
3. **Bloom's Taxonomy:** Documentation and code comments are structured to ensure clear knowledge transfer and reproducibility.

---

## Best Model

**Random Forest** — Recall=0.8824, AUC-ROC=0.9710, F1=0.5063

---

## 🚀 Quickstart Guide

Follow these steps to replicate the pipeline and generate the interactive dashboard locally.

### Dataset

Matzka, S. (2020). *AI4I 2020 Predictive Maintenance Dataset*.
UCI Machine Learning Repository. https://doi.org/10.24432/C5HS5C

### Prerequisites
- Python 3.10+
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/Al-Fuentes-27/ML_PredictiveMaintenance_afz.git
cd ML_PredictiveMaintenance_afz```

### 2. Set Up the Environment

Create a virtual environment and install the required dependencies. The project uses a local utils package for modularization, which is installed in editable mode.

macOS / Linux:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .  # Installs the local 'utils' package
```

Windows:
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

### 3. Run the Pipeline
Execute the scripts sequentially from the root directory. Each script handles its own directory creation and outputs.
```bash
# 1. Exploratory Data Analysis & Statistical Figures
python src/01_exploratory_analysis.py

# 2. Data Preprocessing (Scaling & SMOTE)
python src/02_preprocessing.py

# 3. Model Training (Run all three)
python src/03_train_logistic_regression.py
python src/03_train_decision_tree.py
python src/03_train_random_forest.py

# 4. Model Evaluation & Hypothesis Testing
python src/04_evaluate_models.py

# 5. Consolidate Results & Metrics
python src/05_save_results.py

# 6. Generate the Interactive HTML Dashboard
python src/06_html_dashboard.py
```

### 4. View the Dashboard
Open the generated HTML file in your preferred web browser:

```bash
# macOS
open dashboard/dashboard.html

# Windows
start dashboard/dashboard.html

# Linux
xdg-open dashboard/dashboard.html
```
(The dashboard features a Dark/Light mode toggle, dynamic confusion matrices, and educational tooltips).

---

## 📂 Project Structure

```
ML_PredictiveMaintenance_afz/
│
├── 📄 README.md                 ← You are here
├── 📄 requirements.txt          ← Python dependencies
├── 📄 setup.py                  ← Local 'utils' package installer
│
├── 📁 data/
│   ├── raw/                     ← Original dataset (dataset_ai4i2020.csv)
│   ├── processed/               ← Numpy arrays (Train/Test/SMOTE)
│   └── dashboard_metrics/       ← JSON payload for the HTML dashboard
│
├── 📁 src/                      ← Modular Python Pipeline
│   ├── 01_exploratory_analysis.py
│   ├── 02_preprocessing.py
│   ├── 03_train_*.py            ← Model training scripts
│   ├── 04_evaluate_models.py
│   ├── 05_save_results.py
│   ├── 06_html_dashboard.py     ← Plotly + Tailwind CSS generator
│   └── utils/                   ← Reusable functions (config, metrics, etc.)
│
├── 📁 models/                   ← Serialized .pkl models & scalers
├── 📁 results/                  ← Metrics (.json, .csv) & Figures (.png)
├── 📁 docs/                     ← Technical Reports (PDF)
└── 📁 dashboard/                ← Final HTML Interactive Dashboard
```

## 🛠️ Tech Stack

| Category | Technologies |
| :--- | :--- |
| **Language** | Python 3.12 |
| **Data Manipulation** | Pandas, Numpy |
| **Machine Learning** | Scikit-Learn, Imbalanced-Learn (SMOTE) |
| **Visualization** | Matplotlib, Seaborn, Plotly.js |
| **Frontend / Dashboard** | HTML5, Tailwind CSS, Vanilla JavaScript |
| **Methodologies** | CRISP-DM, Bunge's Scientific Method |

---

## 👨‍🔬 Author

**Aldo Fuentes Zaldivar**  
*Mechanical Engineering | UAEMex*  
Specializing in Data Science, Machine Learning, and Industrial Maintenance.

<!--[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://www.linkedin.com/in/aldo-fuentes-zaldivar/)-->
[![GitHub](https://img.shields.io/badge/GitHub-Al--Fuentes--27-black?logo=github)](https://github.com/Al-Fuentes-27)

---

## 📚 References

- Matzka, S. (2020). *AI4I 2020 Predictive Maintenance Dataset*. UCI Machine Learning Repository. [DOI: 10.24432/C5HS5C](https://doi.org/10.24432/C5HS5C)
- Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5-32.
- Wirth, R., & Hipp, J. (2000). CRISP-DM: Towards a standard process model for data mining.

