<div align="center">

# Classification of Arrhythmia — ECG Data

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![SQLite](https://img.shields.io/badge/SQLite-Experiment%20Analytics-003B57?style=for-the-badge&logo=sqlite&logoColor=white)](https://www.sqlite.org/)
[![Dataset](https://img.shields.io/badge/Dataset-UCI%20ML%20Repository-blue?style=for-the-badge)](https://archive.ics.uci.edu/ml/datasets/Arrhythmia)
[![License](https://img.shields.io/badge/License-MIT-1abc9c?style=for-the-badge)](../LICENSE.md)

> Detecting the **presence or absence of cardiac arrhythmia** and classifying it into one of **16 groups** using classical ML algorithms, PCA, oversampling variants, and a **SQLite experiment analytics** layer on ECG signal data.

</div>

---

## Medical Disclaimer

> **This project is for educational and research purposes only.** It is not a substitute for clinical ECG interpretation or professional medical diagnosis.

---

## About the Project

ECG signals are the primary clinical tool for diagnosing heart conditions. This project applies classical ML algorithms to distinguish normal ECG readings from arrhythmia subtypes using the UCI Arrhythmia dataset (high dimensionality, severe class imbalance).

**What this project covers:**
- Preprocessing and EDA on a heavily imbalanced, high-dimensional tabular dataset
- Missing-value handling and PCA dimensionality reduction
- Class imbalance handling via **RandomOverSampler** and **SMOTE** (train-only)
- Training and comparing 6 classifiers across 4 experiment variants
- Persisting **real** evaluation metrics to SQLite for SQL analytics (optional Streamlit dashboard)

---

## Dataset

| Property | Details |
|----------|---------|
| **Source** | [UCI Machine Learning Repository — Arrhythmia Dataset](https://archive.ics.uci.edu/ml/datasets/Arrhythmia) |
| **File** | `Data/arrhythmia.csv` |
| **Samples** | 452 patient records |
| **Features** | 279 (age, sex, weight, height + ECG attributes) + class label |
| **Classes** | 16 codes (classes 11–13 absent in this file) |
| **Missing Values** | Encoded as `?`; column index 13 (`J`) dominates missingness |

---

## Methodology (executable pipeline)

```
Raw UCI CSV (Data/arrhythmia.csv)
        │
        ▼
  Preprocessing
  ├── Replace '?' with NaN
  ├── Drop column index 13 (J)
  └── Mean imputation (SimpleImputer)
        │
        ▼
  Stratified train/test split (80/20)
        │
        ▼
  Experiment variants
  ├── baseline     — StandardScaler only
  ├── pca          — Scaler → PCA(0.98 variance)
  ├── ros_pca      — RandomOverSampler (train) → Scaler → PCA(0.98)
  └── smote_pca    — SMOTE (train) → Scaler → PCA(0.98)
        │
        ▼
  Six classifiers
  ├── KNN, Logistic Regression, Decision Tree
  └── Linear SVC, Kernelized SVC, Random Forest
        │
        ▼
  Evaluation (sklearn): accuracy, weighted precision/recall/F1, confusion matrix
        │
        ▼
  SQLite (artifacts/experiments.db) → SQL analytics → optional Streamlit
```

Historical notebooks under `Model/`, `Preprocessing and EDA/`, and
`final with pca.ipynb` remain exploratory references. Immutable copies live in
[`notebooks_reference/`](notebooks_reference/). The package pipeline uses
**train-only** oversampling (no leakage) and fixed seeds
(`RANDOM_STATE_SPLIT=1`, `RANDOM_STATE_MODEL=0`).

---

## SQLite Experiment Analytics

Metrics written to SQLite are computed from that run’s predictions — they are **never fabricated or hard-coded**.

### Schema (summary)

- `experiments` — one pipeline invocation
- `experiment_configs` — variant settings (PCA variance, oversampler, row counts)
- `model_runs` — one trained classifier
- `metrics` — train/test scalar metrics
- `confusion_matrices` — flattened test confusion counts
- `class_distributions` — class counts by stage

Canonical DDL: [`sql/schema.sql`](sql/schema.sql). Example queries: [`sql/analytics_queries.sql`](sql/analytics_queries.sql).

### Run experiments

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Train all variants and write real metrics
PYTHONPATH=src python scripts/run_experiments.py

# Optional dashboard (read-only over SQLite)
streamlit run app/streamlit_app.py

# Tests
PYTHONPATH=src pytest -q
```

### Useful analytics (Python)

```python
from arrhythmia_ml import analytics
conn = analytics.load_db("artifacts/experiments.db")
print(analytics.leaderboard(conn))
print(analytics.best_model_per_variant(conn))
print(analytics.compare_model_across_variants(conn, model_name="Kernelized SVC"))
```

---

## Class Distribution

| Code | Class | Instances |
|:----:|-------|:---------:|
| 01 | Normal | 245 |
| 02 | Ischemic Changes (CAD) | 44 |
| 03 | Old Anterior MI | 15 |
| 04 | Old Inferior MI | 15 |
| 05 | Sinus Tachycardia | 13 |
| 06 | Sinus Bradycardia | 25 |
| 07 | PVC | 3 |
| 08 | Supraventricular Premature Contraction | 2 |
| 09 | Left Bundle Branch Block | 9 |
| 10 | Right Bundle Branch Block | 50 |
| 11–13 | AV Blocks | 0 |
| 14 | Left Ventricular Hypertrophy | 4 |
| 15 | Atrial Fibrillation or Flutter | 5 |
| 16 | Others | 22 |
| | **Total** | **452** |

---

## Project Structure

```
Classification-of-Arrhythmia-from-ECG-data/
├── Data/
│   └── arrhythmia.csv                 # UCI dataset (no header)
├── artifacts/
│   └── experiments.db                 # created by pipeline (gitignored)
├── notebooks_reference/               # immutable copies of original notebooks
├── src/arrhythmia_ml/
│   ├── config.py                      # fixed seeds + variant labels
│   ├── preprocess.py
│   ├── models.py
│   ├── pipeline.py
│   ├── evaluate.py
│   ├── db.py
│   └── analytics.py
├── scripts/run_experiments.py
├── sql/
│   ├── schema.sql
│   └── analytics_queries.sql
├── app/streamlit_app.py               # read-only dashboard
├── tests/
├── Preprocessing and EDA/             # exploratory notebooks (unchanged)
├── Model/                             # exploratory notebooks (unchanged)
├── Image/
├── final with pca.ipynb               # original consolidated notebook (preserved)
├── sql_experiment_analytics.ipynb     # SQL/pipeline hooks (does not overwrite originals)
├── requirements.txt
└── README.md
```

---

## Getting Started

### 1. Clone and install

```bash
git clone <your-fork-url>
cd Classification-of-Arrhythmia-from-ECG-data
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the ML + SQL pipeline

```bash
PYTHONPATH=src python scripts/run_experiments.py
```

### 3. Explore notebooks (optional)

```bash
# Original exploratory notebook (results preserved; not overwritten by SQL layer)
jupyter notebook "final with pca.ipynb"

# Reproducible SQL analytics notebook (writes to artifacts/experiments.db)
jupyter notebook sql_experiment_analytics.ipynb
```

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| Language | Python 3.9+ |
| ML | scikit-learn |
| Imbalance | imbalanced-learn (SMOTE, RandomOverSampler) |
| Dimensionality reduction | PCA (98% variance) |
| Experiment store | SQLite |
| Analytics UI | Streamlit (optional) |
| Data | Pandas, NumPy |
| Tests | pytest |

---

## References

- [UCI ML Repository — Arrhythmia Dataset](https://archive.ics.uci.edu/ml/datasets/Arrhythmia)
- Guvenir, H.A., et al. (1997). *A Supervised Machine Learning Algorithm for Arrhythmia Analysis.* Computers in Cardiology.
- [imbalanced-learn documentation](https://imbalanced-learn.org/)
- [scikit-learn PCA](https://scikit-learn.org/stable/modules/decomposition.html#pca)
