# Notebook reference copies

Immutable snapshots of the project's original Jupyter notebooks, taken from
`git HEAD` before the SQLite analytics layer was added.

**Do not edit these files** when iterating on the ML/SQL pipeline. Use them to
compare historical exploratory results against reproducible package runs in
`src/arrhythmia_ml/` and `sql_experiment_analytics.ipynb`.

| Path | Role |
|------|------|
| `final with pca.ipynb` | Consolidated exploratory pipeline |
| `Model/*.ipynb` | Model / oversampling experiments |
| `Preprocessing and EDA/*.ipynb` | Cleaning and EDA |

Working copies of the same notebooks remain at the repository root (and under
`Model/`, `Preprocessing and EDA/`) for historical browsing. New experiment
logging lives in the Python package and `sql_experiment_analytics.ipynb`, not
by overwriting these reference results.
