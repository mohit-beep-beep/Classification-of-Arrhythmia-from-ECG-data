-- SQLite schema for arrhythmia ML experiment analytics.
-- Metrics are written only from real sklearn evaluations — never hard-coded.

CREATE TABLE IF NOT EXISTS experiments (
  experiment_id   INTEGER PRIMARY KEY,
  run_name        TEXT NOT NULL,
  started_at      TEXT NOT NULL,
  finished_at     TEXT,
  dataset_path    TEXT NOT NULL,
  n_samples       INTEGER,
  n_features_raw  INTEGER,
  git_commit      TEXT,
  notes           TEXT
);

CREATE TABLE IF NOT EXISTS experiment_configs (
  config_id            INTEGER PRIMARY KEY,
  experiment_id        INTEGER NOT NULL REFERENCES experiments(experiment_id),
  variant              TEXT NOT NULL,
  test_size            REAL NOT NULL,
  random_state_split   INTEGER NOT NULL,
  random_state_model   INTEGER,
  variant_label        TEXT,
  imputer_strategy     TEXT NOT NULL,
  drop_column_index    INTEGER,
  scaler               TEXT,
  pca_variance         REAL,
  pca_n_components     INTEGER,
  oversampler          TEXT,
  oversample_scope     TEXT,
  train_rows_before    INTEGER,
  train_rows_after     INTEGER,
  test_rows            INTEGER
);

CREATE TABLE IF NOT EXISTS model_runs (
  run_id               INTEGER PRIMARY KEY,
  config_id            INTEGER NOT NULL REFERENCES experiment_configs(config_id),
  model_name           TEXT NOT NULL,
  sklearn_class        TEXT NOT NULL,
  hyperparameters_json TEXT NOT NULL,
  trained_at           TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS metrics (
  metric_id    INTEGER PRIMARY KEY,
  run_id       INTEGER NOT NULL REFERENCES model_runs(run_id),
  split        TEXT NOT NULL,
  metric_name  TEXT NOT NULL,
  metric_value REAL NOT NULL,
  UNIQUE(run_id, split, metric_name)
);

CREATE TABLE IF NOT EXISTS confusion_matrices (
  cm_id      INTEGER PRIMARY KEY,
  run_id     INTEGER NOT NULL REFERENCES model_runs(run_id),
  true_label INTEGER NOT NULL,
  pred_label INTEGER NOT NULL,
  count      INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS class_distributions (
  dist_id     INTEGER PRIMARY KEY,
  config_id   INTEGER NOT NULL REFERENCES experiment_configs(config_id),
  stage       TEXT NOT NULL,
  class_label INTEGER NOT NULL,
  count       INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_configs_variant ON experiment_configs(variant);
CREATE INDEX IF NOT EXISTS idx_configs_experiment ON experiment_configs(experiment_id);
CREATE INDEX IF NOT EXISTS idx_model_runs_config ON model_runs(config_id);
CREATE INDEX IF NOT EXISTS idx_model_runs_name ON model_runs(model_name);
CREATE INDEX IF NOT EXISTS idx_metrics_run ON metrics(run_id);
CREATE INDEX IF NOT EXISTS idx_metrics_name_split ON metrics(metric_name, split);
CREATE INDEX IF NOT EXISTS idx_cm_run ON confusion_matrices(run_id);
CREATE INDEX IF NOT EXISTS idx_class_dist_config ON class_distributions(config_id);
