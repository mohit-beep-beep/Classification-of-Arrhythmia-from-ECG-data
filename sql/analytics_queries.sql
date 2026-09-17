-- Useful analytics queries for arrhythmia ML experiments.
-- All metric values come from real pipeline runs stored in experiments.db.

-- 1) Leaderboard: all runs ordered by test F1
SELECT e.run_name, c.variant, m.model_name, met.metric_value AS test_f1
FROM metrics met
JOIN model_runs m ON m.run_id = met.run_id
JOIN experiment_configs c ON c.config_id = m.config_id
JOIN experiments e ON e.experiment_id = c.experiment_id
WHERE met.split = 'test' AND met.metric_name = 'f1_weighted'
ORDER BY met.metric_value DESC;

-- 2) Best model per variant (test F1)
WITH ranked AS (
  SELECT e.run_name, c.variant, m.model_name, met.metric_value AS test_f1,
         ROW_NUMBER() OVER (
           PARTITION BY e.experiment_id, c.variant
           ORDER BY met.metric_value DESC
         ) AS rn
  FROM metrics met
  JOIN model_runs m ON m.run_id = met.run_id
  JOIN experiment_configs c ON c.config_id = m.config_id
  JOIN experiments e ON e.experiment_id = c.experiment_id
  WHERE met.split = 'test' AND met.metric_name = 'f1_weighted'
)
SELECT run_name, variant, model_name, test_f1
FROM ranked
WHERE rn = 1
ORDER BY variant;

-- 3) Compare Kernelized SVC across variants
SELECT e.run_name, c.variant, met.metric_value AS test_f1
FROM metrics met
JOIN model_runs m ON m.run_id = met.run_id
JOIN experiment_configs c ON c.config_id = m.config_id
JOIN experiments e ON e.experiment_id = c.experiment_id
WHERE m.model_name = 'Kernelized SVC'
  AND met.split = 'test'
  AND met.metric_name = 'f1_weighted'
ORDER BY c.variant;

-- 4) Train–test accuracy gap (overfitting signal)
SELECT c.variant, m.model_name,
       train.metric_value AS train_accuracy,
       test.metric_value AS test_accuracy,
       (train.metric_value - test.metric_value) AS accuracy_gap
FROM model_runs m
JOIN experiment_configs c ON c.config_id = m.config_id
JOIN metrics train
  ON train.run_id = m.run_id
 AND train.split = 'train'
 AND train.metric_name = 'accuracy'
JOIN metrics test
  ON test.run_id = m.run_id
 AND test.split = 'test'
 AND test.metric_name = 'accuracy'
ORDER BY accuracy_gap DESC;

-- 5) PCA components by variant
SELECT c.variant, c.pca_variance, c.pca_n_components,
       c.train_rows_before, c.train_rows_after, c.test_rows
FROM experiment_configs c
WHERE c.pca_n_components IS NOT NULL
ORDER BY c.variant;

-- 6) Class balance impact (train before vs after oversampling)
SELECT c.variant, d.stage, d.class_label, d.count
FROM class_distributions d
JOIN experiment_configs c ON c.config_id = d.config_id
WHERE d.stage IN ('train_before', 'train_after')
ORDER BY c.variant, d.stage, d.class_label;

-- 7) Confusion hotspots (highest off-diagonal counts)
SELECT c.variant, m.model_name, cm.true_label, cm.pred_label, cm.count
FROM confusion_matrices cm
JOIN model_runs m ON m.run_id = cm.run_id
JOIN experiment_configs c ON c.config_id = m.config_id
WHERE cm.true_label != cm.pred_label
ORDER BY cm.count DESC
LIMIT 20;

-- 8) Latest experiment summary
SELECT e.run_name, e.started_at, c.variant, m.model_name, met.metric_value AS test_f1
FROM experiments e
JOIN experiment_configs c ON c.experiment_id = e.experiment_id
JOIN model_runs m ON m.config_id = c.config_id
JOIN metrics met ON met.run_id = m.run_id
WHERE e.experiment_id = (SELECT MAX(experiment_id) FROM experiments)
  AND met.split = 'test'
  AND met.metric_name = 'f1_weighted'
ORDER BY c.variant, met.metric_value DESC;
