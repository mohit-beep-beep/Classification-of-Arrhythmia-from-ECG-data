"""Streamlit dashboard for arrhythmia ML experiment analytics (read-only)."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from arrhythmia_ml import analytics  # noqa: E402

DEFAULT_DB = ROOT / "artifacts" / "experiments.db"


def main():
    st.set_page_config(page_title="Arrhythmia Experiment Analytics", layout="wide")
    st.title("Arrhythmia ML Experiment Analytics")
    st.caption(
        "Read-only views over SQLite results from the training pipeline. "
        "This app never trains models or writes metrics."
    )

    db_path = st.sidebar.text_input("SQLite database path", str(DEFAULT_DB))
    path = Path(db_path)

    if not path.exists():
        st.warning(
            f"Database not found at `{path}`.\n\n"
            "Run the pipeline first:\n\n"
            "```bash\npython scripts/run_experiments.py\n```"
        )
        return

    conn = analytics.load_db(path, read_only=True)

    experiments = pd.read_sql_query(
        "SELECT experiment_id, run_name, started_at, finished_at FROM experiments "
        "ORDER BY experiment_id DESC",
        conn,
    )
    if experiments.empty:
        st.info("No experiments in the database yet.")
        conn.close()
        return

    exp_id = st.sidebar.selectbox(
        "Experiment",
        experiments["experiment_id"].tolist(),
        format_func=lambda i: (
            f"{i} — "
            f"{experiments.loc[experiments.experiment_id == i, 'run_name'].iloc[0]}"
        ),
    )

    st.subheader("Leaderboard (test F1)")
    board = analytics.leaderboard(conn, experiment_id=exp_id)
    st.dataframe(board, use_container_width=True)

    st.subheader("Best model per variant")
    best = analytics.best_model_per_variant(conn, experiment_id=exp_id)
    st.dataframe(best, use_container_width=True)

    st.subheader("Kernelized SVC across variants")
    svc = analytics.compare_model_across_variants(
        conn, model_name="Kernelized SVC", experiment_id=exp_id
    )
    if not svc.empty:
        st.bar_chart(svc.set_index("variant")["score"])
    else:
        st.write("No Kernelized SVC rows found.")

    st.subheader("Train–test accuracy gap")
    gaps = analytics.train_test_accuracy_gap(conn, experiment_id=exp_id)
    st.dataframe(gaps.head(20), use_container_width=True)

    st.subheader("PCA components")
    pca_df = analytics.pca_components_by_variant(conn, experiment_id=exp_id)
    st.dataframe(pca_df, use_container_width=True)

    st.subheader("Confusion hotspots (off-diagonal)")
    hot = analytics.confusion_hotspots(conn, experiment_id=exp_id, limit=20)
    st.dataframe(hot, use_container_width=True)

    if not hot.empty:
        run_ids = hot["run_id"].unique().tolist()
        run_id = st.selectbox("Confusion matrix run_id", run_ids)
        cm = pd.read_sql_query(
            """
            SELECT true_label, pred_label, count
            FROM confusion_matrices
            WHERE run_id = ?
            """,
            conn,
            params=[int(run_id)],
        )
        if not cm.empty:
            pivot = cm.pivot_table(
                index="true_label",
                columns="pred_label",
                values="count",
                fill_value=0,
            )
            st.write("Confusion matrix heatmap data")
            st.dataframe(pivot, use_container_width=True)
            try:
                import matplotlib.pyplot as plt
                import seaborn as sns

                fig, ax = plt.subplots(figsize=(6, 5))
                sns.heatmap(pivot, annot=True, fmt="g", cmap="Blues", ax=ax)
                ax.set_xlabel("Predicted")
                ax.set_ylabel("True")
                st.pyplot(fig)
            except Exception as exc:  # pragma: no cover
                st.caption(f"Heatmap unavailable: {exc}")

    conn.close()


if __name__ == "__main__":
    main()
