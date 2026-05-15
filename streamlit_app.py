"""IntentSight - Interactive Connection Readiness Dashboard.

Dynamic data product for exploring connection-readiness predictions,
model performance, feature importance, and scenario simulation.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix

from connection_scoring import add_connection_features
from feature_store import FeatureStore

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
PREPROCESSED_DIR = ROOT / "Preprocessed_Data_V2"
RESULTS_DIR = ROOT / "ML_Results"

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="IntentSight - Connection Readiness Explorer",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Feature descriptions from registry
# ---------------------------------------------------------------------------
_FEATURE_STORE = FeatureStore()
FEATURE_DESCRIPTIONS: dict[str, str] = {
    item["name"]: item["description"] for item in _FEATURE_STORE.to_dict()
}

# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------


@st.cache_resource
def load_model():
    path = RESULTS_DIR / "best_tuned_model.pkl"
    if not path.exists():
        return None
    return joblib.load(path)


@st.cache_resource
def load_stacking_model():
    path = RESULTS_DIR / "stacking_ensemble.pkl"
    if not path.exists():
        return None
    return joblib.load(path)


@st.cache_resource
def load_artifacts():
    artifacts = {}
    for name in ["target_encoder", "scaler", "selected_features"]:
        pkl_path = PREPROCESSED_DIR / f"{name}.pkl"
        if pkl_path.exists():
            artifacts[name] = joblib.load(pkl_path)
    return artifacts


@st.cache_data
def load_data():
    data = {}
    for name in [
        "X_train_selected_unresampled",
        "X_test_selected",
        "y_train_original",
        "y_test",
    ]:
        csv_path = PREPROCESSED_DIR / f"{name}.csv"
        if csv_path.exists():
            data[name] = pd.read_csv(csv_path)
    return data


@st.cache_data
def load_comparison():
    path = RESULTS_DIR / "final_comparison.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data
def load_nested_cv():
    path = RESULTS_DIR / "nested_cv_results.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data
def load_predictions_log():
    path = ROOT / "predictions_log.jsonl"
    if not path.exists():
        return []
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


@st.cache_data
def load_feature_importance():
    csv_path = RESULTS_DIR / "shap_feature_importance.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    csv_path = RESULTS_DIR / "final_comparison.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return None


@st.cache_data
def load_shap_values():
    path = RESULTS_DIR / "shap_values.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data
def load_shap_values_class(class_idx: int):
    path = RESULTS_DIR / f"shap_values_class{class_idx}.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data
def load_classification_report():
    path = RESULTS_DIR / "classification_report.txt"
    if not path.exists():
        return None
    text = path.read_text()
    pattern = r"^\s*(.+?)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)$"
    rows = []
    for line in text.splitlines():
        m = re.match(pattern, line)
        if m and m.group(1).strip() not in ("accuracy", "macro avg", "weighted avg"):
            rows.append(
                {
                    "Class": m.group(1).strip(),
                    "Precision": float(m.group(2)),
                    "Recall": float(m.group(3)),
                    "F1": float(m.group(4)),
                    "Support": int(m.group(5)),
                }
            )
    return pd.DataFrame(rows) if rows else None


@st.cache_data
def load_raw_dataset():
    for name in ["Behaviour_Extended_Dataset.csv", "Behaviour_Dataset.csv"]:
        path = ROOT / name
        if path.exists():
            return pd.read_csv(path), name
    return None, None


@st.cache_data
def load_train_sample(n: int = 5000):
    data = load_data()
    if "X_train_selected_unresampled" not in data:
        return None
    df = data["X_train_selected_unresampled"]
    return df.sample(n=min(n, len(df)), random_state=42)


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------

st.sidebar.title("IntentSight")
st.sidebar.markdown("**Connection Readiness Explorer**")
page = st.sidebar.radio(
    "Navigate",
    [
        "Overview",
        "Model Comparison",
        "Feature Importance",
        "Scenario Predictor",
        "Data Explorer",
        "Audit Log",
        "Insights & Diagnostics",
    ],
)
st.sidebar.markdown("---")
st.sidebar.markdown('WID3006 ML Group Assignment  \n"Tying the (Data) Knot"')


# ---------------------------------------------------------------------------
# Helper: class colour map
# ---------------------------------------------------------------------------
STAGE_COLORS = {
    "Likely To Connect": "#2ecc71",
    "Ready To Chat": "#27ae60",
    "Mostly Browsing": "#f39c12",
    "Swipes Too Freely": "#e74c3c",
    "Needs Profile Help": "#c0392b",
}


# ===========================================================================
# PAGE: Overview
# ===========================================================================


def page_overview():
    st.title("IntentSight - Overview")
    st.markdown(
        "Exploratory data product for classifying connection readiness from "
        "dating-app behaviour data.  \n"
        "Predictions are **product signals**, not claims about private intent."
    )

    comparison = load_comparison()
    data = load_data()
    report_df = load_classification_report()

    # --- KPI tiles with deltas ---
    col1, col2, col3, col4, col5 = st.columns(5)

    if comparison is not None:
        best = comparison.sort_values("Test Accuracy", ascending=False).iloc[0]
        baseline = comparison[comparison["Model"] == "Majority Baseline"]
        baseline_acc = (
            float(baseline["Test Accuracy"].iloc[0]) if len(baseline) > 0 else 0.0
        )
        best_acc = float(best["Test Accuracy"])
        worst_non_baseline = comparison[comparison["Model"] != "Majority Baseline"][
            "Test Accuracy"
        ]
        model_gap = float(worst_non_baseline.max() - worst_non_baseline.min())

        col1.metric("Best Model", str(best["Model"]))
        col2.metric(
            "Test Accuracy",
            f"{best_acc:.1%}",
            delta=f"+{(best_acc - baseline_acc):.1%} vs baseline",
        )
        col3.metric(
            "Weighted F1",
            f"{best['Test F1 (weighted)']:.1%}",
        )
        col4.metric("Model Gap", f"{model_gap:.1%}")

    if "y_test" in data:
        y_test = data["y_test"]
        n_classes = y_test.iloc[:, 0].nunique()
        col5.metric("Classes", str(n_classes))

    st.markdown("---")

    # --- Class distribution treemap with per-class metrics ---
    left_col, right_col = st.columns([1, 1])

    with left_col:
        if "y_train_original" in data and "target_encoder" in load_artifacts():
            st.subheader("Training Label Distribution")
            y_train = data["y_train_original"].iloc[:, 0]
            encoder = load_artifacts()["target_encoder"]
            labels = encoder.inverse_transform(y_train)
            dist = pd.Series(labels).value_counts().reset_index()
            dist.columns = ["Class", "Count"]
            dist["Pct"] = (dist["Count"] / dist["Count"].sum() * 100).round(1)

            if report_df is not None:
                dist = dist.merge(report_df[["Class", "F1"]], on="Class", how="left")
                dist["hover"] = dist.apply(
                    lambda r: (
                        f"{r['Class']}<br>"
                        f"Count: {r['Count']:,}<br>"
                        f"Share: {r['Pct']}%<br>"
                        f"F1: {r.get('F1', 'N/A')}"
                    ),
                    axis=1,
                )
                fig = px.treemap(
                    dist,
                    path=["Class"],
                    values="Count",
                    color="F1",
                    color_continuous_scale="RdYlGn",
                    range_color=[0.9, 1.0],
                    hover_data={"F1": ":.3f", "Pct": ":.1f"},
                )
            else:
                fig = px.treemap(dist, path=["Class"], values="Count")

            fig.update_layout(height=400, margin=dict(t=10, l=10, r=10, b=10))
            st.plotly_chart(fig, width="stretch")

    with right_col:
        if report_df is not None:
            st.subheader("Per-Class Performance")
            for _, row in report_df.iterrows():
                c1, c2, c3 = st.columns(3)
                c1.markdown(f"**{row['Class']}**")
                c2.metric("Precision", f"{row['Precision']:.0%}")
                c3.metric("Recall", f"{row['Recall']:.0%}")

    # --- Parallel coordinates for multi-metric comparison ---
    if comparison is not None:
        st.subheader("Multi-Metric Model Comparison")
        plot_df = comparison[comparison["Model"] != "Majority Baseline"].copy()
        if len(plot_df) > 0:
            fig = px.parallel_coordinates(
                plot_df,
                dimensions=[
                    "Test Accuracy",
                    "Test F1 (weighted)",
                    "CV Accuracy (mean)",
                    "Train Time (s)",
                ],
                color="Test Accuracy",
                color_continuous_scale="viridis",
                labels={
                    "Test Accuracy": "Accuracy",
                    "Test F1 (weighted)": "F1",
                    "CV Accuracy (mean)": "CV Mean",
                    "Train Time (s)": "Time (s)",
                },
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, width="stretch")

    # --- Insight card ---
    if comparison is not None:
        best = comparison.sort_values("Test Accuracy", ascending=False).iloc[0]
        baseline = comparison[comparison["Model"] == "Majority Baseline"]
        baseline_acc = (
            float(baseline["Test Accuracy"].iloc[0]) if len(baseline) > 0 else 0.0
        )
        best_acc = float(best["Test Accuracy"])
        boosting = comparison[
            comparison["Model"].str.contains("CatBoost|LightGBM|XGBoost", regex=True)
        ]
        non_boosting = comparison[
            ~comparison["Model"].str.contains(
                "CatBoost|LightGBM|XGBoost|Majority", regex=True
            )
        ]
        if len(boosting) > 0 and len(non_boosting) > 0:
            boost_avg = boosting["Test Accuracy"].mean()
            other_avg = non_boosting["Test Accuracy"].mean()
            gap = boost_avg - other_avg
            st.info(
                f"**Key Insight:** {best['Model']} achieves {best_acc:.1%} accuracy, "
                f"a {(best_acc - baseline_acc) * 100:.1f}pp improvement over the "
                f"majority baseline. Boosting models outperform non-boosting by "
                f"{gap * 100:.1f}pp on average, indicating strong signal in the data."
            )


# ===========================================================================
# PAGE: Model Comparison
# ===========================================================================


def page_model_comparison():
    st.title("Model Comparison")

    comparison = load_comparison()
    nested = load_nested_cv()

    if comparison is not None:
        st.subheader("Base Model Results")

        display_df = comparison.copy()
        st.dataframe(
            display_df.style.format(
                {
                    "CV Accuracy (mean)": "{:.4f}",
                    "CV Accuracy (std)": "{:.4f}",
                    "Test Accuracy": "{:.4f}",
                    "Test F1 (weighted)": "{:.4f}",
                    "Train Time (s)": "{:.1f}",
                }
            ),
            width="stretch",
        )

        # --- Radar chart ---
        st.subheader("Model Radar Comparison")
        plot_df = comparison[comparison["Model"] != "Majority Baseline"].copy()
        if len(plot_df) > 0:
            # Normalize train time (inverse — lower is better)
            max_time = plot_df["Train Time (s)"].max()
            if max_time > 0:
                plot_df["Speed Score"] = 1 - (plot_df["Train Time (s)"] / max_time)
            else:
                plot_df["Speed Score"] = 1.0

            categories = [
                "Test Accuracy",
                "Test F1 (weighted)",
                "CV Accuracy (mean)",
                "Speed Score",
            ]
            fig = go.Figure()
            for _, row in plot_df.iterrows():
                values = [row[c] for c in categories]
                values.append(values[0])  # close the polygon
                fig.add_trace(
                    go.Scatterpolar(
                        r=values,
                        theta=categories + [categories[0]],
                        fill="toself",
                        name=row["Model"],
                        opacity=0.6,
                    )
                )
            fig.update_layout(
                height=500,
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                showlegend=True,
            )
            st.plotly_chart(fig, width="stretch")

        # --- Interactive calibration curve ---
        st.subheader("Calibration Curves (Interactive)")
        model = load_model()
        artifacts = load_artifacts()
        data = load_data()

        if (
            model is not None
            and "X_test_selected" in data
            and "y_test" in data
            and "target_encoder" in artifacts
        ):
            X_test = data["X_test_selected"]
            y_test = data["y_test"].iloc[:, 0]
            encoder = artifacts["target_encoder"]

            proba = model.predict_proba(X_test)
            n_classes = proba.shape[1]
            class_names = encoder.inverse_transform(range(n_classes))

            fig = go.Figure()
            for cls_idx in range(n_classes):
                y_binary = (y_test == cls_idx).astype(int)
                frac_pos, mean_pred = calibration_curve(
                    y_binary, proba[:, cls_idx], n_bins=10, strategy="uniform"
                )
                fig.add_trace(
                    go.Scatter(
                        x=mean_pred,
                        y=frac_pos,
                        mode="lines+markers",
                        name=class_names[cls_idx],
                        line=dict(color=STAGE_COLORS.get(class_names[cls_idx], None)),
                    )
                )
            fig.add_trace(
                go.Scatter(
                    x=[0, 1],
                    y=[0, 1],
                    mode="lines",
                    name="Perfectly calibrated",
                    line=dict(dash="dash", color="gray"),
                )
            )
            fig.update_layout(
                height=500,
                xaxis_title="Mean predicted probability",
                yaxis_title="Fraction of positives",
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
                ),
            )
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("Model artifacts not available for calibration curves.")

    if nested is not None:
        st.subheader("Nested Cross-Validation (Outer 5-Fold)")
        st.dataframe(
            nested.style.format(
                {
                    "Nested CV Mean": "{:.4f}",
                    "Nested CV Std": "{:.4f}",
                }
            ),
            width="stretch",
        )

    # --- Insight card ---
    if comparison is not None:
        best = comparison.sort_values("Test Accuracy", ascending=False).iloc[0]
        worst = (
            comparison[comparison["Model"] != "Majority Baseline"]
            .sort_values("Test Accuracy")
            .iloc[0]
        )
        st.success(
            f"**Winner: {best['Model']}** — {float(best['Test Accuracy']):.1%} "
            f"accuracy, {float(best['Test F1 (weighted)']):.1%} F1. "
            f"Worst non-baseline: {worst['Model']} at "
            f"{float(worst['Test Accuracy']):.1%}. "
            f"Gap: {(float(best['Test Accuracy']) - float(worst['Test Accuracy'])) * 100:.1f}pp."
        )


# ===========================================================================
# PAGE: Feature Importance
# ===========================================================================


def page_feature_importance():
    st.title("Feature Importance")

    importance_df = load_feature_importance()
    shap_df = load_shap_values()
    data = load_data()

    tab_shap, tab_builtin, tab_corr = st.tabs(
        ["SHAP Analysis", "Built-in Importance", "Correlations"]
    )

    with tab_shap:
        if shap_df is not None:
            # --- Interactive SHAP beeswarm ---
            st.subheader("SHAP Beeswarm (Interactive)")

            artifacts = load_artifacts()
            encoder = artifacts.get("target_encoder")
            n_classes = len(encoder.classes_) if encoder is not None else 5
            class_names = (
                encoder.inverse_transform(range(n_classes))
                if encoder is not None
                else [f"Class {i}" for i in range(n_classes)]
            )
            selected_class = st.selectbox(
                "Select class for SHAP values",
                range(n_classes),
                format_func=lambda i: (
                    class_names[i] if i < len(class_names) else f"Class {i}"
                ),
            )

            cls_shap = load_shap_values_class(selected_class)
            if cls_shap is not None:
                train_sample = load_train_sample(len(cls_shap))
                _render_shap_beeswarm(
                    cls_shap, train_sample, selected_class, class_names
                )
            else:
                # Fallback to combined SHAP
                _render_shap_beeswarm(
                    shap_df, load_train_sample(len(shap_df)), 0, class_names
                )

            # --- SHAP bar chart ---
            st.subheader("Feature Importance (Mean |SHAP|)")
            if importance_df is not None and "mean_abs_shap" in importance_df.columns:
                top_n = st.slider("Top N features", 5, 66, 20, key="shap_top_n")
                top = importance_df.head(top_n)
                fig = px.bar(
                    top.sort_values("mean_abs_shap"),
                    x="mean_abs_shap",
                    y="feature",
                    orientation="h",
                    color="mean_abs_shap",
                    color_continuous_scale="viridis",
                    hover_data={"mean_abs_shap": ":.4f"},
                )
                fig.update_layout(
                    height=max(300, top_n * 25),
                    xaxis_title="Mean |SHAP|",
                    yaxis_title="",
                )
                st.plotly_chart(fig, width="stretch")

                # Feature insights
                top3 = importance_df.head(3)["feature"].tolist()
                engineered_in_top = [
                    f
                    for f in top3
                    if FEATURE_DESCRIPTIONS.get(f, "").startswith("Bounded")
                    or FEATURE_DESCRIPTIONS.get(f, "").startswith("Composite")
                    or "score" in f.lower()
                    or "quality" in f.lower()
                ]
                if engineered_in_top:
                    st.info(
                        f"**Insight:** The top features ({', '.join(top3)}) are all "
                        f"engineered composites — the model relies on domain-knowledge "
                        f"feature engineering, not just raw behaviour columns."
                    )
        else:
            st.info("SHAP values not found. Run `python compute_shap_values.py` first.")

    with tab_builtin:
        model = load_model()
        if model is not None:
            st.subheader("CatBoost Native Feature Importance")
            try:
                calibrated = model.named_steps["model"]
                cb_model = calibrated.calibrated_classifiers_[0].estimator
                feat_names = list(
                    load_data()
                    .get("X_train_selected_unresampled", pd.DataFrame())
                    .columns
                )
                if feat_names:
                    native_imp = cb_model.get_feature_importance()
                    imp_df = pd.DataFrame(
                        {"feature": feat_names, "importance": native_imp}
                    ).sort_values("importance", ascending=False)
                    top_n = st.slider("Top N features", 5, 66, 20, key="builtin_top_n")
                    top = imp_df.head(top_n)
                    fig = px.bar(
                        top.sort_values("importance"),
                        x="importance",
                        y="feature",
                        orientation="h",
                        color="importance",
                        color_continuous_scale="plasma",
                    )
                    fig.update_layout(
                        height=max(300, top_n * 25),
                        xaxis_title="Importance",
                        yaxis_title="",
                    )
                    st.plotly_chart(fig, width="stretch")
            except Exception as e:
                st.warning(f"Could not extract native importance: {e}")
        fi_path = PREPROCESSED_DIR / "feature_importances.png"
        if fi_path.exists():
            st.image(str(fi_path), width="stretch")

    with tab_corr:
        if "X_train_selected_unresampled" in data:
            st.subheader("Feature Correlation Matrix")
            X = data["X_train_selected_unresampled"]
            numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

            if importance_df is not None and "feature" in importance_df.columns:
                default_cols = [
                    f
                    for f in importance_df["feature"].head(15).tolist()
                    if f in numeric_cols
                ]
            else:
                default_cols = numeric_cols[:15]

            selected_cols = st.multiselect(
                "Select features for correlation matrix",
                numeric_cols,
                default=default_cols,
                max_selections=25,
            )

            if len(selected_cols) >= 2:
                corr = X[selected_cols].corr()
                fig = px.imshow(
                    corr,
                    color_continuous_scale="RdBu_r",
                    zmin=-1,
                    zmax=1,
                    text_auto=".2f",
                )
                fig.update_layout(
                    height=max(400, len(selected_cols) * 30),
                )
                st.plotly_chart(fig, width="stretch")

                # Find highest/lowest correlations
                mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
                corr_pairs = corr.where(mask).stack().reset_index()
                corr_pairs.columns = ["Feature A", "Feature B", "Correlation"]
                if len(corr_pairs) > 0:
                    highest = corr_pairs.loc[corr_pairs["Correlation"].abs().idxmax()]
                    lowest = corr_pairs.loc[corr_pairs["Correlation"].abs().idxmin()]
                    c1, c2 = st.columns(2)
                    c1.metric(
                        "Strongest Correlation",
                        f"{highest['Correlation']:.3f}",
                        f"{highest['Feature A']} <-> {highest['Feature B']}",
                    )
                    c2.metric(
                        "Weakest Correlation",
                        f"{lowest['Correlation']:.3f}",
                        f"{lowest['Feature A']} <-> {lowest['Feature B']}",
                    )


def _render_shap_beeswarm(
    shap_df: pd.DataFrame,
    feature_df: pd.DataFrame | None,
    class_idx: int,
    class_names: list[str],
) -> None:
    """Render an interactive SHAP beeswarm plot."""
    mean_abs = shap_df.abs().mean().sort_values(ascending=False)
    top_features = mean_abs.head(20).index.tolist()

    # Sample for performance
    n_plot = min(500, len(shap_df))
    plot_idx = np.random.RandomState(42).choice(len(shap_df), n_plot, replace=False)
    shap_sample = shap_df.iloc[plot_idx][top_features]

    records = []
    for feat in top_features:
        for i in range(n_plot):
            val = float(shap_sample[feat].iloc[i])
            feat_val = (
                float(feature_df[feat].iloc[plot_idx[i]])
                if feature_df is not None and feat in feature_df.columns
                else 0.0
            )
            records.append(
                {"Feature": feat, "SHAP Value": val, "Feature Value": feat_val}
            )

    plot_data = pd.DataFrame(records)

    fig = px.scatter(
        plot_data,
        x="SHAP Value",
        y="Feature",
        color="Feature Value",
        color_continuous_scale="RdBu_r",
        opacity=0.5,
        hover_data={"SHAP Value": ":.3f", "Feature Value": ":.2f"},
    )
    fig.update_layout(
        height=max(400, len(top_features) * 25),
        xaxis_title=f"SHAP Value ({class_names[class_idx] if class_idx < len(class_names) else f'Class {class_idx}'})",
        yaxis_title="",
    )
    st.plotly_chart(fig, width="stretch")


# ===========================================================================
# PAGE: Scenario Predictor
# ===========================================================================


def page_scenario_predictor():
    st.title("Scenario Predictor")
    st.markdown(
        "Simulate a user profile and see the predicted connection stage.  \n"
        "Move the sliders — predictions update automatically."
    )

    model = load_model()
    artifacts = load_artifacts()

    if model is None:
        st.error("Model artifacts not found in `ML_Results/`. Run `train.py` first.")
        return

    if not all(
        k in artifacts for k in ["target_encoder", "scaler", "selected_features"]
    ):
        st.error("Preprocessing artifacts missing from `Preprocessed_Data_V2/`.")
        return

    encoder = artifacts["target_encoder"]
    scaler = artifacts["scaler"]
    selected_features = artifacts["selected_features"]

    scaler_obj = artifacts["scaler"]
    if hasattr(scaler_obj, "feature_names_in_"):
        full_columns = list(scaler_obj.feature_names_in_)
    else:
        full_columns = list(artifacts["selected_features"])

    # Input form
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Profile Inputs")
        app_usage = st.slider("App Usage (min/day)", 0, 1000, 120)
        swipe_ratio = st.slider("Swipe Right Ratio", 0.0, 1.0, 0.5, 0.01)
        likes_received = st.slider("Likes Received", 0, 10000, 50)
        mutual_matches = st.slider("Mutual Matches", 0, 10000, 10)
        msg_sent = st.slider("Messages Sent", 0, 10000, 30)
        bio_length = st.slider("Bio Length (chars)", 0, 5000, 140)

    with col_right:
        st.subheader("Physical & Activity")
        emoji_rate = st.slider("Emoji Usage Rate", 0.0, 10.0, 0.3, 0.1)
        height_cm = st.slider("Height (cm)", 80, 250, 170)
        weight_kg = st.slider("Weight (kg)", 20, 300, 70)
        profile_pics = st.slider("Profile Pictures", 0, 50, 3)
        last_active = st.slider("Last Active Hour", 0, 23, 12)

    # Build input vector
    input_df = pd.DataFrame(np.zeros((1, len(full_columns))), columns=full_columns)

    values = {
        "app_usage_time_min": app_usage,
        "swipe_right_ratio": swipe_ratio,
        "likes_received": likes_received,
        "mutual_matches": mutual_matches,
        "message_sent_count": msg_sent,
        "bio_length": bio_length,
        "emoji_usage_rate": emoji_rate,
        "height_cm": height_cm,
        "weight_kg": weight_kg,
        "profile_pics_count": profile_pics,
        "last_active_hour": last_active,
    }

    for col, val in values.items():
        if col in input_df.columns:
            input_df.at[0, col] = val

    engineered = add_connection_features(input_df)
    for col in full_columns:
        if col in engineered.columns:
            input_df[col] = engineered[col]

    scaled = pd.DataFrame(scaler.transform(input_df), columns=full_columns)
    selected = scaled[selected_features]

    # Predict
    encoded = model.predict(selected).astype(int)
    label = encoder.inverse_transform(encoded)[0]

    st.markdown("---")

    # Result with color-coded gauge
    if label == "Likely To Connect":
        st.success(f"**Predicted Stage: {label}**")
        gauge_color = "#2ecc71"
    elif label == "Ready To Chat":
        st.success(f"**Predicted Stage: {label}**")
        gauge_color = "#27ae60"
    elif label == "Mostly Browsing":
        st.warning(f"**Predicted Stage: {label}**")
        gauge_color = "#f39c12"
    elif label == "Swipes Too Freely":
        st.error(f"**Predicted Stage: {label}**")
        gauge_color = "#e74c3c"
    else:
        st.info(f"**Predicted Stage: {label}**")
        gauge_color = "#c0392b"

    # Probabilities + gauge
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(selected)
        prob_df = pd.DataFrame(
            {
                "Class": encoder.inverse_transform(range(proba.shape[1])),
                "Probability": proba[0],
            }
        ).sort_values("Probability", ascending=False)

        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Class Probabilities")
            fig = px.bar(
                prob_df,
                x="Probability",
                y="Class",
                orientation="h",
                color="Class",
                color_discrete_map=STAGE_COLORS,
            )
            fig.update_layout(
                height=300,
                xaxis_range=[0, 1],
                showlegend=False,
            )
            st.plotly_chart(fig, width="stretch")

        with col_b:
            confidence = float(np.max(proba))
            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=confidence * 100,
                    title={"text": "Confidence (%)"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": gauge_color},
                        "steps": [
                            {"range": [0, 50], "color": "lightcoral"},
                            {"range": [50, 80], "color": "lightyellow"},
                            {"range": [80, 100], "color": "lightgreen"},
                        ],
                    },
                )
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, width="stretch")

    # --- Comparison with dataset averages ---
    st.subheader("Your Profile vs Dataset Averages")
    train_data = load_data()
    if "X_train_selected_unresampled" in train_data:
        train_df = train_data["X_train_selected_unresampled"]
        compare_features = [f for f in values.keys() if f in train_df.columns]
        if compare_features:
            compare_records = []
            for feat in compare_features:
                user_val = values[feat]
                mean_val = float(train_df[feat].mean())
                std_val = float(train_df[feat].std())
                compare_records.append(
                    {
                        "Feature": feat,
                        "Your Value": user_val,
                        "Dataset Mean": mean_val,
                        "Z-Score": (user_val - mean_val) / std_val
                        if std_val > 0
                        else 0,
                    }
                )
            compare_df = pd.DataFrame(compare_records)

            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    name="Your Value",
                    y=compare_df["Feature"],
                    x=compare_df["Your Value"],
                    orientation="h",
                    marker_color="#3498db",
                )
            )
            fig.add_trace(
                go.Bar(
                    name="Dataset Mean",
                    y=compare_df["Feature"],
                    x=compare_df["Dataset Mean"],
                    orientation="h",
                    marker_color="#95a5a6",
                )
            )
            fig.update_layout(
                height=350,
                barmode="group",
                xaxis_title="Value",
            )
            st.plotly_chart(fig, width="stretch")

            # Highlight outliers
            outliers = compare_df[compare_df["Z-Score"].abs() > 1.5]
            if len(outliers) > 0:
                outlier_text = ", ".join(
                    f"{r['Feature']} (z={r['Z-Score']:.1f})"
                    for _, r in outliers.iterrows()
                )
                st.warning(f"**Unusual values:** {outlier_text}")

    # --- Sensitivity analysis ---
    st.subheader("Sensitivity Analysis")
    st.markdown("What single change would most affect the prediction?")

    base_proba = (
        model.predict_proba(selected)[0] if hasattr(model, "predict_proba") else None
    )
    if base_proba is not None:
        sensitivity_records = []
        perturb_pct = 0.10

        for feat_name, feat_val in values.items():
            if feat_val == 0:
                perturb_up = 1.0
                perturb_down = -1.0
            else:
                perturb_up = feat_val * (1 + perturb_pct)
                perturb_down = feat_val * (1 - perturb_pct)

            for direction, new_val in [("up", perturb_up), ("down", perturb_down)]:
                test_input = input_df.copy()
                if feat_name in test_input.columns:
                    test_input.at[0, feat_name] = new_val
                test_eng = add_connection_features(test_input)
                for col in full_columns:
                    if col in test_eng.columns:
                        test_input[col] = test_eng[col]
                test_scaled = pd.DataFrame(
                    scaler.transform(test_input), columns=full_columns
                )
                test_selected = test_scaled[selected_features]
                test_proba = model.predict_proba(test_selected)[0]
                prob_change = float(np.max(test_proba) - np.max(base_proba))
                sensitivity_records.append(
                    {
                        "Feature": feat_name,
                        "Direction": direction,
                        "Probability Change": prob_change,
                    }
                )

        sens_df = pd.DataFrame(sensitivity_records)
        # Aggregate: max absolute change per feature
        agg = (
            sens_df.groupby("Feature")["Probability Change"]
            .apply(lambda x: x.abs().max())
            .reset_index()
        )
        agg.columns = ["Feature", "Max Impact"]
        agg = agg.sort_values("Max Impact", ascending=True).tail(10)

        fig = px.bar(
            agg,
            x="Max Impact",
            y="Feature",
            orientation="h",
            color="Max Impact",
            color_continuous_scale="OrRd",
        )
        fig.update_layout(
            height=350,
            xaxis_title="Max probability change (±10% perturbation)",
            yaxis_title="",
        )
        st.plotly_chart(fig, width="stretch")

    # OOD warning
    ood_features = []
    for col in selected.columns:
        val = float(selected[col].iloc[0])
        if "X_train_selected_unresampled" in load_data():
            train_col = load_data()["X_train_selected_unresampled"]
            if col in train_col.columns:
                mean = train_col[col].mean()
                std = train_col[col].std()
                if std > 0 and abs(val - mean) / std > 3:
                    ood_features.append(col)

    if ood_features:
        st.warning(
            f"**Out-of-distribution warning:** The following features "
            f"are >3 std from training mean: {', '.join(ood_features)}. "
            f"Prediction may be unreliable."
        )


# ===========================================================================
# PAGE: Data Explorer
# ===========================================================================


def page_data_explorer():
    st.title("Data Explorer")

    raw, raw_name = load_raw_dataset()
    if raw is None:
        st.error("No behaviour dataset found.")
        return

    st.markdown(
        f"**Source:** `{raw_name}`  |  **Rows:** {len(raw):,}  |  "
        f"**Columns:** {raw.shape[1]}"
    )

    # --- Interactive filters ---
    st.subheader("Filters")
    filter_cols = st.columns(4)

    filtered = raw.copy()

    with filter_cols[0]:
        if "gender" in raw.columns:
            genders = st.multiselect(
                "Gender", raw["gender"].unique().tolist(), default=None
            )
            if genders:
                filtered = filtered[filtered["gender"].isin(genders)]

    with filter_cols[1]:
        if "location_type" in raw.columns:
            locations = st.multiselect(
                "Location", raw["location_type"].unique().tolist(), default=None
            )
            if locations:
                filtered = filtered[filtered["location_type"].isin(locations)]

    with filter_cols[2]:
        if "relationship_intent" in raw.columns:
            intents = st.multiselect(
                "Intent",
                raw["relationship_intent"].unique().tolist(),
                default=None,
            )
            if intents:
                filtered = filtered[filtered["relationship_intent"].isin(intents)]

    with filter_cols[3]:
        if "app_usage_time_min" in raw.columns:
            min_usage, max_usage = st.slider(
                "App Usage (min)",
                float(raw["app_usage_time_min"].min()),
                float(raw["app_usage_time_min"].max()),
                (
                    float(raw["app_usage_time_min"].min()),
                    float(raw["app_usage_time_min"].max()),
                ),
            )
            filtered = filtered[
                (filtered["app_usage_time_min"] >= min_usage)
                & (filtered["app_usage_time_min"] <= max_usage)
            ]

    st.markdown(f"**Filtered:** {len(filtered):,} rows")

    tab_dist, tab_scatter, tab_insights, tab_preview = st.tabs(
        ["Distributions", "Scatter Explorer", "Insights", "Preview"]
    )

    with tab_dist:
        numeric_cols = filtered.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            selected_col = st.selectbox("Select feature", numeric_cols, key="dist_col")
            group_by = st.selectbox(
                "Group by",
                ["None"]
                + [
                    c
                    for c in ["relationship_intent", "gender", "location_type"]
                    if c in filtered.columns
                ],
                key="dist_group",
            )
            if selected_col:
                color_col = None if group_by == "None" else group_by
                fig = px.violin(
                    filtered,
                    y=selected_col,
                    color=color_col,
                    box=True,
                    points=False,
                    title=f"Distribution of {selected_col}",
                )
                fig.update_layout(height=450)
                st.plotly_chart(fig, width="stretch")

    with tab_scatter:
        numeric_cols = filtered.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = [
            c
            for c in ["relationship_intent", "gender", "location_type"]
            if c in filtered.columns
        ]
        if len(numeric_cols) >= 2:
            c1, c2, c3 = st.columns(3)
            with c1:
                x_col = st.selectbox("X axis", numeric_cols, index=0, key="scatter_x")
            with c2:
                y_col = st.selectbox(
                    "Y axis",
                    numeric_cols,
                    index=min(1, len(numeric_cols) - 1),
                    key="scatter_y",
                )
            with c3:
                color_col = st.selectbox(
                    "Color by", ["None"] + cat_cols, key="scatter_color"
                )

            sample_n = min(5000, len(filtered))
            scatter_data = filtered.sample(n=sample_n, random_state=42)
            fig = px.scatter(
                scatter_data,
                x=x_col,
                y=y_col,
                color=None if color_col == "None" else color_col,
                opacity=0.5,
                title=f"{x_col} vs {y_col} (n={sample_n:,})",
                hover_data=cat_cols[:3],
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, width="stretch")

    with tab_insights:
        st.subheader("Auto-Generated Insights")
        insights = _generate_insights(filtered)
        for insight in insights:
            st.markdown(f"- {insight}")

    with tab_preview:
        st.dataframe(filtered.head(100), width="stretch")
        st.dataframe(filtered.describe(), width="stretch")


def _generate_insights(df: pd.DataFrame) -> list[str]:
    """Generate automatic insight cards from the dataset."""
    insights = []

    if "swipe_right_ratio" in df.columns:
        high_swipe = df[df["swipe_right_ratio"] > 0.7]
        low_swipe = df[df["swipe_right_ratio"] <= 0.3]
        if len(high_swipe) > 100 and len(low_swipe) > 100:
            insights.append(
                f"**High swipers** (ratio > 0.7): {len(high_swipe):,} users "
                f"({len(high_swipe) / len(df) * 100:.1f}%). "
                f"Low swipers (ratio < 0.3): {len(low_swipe):,} users."
            )

    if "app_usage_time_min" in df.columns and "mutual_matches" in df.columns:
        corr = df["app_usage_time_min"].corr(df["mutual_matches"])
        if abs(corr) > 0.1:
            direction = "positive" if corr > 0 else "negative"
            insights.append(
                f"App usage and mutual matches have a {direction} "
                f"correlation (r={corr:.2f})."
            )

    if "bio_length" in df.columns:
        zero_bio = df[df["bio_length"] == 0]
        if len(zero_bio) > 0:
            insights.append(
                f"**{len(zero_bio):,} users** ({len(zero_bio) / len(df) * 100:.1f}%) "
                f"have a bio length of 0 — an empty profile."
            )

    if "profile_pics_count" in df.columns:
        zero_pics = df[df["profile_pics_count"] == 0]
        if len(zero_pics) > 0:
            insights.append(f"**{len(zero_pics):,} users** have 0 profile pictures.")

    if "message_sent_count" in df.columns:
        median_msgs = df["message_sent_count"].median()
        insights.append(
            f"Median messages sent: {median_msgs:.0f}. "
            f"Top 10% senders average "
            f"{df['message_sent_count'].quantile(0.9):.0f} messages."
        )

    if "emoji_usage_rate" in df.columns:
        high_emoji = df[df["emoji_usage_rate"] > 1.0]
        if len(high_emoji) > 0:
            insights.append(
                f"**{len(high_emoji):,} users** use more than 1 emoji per message "
                f"(emoji rate > 1.0)."
            )

    return (
        insights if insights else ["No notable patterns detected in the filtered data."]
    )


# ===========================================================================
# PAGE: Audit Log
# ===========================================================================


def page_audit_log():
    st.title("Prediction Audit Log")
    st.markdown("History of all predictions made through the API and dashboard.")

    records = load_predictions_log()

    if not records:
        st.info(
            "No predictions logged yet. Make a prediction via the API or Scenario Predictor."
        )
        return

    # Parse into DataFrame
    rows = []
    for rec in records:
        payload = rec.get("payload", {})
        result = rec.get("result", {})
        ood = result.get("ood_flags", {})
        rows.append(
            {
                "timestamp": rec.get("timestamp", ""),
                "prediction": result.get("prediction", ""),
                "confidence": result.get("confidence", None),
                "app_usage": payload.get("app_usage_time_min", None),
                "swipe_ratio": payload.get("swipe_right_ratio", None),
                "n_ood_flags": len(ood) if isinstance(ood, (list, dict)) else 0,
            }
        )

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    st.subheader(f"Total Predictions: {len(df)}")

    # --- Timeline chart ---
    if df["timestamp"].notna().any() and "confidence" in df.columns:
        st.subheader("Prediction Timeline")
        fig = px.scatter(
            df,
            x="timestamp",
            y="confidence",
            color="prediction",
            hover_data=["app_usage", "swipe_ratio"],
            title="Predictions Over Time",
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, width="stretch")

    # --- Prediction distribution + confidence ---
    col1, col2 = st.columns(2)
    with col1:
        if "prediction" in df.columns:
            pred_counts = df["prediction"].value_counts().reset_index()
            pred_counts.columns = ["Prediction", "Count"]
            fig = px.pie(pred_counts, names="Prediction", values="Count", hole=0.35)
            st.plotly_chart(fig, width="stretch")

    with col2:
        if "confidence" in df.columns and df["confidence"].notna().any():
            fig = px.histogram(
                df,
                x="confidence",
                nbins=20,
                title="Confidence Distribution",
                color="prediction",
            )
            st.plotly_chart(fig, width="stretch")

    # --- OOD flag analysis ---
    if "n_ood_flags" in df.columns and df["n_ood_flags"].sum() > 0:
        st.subheader("Out-of-Distribution Flags")
        ood_data = []
        for rec in records:
            ood = rec.get("result", {}).get("ood_flags", {})
            if isinstance(ood, list):
                for feat in ood:
                    ood_data.append({"feature": feat, "count": 1})
            elif isinstance(ood, dict):
                for feat, flagged in ood.items():
                    if flagged:
                        ood_data.append({"feature": feat, "count": 1})

        if ood_data:
            ood_df = pd.DataFrame(ood_data)
            ood_counts = ood_df.groupby("feature").sum().reset_index()
            ood_counts = ood_counts.sort_values("count", ascending=True).tail(10)
            fig = px.bar(
                ood_counts,
                x="count",
                y="feature",
                orientation="h",
                title="Most Frequently OOD Features",
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, width="stretch")

    # --- Recent predictions table ---
    st.subheader("Recent Predictions")
    st.dataframe(df.tail(50).iloc[::-1], width="stretch")

    # --- Export ---
    csv_data = df.to_csv(index=False)
    st.download_button(
        "Download Audit Log as CSV",
        csv_data,
        "audit_log.csv",
        "text/csv",
    )


# ===========================================================================
# PAGE: Insights & Diagnostics
# ===========================================================================


def page_insights_diagnostics():
    st.title("Insights & Diagnostics")
    st.markdown("Deep-dive into model behaviour, drift detection, and error patterns.")

    model = load_model()
    data = load_data()
    artifacts = load_artifacts()

    tab_drift, tab_confidence, tab_errors, tab_interactions = st.tabs(
        [
            "Drift Detection",
            "Confidence Analysis",
            "Error Analysis",
            "Feature Interactions",
        ]
    )

    with tab_drift:
        _render_drift_tab(data)

    with tab_confidence:
        _render_confidence_tab(model, data, artifacts)

    with tab_errors:
        _render_error_tab(model, data, artifacts)

    with tab_interactions:
        _render_interactions_tab(model, data, artifacts)


def _render_drift_tab(data: dict) -> None:
    """Drift detection between train and test sets."""
    st.subheader("Train vs Test Distribution Drift")

    if "X_train_selected_unresampled" not in data or "X_test_selected" not in data:
        st.info("Train/test data not available for drift analysis.")
        return

    train = data["X_train_selected_unresampled"]
    test = data["X_test_selected"]
    common_cols = [c for c in train.columns if c in test.columns]

    drift_records = []
    for col in common_cols:
        train_vals = train[col].dropna()
        test_vals = test[col].dropna()

        # PSI computation
        n_bins = 10
        breakpoints = np.linspace(
            min(train_vals.min(), test_vals.min()),
            max(train_vals.max(), test_vals.max()),
            n_bins + 1,
        )
        train_hist = np.histogram(train_vals, bins=breakpoints)[0] / len(train_vals)
        test_hist = np.histogram(test_vals, bins=breakpoints)[0] / len(test_vals)

        # Avoid division by zero
        train_hist = np.clip(train_hist, 1e-6, None)
        test_hist = np.clip(test_hist, 1e-6, None)

        psi = float(np.sum((test_hist - train_hist) * np.log(test_hist / train_hist)))

        # KS test
        from scipy.stats import ks_2samp

        ks_stat, ks_p = ks_2samp(train_vals, test_vals)

        drift_detected = psi > 0.25 or ks_p < 0.05
        drift_records.append(
            {
                "Feature": col,
                "PSI": psi,
                "KS Statistic": ks_stat,
                "KS p-value": ks_p,
                "Drift": "Yes" if drift_detected else "No",
            }
        )

    drift_df = pd.DataFrame(drift_records).sort_values("PSI", ascending=False)

    # Summary metrics
    n_drift = (drift_df["Drift"] == "Yes").sum()
    c1, c2, c3 = st.columns(3)
    c1.metric("Features Analysed", len(drift_df))
    c2.metric("Drift Detected", f"{n_drift} features")
    c3.metric("Max PSI", f"{drift_df['PSI'].max():.3f}")

    # Drift scatter
    fig = px.scatter(
        drift_df,
        x="PSI",
        y="KS p-value",
        color="Drift",
        color_discrete_map={"Yes": "#e74c3c", "No": "#2ecc71"},
        hover_data=["Feature", "PSI", "KS Statistic"],
        title="Drift Landscape (PSI vs KS p-value)",
    )
    fig.add_hline(
        y=0.05, line_dash="dash", line_color="red", annotation_text="KS p=0.05"
    )
    fig.add_vline(
        x=0.25, line_dash="dash", line_color="red", annotation_text="PSI=0.25"
    )
    fig.update_layout(height=450)
    st.plotly_chart(fig, width="stretch")

    # Table
    st.dataframe(
        drift_df.style.format(
            {"PSI": "{:.4f}", "KS Statistic": "{:.4f}", "KS p-value": "{:.4f}"}
        ),
        width="stretch",
        height=400,
    )


def _render_confidence_tab(model, data: dict, artifacts: dict) -> None:
    """Model confidence analysis on the test set."""
    st.subheader("Confidence Analysis")

    if model is None or "X_test_selected" not in data or "y_test" not in data:
        st.info("Model or test data not available.")
        return

    if "target_encoder" not in artifacts:
        st.info("Target encoder not available.")
        return

    X_test = data["X_test_selected"]
    y_test = data["y_test"].iloc[:, 0]
    encoder = artifacts["target_encoder"]

    proba = model.predict_proba(X_test)
    preds = model.predict(X_test).astype(int)
    max_proba = np.max(proba, axis=1)
    correct = preds == y_test.values

    # Confidence histogram by correctness
    conf_df = pd.DataFrame(
        {
            "Confidence": max_proba,
            "Correct": ["Correct" if c else "Incorrect" for c in correct],
        }
    )
    fig = px.histogram(
        conf_df,
        x="Confidence",
        color="Correct",
        barmode="overlay",
        opacity=0.7,
        nbins=30,
        title="Prediction Confidence Distribution",
        color_discrete_map={"Correct": "#2ecc71", "Incorrect": "#e74c3c"},
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, width="stretch")

    # Per-class confidence
    class_names = encoder.inverse_transform(range(proba.shape[1]))
    class_conf = []
    for cls_idx in range(proba.shape[1]):
        mask = y_test == cls_idx
        if mask.sum() > 0:
            mean_conf = float(proba[mask, cls_idx].mean())
            acc = float((preds[mask] == cls_idx).mean())
            class_conf.append(
                {
                    "Class": class_names[cls_idx],
                    "Mean Confidence": mean_conf,
                    "Accuracy": acc,
                }
            )

    if class_conf:
        conf_summary = pd.DataFrame(class_conf)
        st.subheader("Per-Class Confidence")
        for _, row in conf_summary.iterrows():
            c1, c2, c3 = st.columns(3)
            c1.markdown(f"**{row['Class']}**")
            c2.metric("Mean Confidence", f"{row['Mean Confidence']:.1%}")
            c3.metric("Accuracy", f"{row['Accuracy']:.1%}")

    # Confusion zone
    unsure = conf_df[conf_df["Confidence"] < 0.7]
    if len(unsure) > 0:
        st.warning(
            f"**{len(unsure):,} predictions** ({len(unsure) / len(conf_df) * 100:.1f}%) "
            f"have confidence below 70% — the model is uncertain."
        )


def _render_error_tab(model, data: dict, artifacts: dict) -> None:
    """Confusion matrix and error analysis."""
    st.subheader("Error Analysis")

    if model is None or "X_test_selected" not in data or "y_test" not in data:
        st.info("Model or test data not available.")
        return

    if "target_encoder" not in artifacts:
        st.info("Target encoder not available.")
        return

    X_test = data["X_test_selected"]
    y_test = data["y_test"].iloc[:, 0]
    encoder = artifacts["target_encoder"]

    preds = model.predict(X_test).astype(int)
    class_names = encoder.inverse_transform(range(len(encoder.classes_)))

    # Confusion matrix
    cm = confusion_matrix(y_test, preds)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    text = []
    for i in range(len(cm)):
        row = []
        for j in range(len(cm)):
            row.append(f"{cm[i][j]:,}<br>({cm_pct[i][j]:.1f}%)")
        text.append(row)

    fig = go.Figure(
        go.Heatmap(
            z=cm,
            x=class_names,
            y=class_names,
            text=text,
            texttemplate="%{text}",
            colorscale="Blues",
            hovertemplate="True: %{y}<br>Predicted: %{x}<br>Count: %{z:,}<extra></extra>",
        )
    )
    fig.update_layout(
        height=450,
        xaxis_title="Predicted",
        yaxis_title="True",
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig, width="stretch")

    # Sankey diagram: true -> predicted flows
    st.subheader("Misclassification Flow")
    n_classes = len(class_names)

    # Build Sankey links for misclassifications only
    source, target, value, link_color = [], [], [], []
    for i in range(n_classes):
        for j in range(n_classes):
            if i != j and cm[i][j] > 0:
                source.append(i)
                target.append(n_classes + j)  # offset predicted labels
                value.append(cm[i][j])
                link_color.append("rgba(231,76,60,0.3)")

    if source:
        node_labels = list(class_names) + [f"Pred: {n}" for n in class_names]
        node_colors = [STAGE_COLORS.get(n, "#95a5a6") for n in class_names] + [
            STAGE_COLORS.get(n, "#95a5a6") for n in class_names
        ]

        fig = go.Figure(
            go.Sankey(
                node=dict(
                    label=node_labels,
                    color=node_colors,
                    pad=15,
                    thickness=20,
                ),
                link=dict(
                    source=source,
                    target=target,
                    value=value,
                    color=link_color,
                ),
            )
        )
        fig.update_layout(
            height=400, title="True Class -> Predicted Class (Errors Only)"
        )
        st.plotly_chart(fig, width="stretch")

    # Per-class error breakdown
    error_records = []
    for i in range(n_classes):
        total_errors = cm[i].sum() - cm[i][i]
        if total_errors > 0:
            worst_j = np.argmax([cm[i][j] if j != i else 0 for j in range(n_classes)])
            error_records.append(
                {
                    "True Class": class_names[i],
                    "Total Errors": int(total_errors),
                    "Most Confused With": class_names[worst_j],
                    "Confusion Count": int(cm[i][worst_j]),
                }
            )

    if error_records:
        st.dataframe(
            pd.DataFrame(error_records), width="stretch", hide_index=True
        )


def _render_interactions_tab(model, data: dict, artifacts: dict) -> None:
    """Feature interaction heatmaps for top features."""
    st.subheader("Feature Interaction Effects")

    if model is None or "X_train_selected_unresampled" not in data:
        st.info("Model or training data not available.")
        return

    if "selected_features" not in artifacts:
        st.info("Selected features not available.")
        return

    X_train = data["X_train_selected_unresampled"]
    selected_features = artifacts["selected_features"]

    importance_df = load_feature_importance()
    if importance_df is not None and "feature" in importance_df.columns:
        top_features = importance_df["feature"].head(6).tolist()
    else:
        top_features = selected_features[:6]

    top_features = [f for f in top_features if f in X_train.columns]

    if len(top_features) < 2:
        st.info("Not enough features for interaction analysis.")
        return

    col1, col2 = st.columns(2)
    with col1:
        feat_a = st.selectbox("Feature A", top_features, index=0, key="interact_a")
    with col2:
        feat_b = st.selectbox(
            "Feature B",
            [f for f in top_features if f != feat_a],
            index=0,
            key="interact_b",
        )

    if feat_a and feat_b:
        # Create a grid of predictions holding other features at median
        n_grid = 20
        a_vals = np.linspace(X_train[feat_a].min(), X_train[feat_a].max(), n_grid)
        b_vals = np.linspace(X_train[feat_b].min(), X_train[feat_b].max(), n_grid)

        # Use median of all features as baseline
        baseline = X_train[selected_features].median().to_dict()

        scaler = artifacts.get("scaler")
        encoder = artifacts.get("target_encoder")

        if scaler is not None and encoder is not None:
            grid_proba = np.zeros((n_grid, n_grid))

            for i, av in enumerate(a_vals):
                for j, bv in enumerate(b_vals):
                    row = baseline.copy()
                    row[feat_a] = av
                    row[feat_b] = bv
                    grid_df = pd.DataFrame([row])[selected_features]

                    try:
                        proba = model.predict_proba(grid_df)[0]
                        grid_proba[i, j] = np.max(proba)
                    except Exception:
                        grid_proba[i, j] = np.nan

            fig = px.imshow(
                grid_proba,
                x=np.round(b_vals, 2),
                y=np.round(a_vals, 2),
                color_continuous_scale="Viridis",
                labels={"x": feat_b, "y": feat_a, "color": "Max Probability"},
                title=f"Prediction Confidence: {feat_a} vs {feat_b}",
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, width="stretch")

            st.info(
                "This heatmap shows the model's maximum predicted probability "
                "across all classes for each combination of the two features, "
                "holding all other features at their median values."
            )


# ===========================================================================
# Router
# ===========================================================================

PAGES = {
    "Overview": page_overview,
    "Model Comparison": page_model_comparison,
    "Feature Importance": page_feature_importance,
    "Scenario Predictor": page_scenario_predictor,
    "Data Explorer": page_data_explorer,
    "Audit Log": page_audit_log,
    "Insights & Diagnostics": page_insights_diagnostics,
}

PAGES[page]()
