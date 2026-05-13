"""IntentSight — Streamlit Dashboard (Tier 1 Deployment).

Interactive data product for exploring user-intent predictions,
model performance, feature importance, and scenario simulation.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

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
    page_title="IntentSight — User Intent Explorer",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded",
)

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


# ---------------------------------------------------------------------------
# Feature importance (from SHAP or saved CSV)
# ---------------------------------------------------------------------------

@st.cache_data
def load_feature_importance():
    csv_path = RESULTS_DIR / "final_comparison.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return None


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------

st.sidebar.title("IntentSight")
st.sidebar.markdown("**User Intent Signal Explorer**")
page = st.sidebar.radio(
    "Navigate",
    [
        "Overview",
        "Model Comparison",
        "Feature Importance",
        "Scenario Predictor",
        "Data Explorer",
        "Audit Log",
    ],
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    "WID3006 ML Group Assignment  \n"
    "\"Tying the (Data) Knot\""
)


# ===========================================================================
# PAGE: Overview
# ===========================================================================

def page_overview():
    st.title("IntentSight — Overview")
    st.markdown(
        "Exploratory data product for classifying user intent from "
        "dating-app behaviour data.  \n"
        "Predictions are **exploratory signals**, not ground truth — "
        "model accuracy is comparable to the majority-class baseline."
    )

    comparison = load_comparison()
    data = load_data()

    # KPI tiles
    col1, col2, col3, col4 = st.columns(4)

    if comparison is not None:
        best = comparison.sort_values("Test Accuracy", ascending=False).iloc[0]
        col1.metric("Best Model", str(best["Model"]))
        col2.metric("Test Accuracy", f"{best['Test Accuracy']:.4f}")
        col3.metric("Weighted F1", f"{best['Test F1 (weighted)']:.4f}")

    if "y_test" in data:
        y_test = data["y_test"]
        n_classes = y_test.iloc[:, 0].nunique()
        col4.metric("Classes", str(n_classes))

    st.markdown("---")

    # Class distribution
    if "y_train_original" in data and "target_encoder" in load_artifacts():
        st.subheader("Training Label Distribution")
        y_train = data["y_train_original"].iloc[:, 0]
        encoder = load_artifacts()["target_encoder"]
        labels = encoder.inverse_transform(y_train)
        dist = pd.Series(labels).value_counts().reset_index()
        dist.columns = ["Class", "Count"]
        fig = px.pie(dist, names="Class", values="Count", hole=0.35)
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    # Model comparison bar
    if comparison is not None:
        st.subheader("All Models — Test Accuracy")
        comp_sorted = comparison.sort_values("Test Accuracy", ascending=True)
        fig = px.bar(
            comp_sorted,
            x="Test Accuracy",
            y="Model",
            orientation="h",
            color="Test Accuracy",
            color_continuous_scale="viridis",
        )
        fig.update_layout(height=max(300, len(comp_sorted) * 40))
        st.plotly_chart(fig, use_container_width=True)


# ===========================================================================
# PAGE: Model Comparison
# ===========================================================================

def page_model_comparison():
    st.title("Model Comparison")

    comparison = load_comparison()
    nested = load_nested_cv()

    if comparison is not None:
        st.subheader("Base Model Results")
        st.dataframe(
            comparison.style.format({
                "CV Accuracy (mean)": "{:.4f}",
                "CV Accuracy (std)": "{:.4f}",
                "Test Accuracy": "{:.4f}",
                "Test F1 (weighted)": "{:.4f}",
                "Train Time (s)": "{:.1f}",
            }),
            use_container_width=True,
        )

        # Accuracy vs F1 scatter
        plot_df = comparison[comparison["Test F1 (weighted)"] > 0].copy()
        if len(plot_df) > 0:
            fig = px.scatter(
                plot_df,
                x="Test Accuracy",
                y="Test F1 (weighted)",
                text="Model",
                size="Train Time (s)",
                color="Model",
                title="Accuracy vs F1 (bubble = train time)",
            )
            fig.update_traces(textposition="top center")
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)

    if nested is not None:
        st.subheader("Nested Cross-Validation (Outer 5-Fold)")
        st.dataframe(
            nested.style.format({
                "Nested CV Mean": "{:.4f}",
                "Nested CV Std": "{:.4f}",
            }),
            use_container_width=True,
        )

    # Calibration plot
    cal_path = RESULTS_DIR / "calibration_plot.png"
    if cal_path.exists():
        st.subheader("Calibration Plot")
        st.image(str(cal_path), use_container_width=True)


# ===========================================================================
# PAGE: Feature Importance
# ===========================================================================

def page_feature_importance():
    st.title("Feature Importance")

    importance_df = load_feature_importance()
    data = load_data()

    # SHAP plots
    shap_summary = RESULTS_DIR / "shap_summary.png"
    shap_bar = RESULTS_DIR / "shap_bar.png"

    tab_shap, tab_builtin = st.tabs(["SHAP Analysis", "Built-in Importance"])

    with tab_shap:
        if shap_summary.exists():
            st.subheader("SHAP Beeswarm Plot")
            st.image(str(shap_summary), use_container_width=True)
        if shap_bar.exists():
            st.subheader("SHAP Bar Plot")
            st.image(str(shap_bar), use_container_width=True)
        if not shap_summary.exists() and not shap_bar.exists():
            st.info("SHAP plots not found. Run the ML pipeline first.")

    with tab_builtin:
        fi_path = PREPROCESSED_DIR / "feature_importances.png"
        if fi_path.exists():
            st.image(str(fi_path), use_container_width=True)

        if importance_df is not None:
            st.subheader("Selected Features (Unbiased)")
            st.dataframe(importance_df, use_container_width=True)

    # Feature correlation heatmap
    if "X_train_selected_unresampled" in data:
        st.subheader("Feature Correlation Matrix")
        X = data["X_train_selected_unresampled"]
        numeric_cols = X.select_dtypes(include=[np.number]).columns[:15]
        corr = X[numeric_cols].corr()
        fig = px.imshow(
            corr,
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            title="Top 15 Features — Correlation",
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)


# ===========================================================================
# PAGE: Scenario Predictor
# ===========================================================================

def page_scenario_predictor():
    st.title("Scenario Predictor")
    st.markdown(
        "Simulate a user profile and see the predicted intent.  \n"
        "Predictions are exploratory — fields left at default may "
        "produce unreliable interaction features."
    )

    model = load_model()
    artifacts = load_artifacts()

    if model is None:
        st.error(
            "Model artifacts not found in `ML_Results/`. "
            "Run `train.py` first."
        )
        return

    if not all(k in artifacts for k in ["target_encoder", "scaler", "selected_features"]):
        st.error("Preprocessing artifacts missing from `Preprocessed_Data_V2/`.")
        return

    encoder = artifacts["target_encoder"]
    scaler = artifacts["scaler"]
    selected_features = artifacts["selected_features"]

    # Derive full columns from scaler or selected features
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

    if st.button("Predict Intent", type="primary", use_container_width=True):
        # Build input vector
        input_df = pd.DataFrame(
            np.zeros((1, len(full_columns))), columns=full_columns
        )

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

        # Derived features
        if height_cm > 0:
            values["bmi"] = weight_kg / ((height_cm / 100) ** 2)
        values["match_rate"] = mutual_matches / (likes_received + 1)
        values["msg_per_match"] = msg_sent / (mutual_matches + 1)
        values["engagement_score"] = app_usage * swipe_ratio * emoji_rate

        for col, val in values.items():
            if col in input_df.columns:
                input_df.at[0, col] = val

        # Scale and select
        scaled = pd.DataFrame(
            scaler.transform(input_df), columns=full_columns
        )
        selected = scaled[selected_features]

        # Predict
        encoded = model.predict(selected).astype(int)
        label = encoder.inverse_transform(encoded)[0]

        # Display result
        st.markdown("---")
        if label == "Dating":
            st.success(f"**Predicted Intent: {label}**")
        else:
            st.info(f"**Predicted Intent: {label}**")

        # Probabilities
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(selected)
            prob_df = pd.DataFrame({
                "Class": encoder.inverse_transform(range(proba.shape[1])),
                "Probability": proba[0],
            }).sort_values("Probability", ascending=False)

            col_a, col_b = st.columns(2)
            with col_a:
                st.subheader("Class Probabilities")
                st.dataframe(
                    prob_df.style.format({"Probability": "{:.4f}"}),
                    use_container_width=True,
                    hide_index=True,
                )
            with col_b:
                confidence = float(np.max(proba))
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=confidence * 100,
                    title={"text": "Confidence (%)"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "darkblue"},
                        "steps": [
                            {"range": [0, 50], "color": "lightcoral"},
                            {"range": [50, 80], "color": "lightyellow"},
                            {"range": [80, 100], "color": "lightgreen"},
                        ],
                    },
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)

        # OOD warning
        ood_features = []
        for col in selected.columns:
            val = float(selected[col].iloc[0])
            # Simple z-score check against training data
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

    # Try loading raw dataset
    raw_candidates = [
        ROOT / "Behaviour_Extended_Dataset.csv",
        ROOT / "Behaviour_Dataset.csv",
    ]
    raw_path = None
    for candidate in raw_candidates:
        if candidate.exists():
            raw_path = candidate
            break

    if raw_path is None:
        st.error("No behaviour dataset found.")
        return

    raw = pd.read_csv(raw_path)
    st.markdown(f"**Source:** `{raw_path.name}`  |  **Rows:** {len(raw):,}  |  **Columns:** {raw.shape[1]}")

    tab_preview, tab_stats, tab_dist = st.tabs(["Preview", "Statistics", "Distributions"])

    with tab_preview:
        st.dataframe(raw.head(100), use_container_width=True)

    with tab_stats:
        st.dataframe(raw.describe(), use_container_width=True)

    with tab_dist:
        numeric_cols = raw.select_dtypes(include=[np.number]).columns.tolist()
        selected_col = st.selectbox("Select feature", numeric_cols)
        if selected_col:
            fig = px.histogram(
                raw,
                x=selected_col,
                color="relationship_intent" if "relationship_intent" in raw.columns else None,
                barmode="overlay",
                opacity=0.7,
                title=f"Distribution of {selected_col}",
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

    # Cohort analysis
    if "relationship_intent" in raw.columns:
        st.subheader("Intent Distribution")
        intent_counts = raw["relationship_intent"].value_counts().reset_index()
        intent_counts.columns = ["Intent", "Count"]
        fig = px.bar(intent_counts, x="Intent", y="Count", color="Intent")
        st.plotly_chart(fig, use_container_width=True)


# ===========================================================================
# PAGE: Audit Log
# ===========================================================================

def page_audit_log():
    st.title("Prediction Audit Log")
    st.markdown("History of all predictions made through the API and dashboard.")

    records = load_predictions_log()

    if not records:
        st.info("No predictions logged yet. Make a prediction via the API or Scenario Predictor.")
        return

    # Parse into DataFrame
    rows = []
    for rec in records:
        payload = rec.get("payload", {})
        result = rec.get("result", {})
        rows.append({
            "timestamp": rec.get("timestamp", ""),
            "prediction": result.get("prediction", ""),
            "confidence": result.get("confidence", None),
            "app_usage": payload.get("app_usage_time_min", None),
            "swipe_ratio": payload.get("swipe_right_ratio", None),
        })

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    st.subheader(f"Total Predictions: {len(df)}")

    col1, col2 = st.columns(2)
    with col1:
        if "prediction" in df.columns:
            pred_counts = df["prediction"].value_counts().reset_index()
            pred_counts.columns = ["Prediction", "Count"]
            fig = px.pie(pred_counts, names="Prediction", values="Count", hole=0.35)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        if "confidence" in df.columns and df["confidence"].notna().any():
            fig = px.histogram(df, x="confidence", nbins=20, title="Confidence Distribution")
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Recent Predictions")
    st.dataframe(df.tail(50).iloc[::-1], use_container_width=True)


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
}

PAGES[page]()
