# =============================================================================
# PREPROCESSING - Connection Readiness Pipeline
# WID3006 ML Group Assignment: "Tying the (Data) Knot"
# =============================================================================
# Constructs a five-stage connection-readiness target from dating-app funnel
# signals, engineers features, encodes, scales, and exports train/test artifacts.
# =============================================================================

import logging
import os
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from connection_scoring import (
    TARGET_COL,
    add_connection_features,
    construct_connection_stage,
)

try:
    from logging_config import logger
except ImportError:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    logger = logging.getLogger("intentsight")

warnings.filterwarnings("ignore", category=FutureWarning)

# --- CONFIGURATION ---
ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "Preprocessed_Data_V2"
DATASET_PATH = ROOT / "Behaviour_Extended_Dataset.csv"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# STEP 1: LOAD & VALIDATE
# =============================================================================
def load_dataset() -> pd.DataFrame:
    """Load the extended behavior dataset from local CSV."""
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")
    logger.info("Loading dataset: %s", DATASET_PATH.name)
    df = pd.read_csv(DATASET_PATH)
    logger.info("  Shape: %s", df.shape)
    return df


def validate_dataset(df: pd.DataFrame) -> None:
    """Run basic data quality checks."""
    assert len(df) >= 100, f"Too few rows: {len(df)}"
    nulls = df.isnull().sum().sum()
    if nulls > 0:
        logger.warning("  Found %d null values (will impute)", nulls)
    dups = df.duplicated().sum()
    if dups > 0:
        logger.warning("  Found %d duplicate rows", dups)
    logger.info("  Validation passed: %d rows, %d columns", len(df), len(df.columns))


# =============================================================================
# STEP 2: CONSTRUCT TARGET
# =============================================================================
def construct_target(df: pd.DataFrame) -> pd.DataFrame:
    """Create five plain-language connection-readiness labels."""
    logger.info("Constructing %s target...", TARGET_COL)

    df[TARGET_COL] = construct_connection_stage(df)

    dist = df[TARGET_COL].value_counts().sort_index()
    logger.info("  Connection-stage distribution:\n%s", dist.to_dict())
    return df


# =============================================================================
# STEP 3: FEATURE ENGINEERING
# =============================================================================
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer new features and drop redundant columns."""
    logger.info("Engineering features...")

    df = add_connection_features(df)

    # Interest tags -> binary columns
    tags_series = df["interest_tags"].fillna("").str.split(r",\s*")
    all_tags = set()
    for tag_list in tags_series:
        all_tags.update(tag_list)
    all_tags.discard("")
    sorted_tags = sorted(all_tags)
    logger.info("  Found %d unique interest tags", len(sorted_tags))

    for tag in sorted_tags:
        df[f"tag_{tag}"] = tags_series.apply(lambda x: 1 if tag in x else 0)

    # Drop redundant columns
    cols_to_drop = [
        "interest_tags",
        "app_usage_time_label",
        "swipe_right_label",
        "relationship_intent",
        "match_outcome",
        "connection_score",
        "browser_issue",
        "swipe_issue",
        "engagement_score",
    ]
    dropped = [c for c in cols_to_drop if c in df.columns]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    logger.info("  Dropped: %s", dropped)
    return df


# =============================================================================
# STEP 4: ENCODE CATEGORICALS
# =============================================================================
def encode_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Encode categoricals and separate target."""
    logger.info("Encoding categorical features...")

    y = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL])

    # Ordinal encoding
    income_order = [
        "Very Low",
        "Low",
        "Lower-Middle",
        "Middle",
        "Upper-Middle",
        "High",
        "Very High",
    ]
    education_order = [
        "No Formal Education",
        "High School",
        "Associate's",
        "Bachelor's",
        "Master's",
        "Postdoc",
        "PhD",
    ]

    for col, order in [
        ("income_bracket", income_order),
        ("education_level", education_order),
    ]:
        if col in X.columns:
            mapping = {v: i for i, v in enumerate(order)}
            X[col] = X[col].map(mapping).fillna(-1).astype(int)
    logger.info("  Ordinal encoded: income_bracket, education_level")

    # One-hot encoding for remaining categoricals
    remaining_cat = X.select_dtypes(include=["object"]).columns.tolist()
    if remaining_cat:
        X = pd.get_dummies(X, columns=remaining_cat, drop_first=True)
        logger.info("  One-hot encoded: %s", remaining_cat)

    logger.info("  Final feature count: %d", X.shape[1])
    return X, y


# =============================================================================
# STEP 5: SCALE, SPLIT, SELECT
# =============================================================================
def scale_split_select(
    X: pd.DataFrame, y: pd.Series
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    np.ndarray,
    np.ndarray,
    StandardScaler,
    LabelEncoder,
    list[str],
]:
    """Scale features, split train/test, select top features."""
    logger.info("Scaling, splitting, and selecting features...")

    # Encode target
    target_encoder = LabelEncoder()
    y_encoded = target_encoder.fit_transform(y)
    logger.info("  Target classes: %s", list(target_encoder.classes_))

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    logger.info("  Train: %s | Test: %s", X_train.shape, X_test.shape)

    # Scale
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns, index=X_test.index
    )

    # Feature selection (on unresampled data — unbiased)
    logger.info("  Selecting features via RF importance...")
    selector_rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    selector_rf.fit(X_train_scaled, y_train)

    importances = selector_rf.feature_importances_
    importance_df = pd.DataFrame(
        {
            "Feature": X_train_scaled.columns,
            "Importance": importances,
        }
    ).sort_values("Importance", ascending=False)

    importance_df["Cumulative"] = importance_df["Importance"].cumsum()
    threshold = 0.95 * importance_df["Importance"].sum()
    selected_features = importance_df[importance_df["Cumulative"] <= threshold][
        "Feature"
    ].tolist()

    if len(selected_features) < 20:
        selected_features = importance_df.head(20)["Feature"].tolist()

    logger.info(
        "  Selected %d / %d features", len(selected_features), X_train_scaled.shape[1]
    )

    # Save feature importance plot
    plt.figure(figsize=(10, 8))
    sns.barplot(
        x="Importance",
        y="Feature",
        data=importance_df.head(20),
        palette="magma",
        hue="Feature",
        legend=False,
    )
    plt.title("Top 20 Feature Importances (Unbiased)", fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "feature_importances.png", dpi=150)
    plt.close()

    return (
        X_train_scaled[selected_features],
        X_test_scaled[selected_features],
        y_train,
        y_test,
        scaler,
        target_encoder,
        selected_features,
    )


# =============================================================================
# MAIN
# =============================================================================
def main() -> None:
    logger.info("=" * 60)
    logger.info("PREPROCESSING - Connection Readiness Pipeline")
    logger.info("=" * 60)

    # Step 1: Load & validate
    df = load_dataset()
    validate_dataset(df)

    # Step 2: Construct target
    df = construct_target(df)

    # Step 3: Feature engineering
    df = engineer_features(df)

    # Step 4: Encode
    X, y = encode_features(df)

    # Step 5: Scale, split, select
    X_train, X_test, y_train, y_test, scaler, target_encoder, selected_features = (
        scale_split_select(X, y)
    )

    # =============================================================================
    # STEP 6: SAVE ARTIFACTS
    # =============================================================================
    logger.info("Saving artifacts...")

    X_train.to_csv(OUTPUT_DIR / "X_train_selected_unresampled.csv", index=False)
    X_test.to_csv(OUTPUT_DIR / "X_test_selected.csv", index=False)
    pd.DataFrame(y_train, columns=[TARGET_COL]).to_csv(
        OUTPUT_DIR / "y_train_original.csv", index=False
    )
    pd.DataFrame(y_test, columns=[TARGET_COL]).to_csv(
        OUTPUT_DIR / "y_test.csv", index=False
    )

    joblib.dump(scaler, OUTPUT_DIR / "scaler.pkl")
    joblib.dump(target_encoder, OUTPUT_DIR / "target_encoder.pkl")
    joblib.dump(selected_features, OUTPUT_DIR / "selected_features.pkl")

    logger.info("  Artifacts saved to %s", OUTPUT_DIR)
    logger.info("=" * 60)
    logger.info("Preprocessing Complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
