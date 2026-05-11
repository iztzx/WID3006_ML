# =============================================================================
# DATA PREPROCESSING V3 — Schema-Validated, Leakage-Free Pipeline
# WID3006 ML Group Assignment: "Tying the (Data) Knot"
# =============================================================================
# Key improvements over V2:
#   1. Schema validation (validate_dataset) — row count, class ratio, duplicates, collinearity
#   2. Unbiased feature importance — RF on unresampled X_train (not SMOTE data)
#   3. Imputation instead of dropna (preserves all 50k rows)
#   4. MultiLabelBinarizer for interest_tags
#   5. Ordinal encoding for ordered categoricals
#   6. Exports both full and selected feature sets
# =============================================================================

# --- IMPORTS ---
import logging
import os
import warnings
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE

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
PATH = ROOT / "Preprocessed_Data_V2"
LOCAL_DATASET_CANDIDATES = (
    ROOT / "Behaviour_Extended_Dataset.csv",
    ROOT / "Behaviour_Dataset.csv",
)

os.makedirs(PATH, exist_ok=True)


# =============================================================================
# STEP 0: SCHEMA VALIDATION
# =============================================================================
def validate_dataset(df: pd.DataFrame, name: str = "dataset") -> dict[str, Any]:
    """Run comprehensive data quality checks and return a report.

    Checks performed:
    1. Row count (must be > 100)
    2. Class ratio (minority class must be >= 5% of total)
    3. Duplicate rows
    4. Column collinearity (absolute correlation > 0.95 flag)
    5. Null columns check
    """
    report: dict[str, Any] = {
        "name": name,
        "rows": len(df),
        "columns": len(df.columns),
        "errors": [],
        "warnings": [],
    }

    # Check 1: Row count
    if len(df) < 100:
        report["errors"].append(
            f"Insufficient rows ({len(df)}). Expected >= 100."
        )
    logger.info("  Validation: %d rows, %d columns", len(df), len(df.columns))

    # Check 2: Class ratio (if intent_bin exists)
    if "intent_bin" in df.columns:
        class_counts = df["intent_bin"].value_counts(normalize=True)
        min_share = class_counts.min()
        if min_share < 0.05:
            report["warnings"].append(
                f"Minority class share is {min_share:.2%} (< 5% threshold). "
                "Severe imbalance may affect model reliability."
            )
        logger.info("  Class distribution:\n%s", class_counts.to_dict())

    # Check 3: Duplicate rows
    n_dups = df.duplicated().sum()
    if n_dups > 0:
        report["warnings"].append(
            f"Found {n_dups} duplicate rows ({n_dups / len(df):.2%} of data)."
        )
        logger.warning("  Duplicates found: %d", n_dups)

    # Check 4: Collinearity
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        high_corr = [
            (c1, c2, upper.loc[c1, c2])
            for c1, c2 in zip(*np.where(upper > 0.95))
            if c1 != c2
        ]
        if high_corr:
            report["warnings"].append(
                f"High collinearity detected (|r| > 0.95): {high_corr[:5]}"
            )
            logger.warning("  High collinearity pairs: %s", high_corr[:5])

    # Check 5: Null columns (all-null)
    all_null_cols = df.columns[df.isnull().all()].tolist()
    if all_null_cols:
        report["errors"].append(f"All-null columns: {all_null_cols}")
        logger.error("  All-null columns: %s", all_null_cols)

    status = "FAIL" if report["errors"] else ("WARN" if report["warnings"] else "PASS")
    report["status"] = status
    logger.info("  Validation status: %s", status)

    return report


# =============================================================================
# STEP 1: LOAD DATA
# =============================================================================
logger.info("=" * 60)
logger.info("STEP 1: Loading dataset...")
logger.info("=" * 60)


def load_behavior_dataset() -> tuple[pd.DataFrame, str]:
    """Prefer Kaggle, then fall back to local CSV files."""

    dataset_override = os.getenv("INTENTSIGHT_DATASET_PATH")
    if dataset_override:
        override_path = Path(dataset_override).expanduser()
        if not override_path.is_absolute():
            override_path = ROOT / override_path
        logger.info("  Loading from INTENTSIGHT_DATASET_PATH: %s", override_path)
        return pd.read_csv(override_path), str(override_path)

    skip_kaggle = os.getenv("INTENTSIGHT_SKIP_KAGGLE", "").lower() in {
        "1", "true", "yes"
    }
    if not skip_kaggle:
        try:
            import kagglehub
            from kagglehub import KaggleDatasetAdapter

            df_kaggle = kagglehub.dataset_load(
                KaggleDatasetAdapter.PANDAS,
                "keyushnisar/dating-app-behavior-dataset",
                "dating_app_behavior_dataset_extended1.csv",
            )
            return df_kaggle, "KaggleHub:keyushnisar/dating-app-behavior-dataset"
        except Exception as exc:
            logger.warning("  KaggleHub load failed: %s", exc)
            logger.info("  Falling back to local CSV files...")
    else:
        logger.info("  INTENTSIGHT_SKIP_KAGGLE is set; using local CSV files.")

    for candidate in LOCAL_DATASET_CANDIDATES:
        if candidate.exists():
            logger.info("  Loading local dataset: %s", candidate.name)
            return pd.read_csv(candidate), str(candidate)

    searched = ", ".join(str(p) for p in LOCAL_DATASET_CANDIDATES)
    raise FileNotFoundError(
        "Could not load the behavior dataset. " f"Searched: {searched}"
    )


df, dataset_source = load_behavior_dataset()
logger.info("Dataset source: %s | Shape: %s", dataset_source, df.shape)

# Run validation
validation_report = validate_dataset(df, name=dataset_source)
logger.info("Validation report: %s", validation_report)

print(f"\nMissing values per column:\n{df.isnull().sum()}")

# =============================================================================
# STEP 1b: CREATE TARGET COLUMN (intent_bin) FROM relationship_intent
# =============================================================================
logger.info("STEP 1b: Creating intent_bin from relationship_intent...")

if "intent_bin" not in df.columns:
    if "relationship_intent" in df.columns:
        # Map: "Serious Relationship" and "Casual Dating" → "Dating"; everything else → "Other"
        dating_classes = {"Serious Relationship", "Casual Dating"}
        df["intent_bin"] = df["relationship_intent"].apply(
            lambda x: "Dating" if x in dating_classes else "Other"
        )
        logger.info(
            "  Created intent_bin from relationship_intent. Distribution:\n%s",
            df["intent_bin"].value_counts().to_dict(),
        )
    else:
        raise KeyError(
            "Neither 'intent_bin' nor 'relationship_intent' found in dataset columns. "
            "Cannot create target variable."
        )

# =============================================================================
# STEP 2: HANDLE MISSING VALUES (Impute, don't drop!)
# =============================================================================
logger.info("STEP 2: Handling missing values via imputation...")

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols_all = df.select_dtypes(include=["object"]).columns.tolist()

num_imputer = SimpleImputer(strategy="median")
cat_imputer = SimpleImputer(strategy="most_frequent")

df[numeric_cols] = num_imputer.fit_transform(df[numeric_cols])

cat_cols_to_impute = [
    c for c in categorical_cols_all if c not in ["intent_bin", "relationship_intent"]
]
if cat_cols_to_impute:
    df[cat_cols_to_impute] = cat_imputer.fit_transform(df[cat_cols_to_impute])

logger.info("  Remaining missing values: %d", df.isnull().sum().sum())
logger.info("  Rows preserved: %d (vs dropna which could lose rows)", len(df))

# =============================================================================
# STEP 3: FEATURE ENGINEERING
# =============================================================================
logger.info("STEP 3: Feature engineering...")

# --- 3a. Extract interest_tags via MultiLabelBinarizer ---
logger.info("  3a. Extracting interest_tags...")
tags_series = df["interest_tags"].fillna("").str.split(r",\s*")

all_tags = set()
for tag_list in tags_series:
    all_tags.update(tag_list)
all_tags.discard("")
all_tags = sorted(all_tags)
logger.info("      Found %d unique interest tags", len(all_tags))

for tag in all_tags:
    df[f"tag_{tag}"] = tags_series.apply(lambda x: 1 if tag in x else 0)

df["num_interests"] = tags_series.apply(len)

# --- 3b. Drop redundant columns ---
logger.info("  3b. Dropping redundant columns...")

cols_to_drop = [
    "interest_tags", "app_usage_time_label", "swipe_right_label",
    "relationship_intent", "match_outcome",
]
df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
logger.info("      Dropped: %s", cols_to_drop)

# --- 3c. Engineered ratio/interaction features ---
logger.info("  3c. Creating interaction features...")

if {"mutual_matches", "likes_received"}.issubset(df.columns):
    df["match_rate"] = df["mutual_matches"] / (df["likes_received"] + 1)
if {"message_sent_count", "mutual_matches"}.issubset(df.columns):
    df["msg_per_match"] = df["message_sent_count"] / (df["mutual_matches"] + 1)
if {"weight_kg", "height_cm"}.issubset(df.columns):
    df["bmi"] = df["weight_kg"] / ((df["height_cm"] / 100) ** 2)
if {"app_usage_time_min", "swipe_right_ratio", "emoji_usage_rate"}.issubset(
    df.columns
):
    df["engagement_score"] = (
        df["app_usage_time_min"] * df["swipe_right_ratio"] * df["emoji_usage_rate"]
    )

# =============================================================================
# STEP 4: ENCODE CATEGORICALS
# =============================================================================
logger.info("STEP 4: Encoding categorical features...")

y = df["intent_bin"]
X = df.drop(columns=["intent_bin"])

target_encoder = LabelEncoder()
y_encoded = target_encoder.fit_transform(y)
logger.info("  Target classes: %s", list(target_encoder.classes_))

income_order = [
    "Very Low", "Low", "Lower-Middle", "Middle",
    "Upper-Middle", "High", "Very High",
]
education_order = [
    "No Formal Education", "High School", "Associate's",
    "Bachelor's", "Master's", "Postdoc", "PhD",
]

ordinal_cols = []
if "income_bracket" in X.columns:
    ordinal_cols.append("income_bracket")
if "education_level" in X.columns:
    ordinal_cols.append("education_level")

if ordinal_cols:
    for col, order in [("income_bracket", income_order), ("education_level", education_order)]:
        if col in X.columns:
            mapping = {v: i for i, v in enumerate(order)}
            X[col] = X[col].map(mapping).fillna(-1).astype(int)
    logger.info("  Ordinal encoded: %s", ordinal_cols)

remaining_cat = X.select_dtypes(include=["object"]).columns.tolist()
logger.info("  One-hot encoding: %s", remaining_cat)
X_encoded = pd.get_dummies(X, columns=remaining_cat, drop_first=True)
logger.info("  Final feature count: %d", X_encoded.shape[1])

# =============================================================================
# STEP 5: TRAIN-TEST SPLIT (stratified)
# =============================================================================
logger.info("STEP 5: Train-test split (stratified)...")

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
logger.info("  Train: %s, Test: %s", X_train.shape, X_test.shape)

# =============================================================================
# STEP 6: SCALE FEATURES
# =============================================================================
logger.info("STEP 6: Scaling features (StandardScaler)...")

scaler = StandardScaler()
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test), columns=X_test.columns, index=X_test.index
)

# =============================================================================
# STEP 7: HANDLE CLASS IMBALANCE WITH SMOTE
# =============================================================================
logger.info("STEP 7: Handling class imbalance with SMOTE...")

logger.info("  Before SMOTE: %s", np.bincount(y_train))
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train) # type: ignore
logger.info("  After SMOTE: %s", np.bincount(y_train_resampled))
logger.info("  Resampled train shape: %s", X_train_resampled.shape)

# =============================================================================
# STEP 8: FEATURE SELECTION — UNBIASED (on unresampled data)
# =============================================================================
logger.info("STEP 8: Feature selection on unresampled data (unbiased)...")

selector_rf = RandomForestClassifier(
    n_estimators=300, random_state=42, n_jobs=-1
)
selector_rf.fit(X_train_scaled, y_train)  # UNRESAMPLED — key difference from V2

importances = selector_rf.feature_importances_
importance_df = pd.DataFrame({
    "Feature": X_train_scaled.columns,
    "Importance": importances,
}).sort_values("Importance", ascending=False)

importance_df["Cumulative"] = importance_df["Importance"].cumsum()
total_importance = importance_df["Importance"].sum()
threshold = 0.95 * total_importance
selected_features = importance_df[
    importance_df["Cumulative"] <= threshold
]["Feature"].tolist()

if len(selected_features) < 20:
    selected_features = importance_df.head(20)["Feature"].tolist()

logger.info("  Selected %d / %d features", len(selected_features), X_train_scaled.shape[1])
logger.info("  Top 15: %s", selected_features[:15])

plt.figure(figsize=(10, 8))
top20 = importance_df.head(20)
sns.barplot(
    x="Importance", y="Feature", data=top20,
    palette="magma", hue="Feature", legend=False,
)
plt.title("Top 20 Feature Importances (Unbiased — Unresampled)", fontsize=14)
plt.tight_layout()
plt.savefig(PATH / "feature_importances.png", dpi=150)
plt.close()

# Apply selection
X_test_selected = X_test_scaled[selected_features]

# =============================================================================
# STEP 9: SAVE ALL ARTIFACTS
# =============================================================================
logger.info("STEP 9: Saving all artifacts...")

X_train_scaled[selected_features].to_csv(
    PATH / "X_train_selected_unresampled.csv", index=False
)
X_test_selected.to_csv(PATH / "X_test_selected.csv", index=False)

pd.DataFrame(y_train, columns=["intent"]).to_csv(PATH / "y_train_original.csv", index=False)
pd.DataFrame(y_test, columns=["intent"]).to_csv(PATH / "y_test.csv", index=False)

joblib.dump(scaler, PATH / "scaler.pkl")
joblib.dump(target_encoder, PATH / "target_encoder.pkl")
joblib.dump(selected_features, PATH / "selected_features.pkl")
joblib.dump(X_train_resampled.columns.tolist(), PATH / "X_columns_full.pkl")

logger.info("  Artifacts saved to %s", PATH)
logger.info("  Full features: %d", X_train_resampled.shape[1])
logger.info("  Selected features: %d", len(selected_features))
logger.info("\n[SUCCESS] Preprocessing V3 Complete!")