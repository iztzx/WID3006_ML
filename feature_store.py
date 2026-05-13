"""Feature registry — central definitions for all engineered features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True)
class FeatureDefinition:
    """Metadata for a single feature."""

    name: str
    description: str
    dtype: str
    source_column: str | None = None
    transformation: str = "identity"
    is_engineered: bool = False


class FeatureStore:
    """Registry of all features used in the IntentSight pipeline."""

    # --- Target ---
    TARGET: ClassVar[str] = "engagement_level"

    # --- Raw columns from the behaviour dataset ---
    RAW_NUMERIC: ClassVar[list[str]] = [
        "age",
        "app_usage_time_min",
        "likes_received",
        "mutual_matches",
        "message_sent_count",
        "bio_length",
        "emoji_usage_rate",
        "height_cm",
        "weight_kg",
        "profile_pics_count",
        "last_active_hour",
    ]

    RAW_CATEGORICAL: ClassVar[list[str]] = [
        "gender",
        "income_bracket",
        "education_level",
        "sexual_orientation",
        "location_type",
        "swipe_time_of_day",
        "body_type",
        "interest_tags",
    ]

    # --- Behavioral features used to construct engagement_level target ---
    BEHAVIORAL_FEATURES: ClassVar[list[str]] = [
        "app_usage_time_min",
        "swipe_right_ratio",
        "message_sent_count",
        "likes_received",
        "emoji_usage_rate",
    ]

    def __init__(self) -> None:
        self._features: dict[str, FeatureDefinition] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        """Register all known raw and engineered features."""
        for col in self.RAW_NUMERIC:
            self.register(
                FeatureDefinition(
                    name=col,
                    description=f"Raw numeric column: {col}",
                    dtype="float64",
                    source_column=col,
                )
            )

        for col in self.RAW_CATEGORICAL:
            self.register(
                FeatureDefinition(
                    name=col,
                    description=f"Raw categorical column: {col}",
                    dtype="object",
                    source_column=col,
                )
            )

        # Target
        self.register(
            FeatureDefinition(
                name="engagement_level",
                description="3-class engagement level (Low/Medium/High) from composite behavioral score",
                dtype="int64",
                source_column="behavioral_composite",
                transformation="zscore_sum_quantile",
                is_engineered=True,
            )
        )

        # Engineered features
        engineered = [
            FeatureDefinition(
                name="engagement_score",
                description="Standardized sum of 5 behavioral features (before binning)",
                dtype="float64",
                source_column="behavioral",
                transformation="zscore_sum",
                is_engineered=True,
            ),
            FeatureDefinition(
                name="match_rate",
                description="mutual_matches / (likes_received + 1)",
                dtype="float64",
                source_column="likes_received",
                transformation="ratio",
                is_engineered=True,
            ),
            FeatureDefinition(
                name="msg_per_match",
                description="message_sent_count / (mutual_matches + 1)",
                dtype="float64",
                source_column="message_sent_count",
                transformation="ratio",
                is_engineered=True,
            ),
            FeatureDefinition(
                name="bmi",
                description="weight_kg / (height_cm / 100) ** 2",
                dtype="float64",
                source_column="weight_kg",
                transformation="formula",
                is_engineered=True,
            ),
            FeatureDefinition(
                name="num_interests",
                description="Count of parsed interest tags",
                dtype="int64",
                source_column="interest_tags",
                transformation="count",
                is_engineered=True,
            ),
        ]

        for feat in engineered:
            self.register(feat)

    def register(self, definition: FeatureDefinition) -> None:
        self._features[definition.name] = definition

    def get(self, name: str) -> FeatureDefinition | None:
        return self._features.get(name)

    @property
    def all_names(self) -> list[str]:
        return list(self._features.keys())

    @property
    def engineered_names(self) -> list[str]:
        return [name for name, feat in self._features.items() if feat.is_engineered]

    def to_dict(self) -> list[dict]:
        return [
            {
                "name": f.name,
                "description": f.description,
                "dtype": f.dtype,
                "source_column": f.source_column,
                "transformation": f.transformation,
                "is_engineered": f.is_engineered,
            }
            for f in self._features.values()
        ]
