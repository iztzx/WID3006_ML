"""Connection-readiness feature engineering and weak-label construction."""

from __future__ import annotations

import numpy as np
import pandas as pd


TARGET_COL = "connection_stage"

STAGE_LABELS = [
    "Needs Profile Help",
    "Mostly Browsing",
    "Swipes Too Freely",
    "Ready To Chat",
    "Likely To Connect",
]

STRONG_STAGES = {"Ready To Chat", "Likely To Connect"}


def _numeric(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce").fillna(default)


def _rank01(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").fillna(0)
    if values.nunique(dropna=False) <= 1:
        return pd.Series(0.5, index=values.index, dtype="float64")
    return values.rank(pct=True).clip(0, 1)


def _bounded(series: pd.Series, cap: float) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").fillna(0).clip(lower=0)
    return (values / cap).clip(0, 1)


def _num_interests(frame: pd.DataFrame) -> pd.Series:
    if "interest_tags" in frame.columns:
        tags = frame["interest_tags"].fillna("").astype(str).str.split(r",\s*")
        return tags.apply(lambda values: len([tag for tag in values if tag]))
    if "num_interests" in frame.columns:
        return _numeric(frame, "num_interests")
    return pd.Series(0, index=frame.index, dtype="float64")


def add_connection_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add interpretable product features used by the connection-readiness model."""
    engineered = frame.copy()

    app_usage = _numeric(engineered, "app_usage_time_min")
    swipe_ratio = _numeric(engineered, "swipe_right_ratio")
    likes = _numeric(engineered, "likes_received")
    matches = _numeric(engineered, "mutual_matches")
    messages = _numeric(engineered, "message_sent_count")
    emoji_rate = _numeric(engineered, "emoji_usage_rate")
    bio_length = _numeric(engineered, "bio_length")
    profile_pics = _numeric(engineered, "profile_pics_count")
    height = _numeric(engineered, "height_cm")
    weight = _numeric(engineered, "weight_kg")

    engineered["num_interests"] = _num_interests(engineered)
    engineered["match_rate"] = matches / (likes + 1)
    engineered["msg_per_match"] = messages / (matches + 1)
    if "height_cm" in frame.columns and "weight_kg" in frame.columns:
        engineered["bmi"] = np.where(height > 0, weight / ((height / 100) ** 2), 0)
    engineered["profile_completeness"] = (
        profile_pics.clip(0, 6) / 6 * 0.40
        + bio_length.clip(0, 300) / 300 * 0.40
        + engineered["num_interests"].clip(0, 5) / 5 * 0.20
    )
    engineered["selectivity_balance"] = (1 - (swipe_ratio - 0.55).abs() / 0.55).clip(
        0, 1
    )
    engineered["swipe_excess"] = (swipe_ratio - 0.70).clip(lower=0)
    engineered["like_to_match_gap"] = (likes - matches).clip(lower=0)
    engineered["conversation_depth"] = np.log1p(messages) * np.log1p(
        engineered["msg_per_match"]
    )
    engineered["social_pull"] = likes / (profile_pics + 1)
    engineered["activity_level"] = np.log1p(app_usage)

    if "last_active_hour" in engineered.columns:
        hour = _numeric(engineered, "last_active_hour") % 24
        engineered["last_active_sin"] = np.sin(2 * np.pi * hour / 24)
        engineered["last_active_cos"] = np.cos(2 * np.pi * hour / 24)

    engineered["match_quality"] = (
        0.45 * engineered["match_rate"].clip(0, 1)
        + 0.25 * _bounded(matches, 40)
        + 0.15 * engineered["selectivity_balance"]
        + 0.15 * _bounded(engineered["social_pull"], 50)
    )
    engineered["conversation_quality"] = (
        0.40 * _bounded(engineered["msg_per_match"], 10)
        + 0.30 * _bounded(messages, 80)
        + 0.20 * _bounded(emoji_rate, 1)
        + 0.10 * _bounded(app_usage, 300)
    )
    engineered["profile_quality"] = (
        0.60 * engineered["profile_completeness"]
        + 0.25 * _bounded(bio_length, 300)
        + 0.15 * _bounded(profile_pics, 6)
    )
    engineered["connection_score"] = (
        0.35 * engineered["match_quality"]
        + 0.30 * engineered["conversation_quality"]
        + 0.20 * engineered["profile_quality"]
        + 0.15 * _bounded(app_usage, 300)
        - 0.10 * _bounded(engineered["swipe_excess"], 0.30)
    )
    engineered["browser_issue"] = (
        0.45 * (1 - _bounded(app_usage, 300))
        + 0.35 * (1 - _bounded(messages, 80))
        + 0.20 * (1 - _bounded(matches, 40))
    )
    engineered["swipe_issue"] = 0.55 * _bounded(
        engineered["swipe_excess"], 0.30
    ) + 0.45 * (1 - engineered["match_rate"].clip(0, 1))

    return engineered


def construct_connection_stage(frame: pd.DataFrame) -> pd.Series:
    """Create plain-language connection-readiness labels from funnel signals."""
    scored = add_connection_features(frame)

    score_rank = _rank01(scored["connection_score"])
    browser_issue = scored["browser_issue"]
    swipe_issue = scored["swipe_issue"]

    labels = pd.Series("Ready To Chat", index=scored.index, dtype="object")
    labels[score_rank >= 0.80] = "Likely To Connect"
    labels[score_rank <= 0.20] = "Needs Profile Help"

    middle = labels.eq("Ready To Chat")
    labels[
        middle
        & (browser_issue >= browser_issue.quantile(0.62))
        & (browser_issue >= swipe_issue)
    ] = "Mostly Browsing"

    middle = labels.eq("Ready To Chat")
    labels[middle & (swipe_issue >= swipe_issue.quantile(0.50))] = "Swipes Too Freely"

    ordered = pd.Categorical(labels, categories=STAGE_LABELS, ordered=True)
    return pd.Series(ordered, index=labels.index).astype(str)
