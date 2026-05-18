from __future__ import annotations

from datetime import datetime, timezone
import uuid

import pandas as pd

from .data_loader import save_output


class PerformanceAgent:
    def __init__(self, config: dict):
        self.config = config
        self.output_path = config.get("output", {}).get("save_to")

    def run(self, insights: pd.DataFrame) -> pd.DataFrame:
        filtered = self._apply_filters(insights.copy())
        scored = self._score_and_categorize(filtered)
        save_output(scored, self.output_path)
        return scored

    def _apply_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        filters = self.config.get("input", {}).get("filters", {})
        for column, allowed_values in filters.items():
            if column in df.columns and allowed_values:
                df = df[df[column].isin(allowed_values)]
        return df

    def _score_and_categorize(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(columns=["AgentName", "ResolutionRate", "AvgSentimentScore", "Interactions", "Category", "id", "Timestamp"])

        df["ResolvedFlag"] = df["Resolved"].apply(lambda value: 1 if str(value).strip().lower() == "yes" else 0)
        df["SentimentScore"] = pd.to_numeric(df.get("SentimentScore", 0), errors="coerce").fillna(0)
        df["AgentName"] = df.get("AgentName", "Unknown").fillna("Unknown")

        grouped = (
            df.groupby("AgentName", dropna=False)
            .agg(
                ResolutionRate=("ResolvedFlag", "mean"),
                AvgSentimentScore=("SentimentScore", "mean"),
                Interactions=("ClientID", "count"),
            )
            .reset_index()
        )

        grouped["Category"] = grouped.apply(self._classify, axis=1)
        grouped["id"] = [str(uuid.uuid4()) for _ in range(len(grouped))]
        grouped["Timestamp"] = datetime.now(timezone.utc).isoformat()
        return grouped.sort_values(["Category", "ResolutionRate"], ascending=[True, False])

    def _classify(self, row: pd.Series) -> str:
        thresholds = self.config.get("categorization", {})
        if row["ResolutionRate"] >= thresholds["Best"]["resolution_threshold"] and row["AvgSentimentScore"] >= thresholds["Best"]["sentiment_threshold"]:
            return "Best"
        if row["ResolutionRate"] >= thresholds["Better"]["resolution_threshold"] and row["AvgSentimentScore"] >= thresholds["Better"]["sentiment_threshold"]:
            return "Better"
        if row["ResolutionRate"] >= thresholds["Good"]["resolution_threshold"]:
            return "Good"
        return "Train"
