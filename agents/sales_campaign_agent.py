from __future__ import annotations

from datetime import datetime, timezone
import uuid

import pandas as pd

from .data_loader import save_output


class SalesCampaignAgent:
    def __init__(self, config: dict):
        self.config = config
        self.output_path = config.get("output", {}).get("save_to")

    def run(self, insights: pd.DataFrame) -> pd.DataFrame:
        filtered = self._apply_filters(insights.copy())
        recommendations = self._recommend(filtered)
        save_output(recommendations, self.output_path)
        return recommendations

    def _apply_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        filters = self.config.get("input", {}).get("filters", {})
        for column, allowed_values in filters.items():
            if column in df.columns and allowed_values:
                df = df[df[column].isin(allowed_values)]
        return df

    def _recommend(self, df: pd.DataFrame) -> pd.DataFrame:
        weights = self.config["recommendation_strategy"]["score_weights"]
        priority = self.config["recommendation_strategy"]["priority_order"]
        min_score = self.config["recommendation_strategy"]["minimum_score_threshold"]

        records = []
        for _, row in df.iterrows():
            scores = self._score_channels(row, weights)
            sorted_channels = sorted(scores.items(), key=lambda item: (-item[1], priority.index(item[0])))
            recommended_channel, score = sorted_channels[0]
            if score < min_score or str(row.get("Resolved", "No")).strip().lower() == "yes":
                recommended_channel = "None"
                score = 0.0

            records.append(
                {
                    "id": str(uuid.uuid4()),
                    "Timestamp": datetime.now(timezone.utc).isoformat(),
                    "ClientID": row.get("ClientID"),
                    "RecommendedChannel": recommended_channel,
                    "RecommendationScore": round(score, 3),
                    "PhoneScore": round(scores["Phone"], 3),
                    "EmailScore": round(scores["Email"], 3),
                    "SMSScore": round(scores["SMS"], 3),
                    "Sentiment": row.get("Sentiment"),
                    "Intent": row.get("Intent"),
                    "Resolved": row.get("Resolved"),
                    "SalesChannel": row.get("SalesChannel"),
                    "Region": row.get("Region"),
                }
            )

        return pd.DataFrame(records)

    def _score_channels(self, row: pd.Series, weights: dict[str, float]) -> dict[str, float]:
        unresolved_boost = 0.2 if str(row.get("Resolved", "No")).strip().lower() != "yes" else 0.0
        negative_boost = 0.15 if row.get("Sentiment") == "Negative" else 0.0
        phone_boost = 0.1 if row.get("SalesChannel") == "phone" else 0.0
        chat_boost = 0.1 if row.get("SalesChannel") == "chat" else 0.0

        return {
            "Phone": weights["Phone"] + unresolved_boost + negative_boost + phone_boost,
            "Email": weights["Email"] + unresolved_boost,
            "SMS": weights["SMS"] + unresolved_boost + chat_boost,
        }
