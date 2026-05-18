from __future__ import annotations

import json
import os
from functools import lru_cache

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


class InteractionLlmClient:
    def __init__(self) -> None:
        if load_dotenv:
            load_dotenv()
        self.enabled = os.getenv("LLM_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.api_key = os.getenv("OPENAI_API_KEY")
        self._client = None

        if self.enabled and self.api_key:
            from openai import OpenAI

            self._client = OpenAI(api_key=self.api_key)

    @property
    def available(self) -> bool:
        return self.enabled and self._client is not None

    @lru_cache(maxsize=1024)
    def analyze(self, transcript: str, known_intent: str, resolved: str) -> dict:
        if not self.available:
            return self._fallback(transcript, known_intent, resolved)

        prompt = {
            "transcript": transcript or "",
            "known_intent": known_intent or "Unknown",
            "known_resolved_status": resolved or "No",
            "task": (
                "Analyze this customer sales or support interaction. Return compact JSON with keys: "
                "Sentiment, SentimentScore, Intent, Resolved, RiskLevel, ExecutiveInsight, NextBestAction. "
                "Sentiment must be Positive, Neutral, or Negative. SentimentScore must be from -1.0 to 1.0. "
                "Resolved must be Yes or No. RiskLevel must be Low, Medium, or High."
            ),
        }

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You produce concise, valid JSON for executive sales analytics dashboards.",
                    },
                    {"role": "user", "content": json.dumps(prompt)},
                ],
                response_format={"type": "json_object"},
                temperature=0.2,
            )
            content = response.choices[0].message.content or "{}"
            parsed = json.loads(content)
            return self._normalize(parsed, known_intent, resolved)
        except Exception:
            return self._fallback(transcript, known_intent, resolved)

    def _normalize(self, payload: dict, known_intent: str, resolved: str) -> dict:
        sentiment = payload.get("Sentiment", "Neutral")
        if sentiment not in {"Positive", "Neutral", "Negative"}:
            sentiment = "Neutral"

        try:
            score = float(payload.get("SentimentScore", 0.0))
        except (TypeError, ValueError):
            score = 0.0
        score = max(-1.0, min(1.0, score))

        resolved_value = str(payload.get("Resolved", resolved or "No")).strip().title()
        if resolved_value not in {"Yes", "No"}:
            resolved_value = "No"

        risk = payload.get("RiskLevel", "Medium")
        if risk not in {"Low", "Medium", "High"}:
            risk = "Medium"

        return {
            "Sentiment": sentiment,
            "SentimentScore": score,
            "Intent": payload.get("Intent") or known_intent or "Unknown",
            "Resolved": resolved_value,
            "RiskLevel": risk,
            "ExecutiveInsight": payload.get("ExecutiveInsight") or "Customer interaction requires follow-up review.",
            "NextBestAction": payload.get("NextBestAction") or "Review account and follow up through the preferred channel.",
        }

    def _fallback(self, transcript: str, known_intent: str, resolved: str) -> dict:
        text = (transcript or "").lower()
        negative_terms = ["cancel", "refund", "slow", "not happy", "bad", "issue", "problem"]
        positive_terms = ["love", "thanks", "satisfied", "fine", "quick help"]

        if any(term in text for term in negative_terms):
            sentiment = "Negative"
            score = -0.6
            risk = "High"
        elif any(term in text for term in positive_terms):
            sentiment = "Positive"
            score = 0.6
            risk = "Low"
        else:
            sentiment = "Neutral"
            score = 0.0
            risk = "Medium"

        resolved_value = str(resolved or "No").strip().title()
        if resolved_value not in {"Yes", "No"}:
            resolved_value = "No"
        if resolved_value == "No" and risk == "Low":
            risk = "Medium"

        return {
            "Sentiment": sentiment,
            "SentimentScore": score,
            "Intent": known_intent or "Unknown",
            "Resolved": resolved_value,
            "RiskLevel": risk,
            "ExecutiveInsight": "Interaction classified using local fallback logic because the LLM is not configured.",
            "NextBestAction": "Follow up with the customer and validate the recommended outreach path.",
        }
