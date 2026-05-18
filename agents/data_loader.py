from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Optional

import pandas as pd
import requests


@dataclass(frozen=True)
class GitHubCsvSources:
    chats_url: str
    phone_calls_url: str
    salesforce_clients_url: str

    def validate(self) -> None:
        missing = [
            name
            for name, value in {
                "chats_url": self.chats_url,
                "phone_calls_url": self.phone_calls_url,
                "salesforce_clients_url": self.salesforce_clients_url,
            }.items()
            if not value
        ]
        if missing:
            raise ValueError(f"Missing GitHub CSV URL(s): {', '.join(missing)}")


class GitHubCsvLoader:
    def __init__(self, sources: GitHubCsvSources, timeout_seconds: int = 30):
        sources.validate()
        self.sources = sources
        self.timeout_seconds = timeout_seconds

    def load_all(self) -> dict[str, pd.DataFrame]:
        return {
            "chats": self._read_csv(self.sources.chats_url),
            "phone_calls": self._read_csv(self.sources.phone_calls_url),
            "salesforce_clients": self._read_csv(self.sources.salesforce_clients_url),
        }

    def _read_csv(self, url: str) -> pd.DataFrame:
        response = requests.get(url, timeout=self.timeout_seconds)
        response.raise_for_status()
        return pd.read_csv(StringIO(response.text))


def save_output(df: pd.DataFrame, output_path: Optional[str]) -> None:
    if not output_path:
        return
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
