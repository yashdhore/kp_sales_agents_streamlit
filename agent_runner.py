from __future__ import annotations

from pathlib import Path

import yaml

from agents import InsightsAgent, PerformanceAgent, SalesCampaignAgent
from agents.data_loader import GitHubCsvLoader, GitHubCsvSources


def load_config(path: str = "config/agents_config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def find_agent_config(config: dict, name: str) -> dict:
    return next(agent for agent in config["agents"] if agent["name"] == name)


def run_all(config_path: str = "config/agents_config.yaml", sources: GitHubCsvSources | None = None) -> dict:
    config = load_config(config_path)
    source_config = config["data_sources"]
    sources = sources or GitHubCsvSources(
        chats_url=source_config.get("chats_url", ""),
        phone_calls_url=source_config.get("phone_calls_url", ""),
        salesforce_clients_url=source_config.get("salesforce_clients_url", ""),
    )

    frames = GitHubCsvLoader(sources).load_all()
    insights = InsightsAgent(find_agent_config(config, "InsightsAgent")).run(
        frames["chats"],
        frames["phone_calls"],
        frames["salesforce_clients"],
    )
    performance = PerformanceAgent(find_agent_config(config, "PerformanceAgent")).run(insights)
    campaign = SalesCampaignAgent(find_agent_config(config, "SalesCampaignAgent")).run(insights)
    return {"insights": insights, "performance": performance, "campaign": campaign}


if __name__ == "__main__":
    Path("output").mkdir(exist_ok=True)
    result = run_all()
    for name, df in result.items():
        print(f"{name}: {len(df)} rows")
