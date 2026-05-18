from __future__ import annotations

import pandas as pd
import streamlit as st

from agent_runner import find_agent_config, load_config
from agents import InsightsAgent, PerformanceAgent, SalesCampaignAgent
from agents.data_loader import GitHubCsvLoader, GitHubCsvSources


st.set_page_config(page_title="KP Sales Agents", page_icon="KP", layout="wide")


@st.cache_data(show_spinner=False)
def load_github_data(chats_url: str, phone_calls_url: str, salesforce_clients_url: str) -> dict[str, pd.DataFrame]:
    sources = GitHubCsvSources(
        chats_url=chats_url.strip(),
        phone_calls_url=phone_calls_url.strip(),
        salesforce_clients_url=salesforce_clients_url.strip(),
    )
    return GitHubCsvLoader(sources).load_all()


@st.cache_data(show_spinner=False)
def run_agents(
    chats_csv: str,
    calls_csv: str,
    clients_csv: str,
    config: dict,
) -> dict[str, pd.DataFrame]:
    chats = pd.read_json(chats_csv, orient="split")
    calls = pd.read_json(calls_csv, orient="split")
    clients = pd.read_json(clients_csv, orient="split")

    insights = InsightsAgent(find_agent_config(config, "InsightsAgent")).run(chats, calls, clients)
    performance = PerformanceAgent(find_agent_config(config, "PerformanceAgent")).run(insights)
    campaign = SalesCampaignAgent(find_agent_config(config, "SalesCampaignAgent")).run(insights)
    return {"insights": insights, "performance": performance, "campaign": campaign}


def dataframe_to_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


config = load_config()
source_defaults = config.get("data_sources", {})

st.title("KP Sales Agents")

with st.sidebar:
    st.header("GitHub CSV Sources")
    chats_url = st.text_input("Chats CSV raw URL", value=source_defaults.get("chats_url", ""))
    calls_url = st.text_input("Phone calls CSV raw URL", value=source_defaults.get("phone_calls_url", ""))
    clients_url = st.text_input("Salesforce clients CSV raw URL", value=source_defaults.get("salesforce_clients_url", ""))
    run_clicked = st.button("Run agents", type="primary", use_container_width=True)

if not run_clicked:
    st.info("Enter raw GitHub CSV URLs in the sidebar, then run the agents.")
    st.stop()

try:
    with st.spinner("Reading CSV files from GitHub..."):
        frames = load_github_data(chats_url, calls_url, clients_url)

    with st.spinner("Running insights, performance, and campaign agents..."):
        result = run_agents(
            frames["chats"].to_json(orient="split"),
            frames["phone_calls"].to_json(orient="split"),
            frames["salesforce_clients"].to_json(orient="split"),
            config,
        )
except Exception as exc:
    st.error(f"Unable to run agents: {exc}")
    st.stop()

insights_df = result["insights"]
performance_df = result["performance"]
campaign_df = result["campaign"]

metric_cols = st.columns(4)
metric_cols[0].metric("Insights", f"{len(insights_df):,}")
metric_cols[1].metric("Agents", f"{performance_df['AgentName'].nunique() if not performance_df.empty else 0:,}")
metric_cols[2].metric("Campaign Targets", f"{len(campaign_df):,}")
metric_cols[3].metric("Unresolved", f"{(insights_df['Resolved'].astype(str).str.lower() != 'yes').sum():,}")

tab_insights, tab_performance, tab_campaign = st.tabs(["Insights", "Performance", "Sales Campaign"])

with tab_insights:
    st.subheader("Interaction Insights")
    chart_cols = st.columns(2)
    if "Sentiment" in insights_df.columns:
        chart_cols[0].bar_chart(insights_df["Sentiment"].value_counts())
    if "Intent" in insights_df.columns:
        chart_cols[1].bar_chart(insights_df["Intent"].value_counts())
    st.dataframe(insights_df, use_container_width=True, hide_index=True)
    st.download_button("Download insights CSV", dataframe_to_csv(insights_df), "output_insights.csv", "text/csv")

with tab_performance:
    st.subheader("Agent Performance")
    if not performance_df.empty:
        st.bar_chart(performance_df.set_index("AgentName")[["ResolutionRate", "AvgSentimentScore"]])
    st.dataframe(performance_df, use_container_width=True, hide_index=True)
    st.download_button(
        "Download performance CSV",
        dataframe_to_csv(performance_df),
        "output_agent_performance_report.csv",
        "text/csv",
    )

with tab_campaign:
    st.subheader("Sales Campaign Recommendations")
    if not campaign_df.empty and "RecommendedChannel" in campaign_df.columns:
        st.bar_chart(campaign_df["RecommendedChannel"].value_counts())
    st.dataframe(campaign_df, use_container_width=True, hide_index=True)
    st.download_button(
        "Download campaign CSV",
        dataframe_to_csv(campaign_df),
        "output_sales_campaign_recommendations.csv",
        "text/csv",
    )
