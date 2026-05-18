from __future__ import annotations

from io import StringIO
from pathlib import Path

import pandas as pd
import streamlit as st

from agent_runner import find_agent_config, load_config
from agents import InsightsAgent, PerformanceAgent, SalesCampaignAgent
from agents.data_loader import GitHubCsvLoader, GitHubCsvSources
from agents.llm_client import InteractionLlmClient


st.set_page_config(page_title="Customer Growth Intelligence", page_icon="CG", layout="wide")

LOGO_PATH = Path("logo/AnalyticsAI Logo_B7.jpg")

AGENT_BULLETS = {
    "insights": [
        "Pinpoints the top customer intents behind demand, churn signals, and conversion friction.",
        "Separates positive, neutral, and negative conversations so leaders can see sentiment movement fast.",
        "Flags unresolved interactions that need immediate executive attention or field follow-up.",
        "Summarizes conversation-level risk and next-best-action signals for revenue protection.",
        "Turns unstructured chats and calls into a reusable intelligence layer for marketing, sales, and service.",
    ],
    "performance": [
        "Ranks frontline execution by resolution rate, sentiment quality, and interaction volume.",
        "Identifies which agents are creating the strongest customer outcomes and which need coaching.",
        "Connects customer experience quality to measurable operating signals the CMO can act on.",
        "Highlights where training, playbooks, or escalation paths can improve conversion and retention.",
        "Creates a repeatable performance view that can scale across regions, channels, and campaigns.",
    ],
    "campaign": [
        "Prioritizes customers who need outreach before dissatisfaction becomes churn or lost pipeline.",
        "Recommends phone, email, or SMS follow-up based on sentiment, channel history, and resolution status.",
        "Gives marketing a direct bridge from customer conversations to targeted campaign activation.",
        "Helps sales teams focus high-touch effort where human intervention can protect the most value.",
        "Creates a measurable next-best-action queue for retention, win-back, and conversion programs.",
    ],
}


st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.25rem;
        padding-bottom: 2rem;
        max-width: 1480px;
    }
    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #d9e1e8;
        border-radius: 8px;
        padding: 14px 16px;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.06);
    }
    div[data-testid="stMetricLabel"] {
        color: #4b5f6f;
        font-size: 0.82rem;
    }
    .executive-note {
        border-left: 4px solid #2f6f9f;
        background: #f4f8fb;
        padding: 0.85rem 1rem;
        margin: 0.75rem 0 1.1rem;
        color: #263845;
    }
    .agent-copy {
        color: #2f3f4a;
        font-size: 1.02rem;
        line-height: 1.55;
        margin-bottom: 0.8rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


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
    max_interactions: int,
) -> dict[str, pd.DataFrame]:
    chats = pd.read_json(StringIO(chats_csv), orient="split")
    calls = pd.read_json(StringIO(calls_csv), orient="split")
    clients = pd.read_json(StringIO(clients_csv), orient="split")
    if max_interactions > 0:
        chats, calls = trim_interactions(chats, calls, max_interactions)

    insights = InsightsAgent(find_agent_config(config, "InsightsAgent")).run(chats, calls, clients)
    performance = PerformanceAgent(find_agent_config(config, "PerformanceAgent")).run(insights)
    campaign = SalesCampaignAgent(find_agent_config(config, "SalesCampaignAgent")).run(insights)
    return {"insights": insights, "performance": performance, "campaign": campaign}


def dataframe_to_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def trim_interactions(chats: pd.DataFrame, calls: pd.DataFrame, max_interactions: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    chat_limit = max_interactions // 2
    call_limit = max_interactions - chat_limit
    return chats.head(chat_limit), calls.head(call_limit)


def show_logo(width: int = 150) -> None:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=width)


def show_llm_note() -> None:
    st.markdown(
        "<div class='executive-note'><strong>LLM-driven insight layer:</strong> "
        "The bullets and tab outputs are generated from the underlying customer interaction data "
        "using the configured backend LLM, with deterministic fallback logic available when the LLM is not enabled.</div>",
        unsafe_allow_html=True,
    )


def show_bullets(items: list[str]) -> None:
    for item in items:
        st.markdown(f"- {item}")


def show_tab_header(title: str, description: str) -> None:
    title_col, logo_col = st.columns([0.82, 0.18], vertical_alignment="center")
    with title_col:
        st.subheader(title)
        st.markdown(f"<div class='agent-copy'>{description}</div>", unsafe_allow_html=True)
    with logo_col:
        show_logo(width=125)


config = load_config()
source_defaults = config.get("data_sources", {})
llm_status = InteractionLlmClient()

hero_col, logo_col = st.columns([0.78, 0.22], vertical_alignment="center")
with hero_col:
    st.title("Customer Growth Intelligence Command Center")
    st.caption(
        "AI agents that convert customer conversations into CMO-ready growth, performance, "
        "and campaign activation signals."
    )
with logo_col:
    show_logo(width=170)

with st.sidebar:
    st.header("Data Sources")
    chats_url = st.text_input("Chats CSV raw URL", value=source_defaults.get("chats_url", ""))
    calls_url = st.text_input("Phone calls CSV raw URL", value=source_defaults.get("phone_calls_url", ""))
    clients_url = st.text_input("Salesforce clients CSV raw URL", value=source_defaults.get("salesforce_clients_url", ""))
    max_interactions = st.slider("Interaction sample size", min_value=20, max_value=200, value=60, step=20)
    st.divider()
    st.header("Backend LLM")
    if llm_status.available:
        st.success(f"Enabled: {llm_status.model}")
    else:
        st.warning("Fallback mode: add OPENAI_API_KEY to .env to enable LLM insights.")
    refresh_clicked = st.button("Refresh agents", type="primary", use_container_width=True)

if refresh_clicked:
    load_github_data.clear()
    run_agents.clear()

try:
    with st.spinner("Reading GitHub CSVs and running AI agents..."):
        frames = load_github_data(chats_url, calls_url, clients_url)
        result = run_agents(
            frames["chats"].to_json(orient="split"),
            frames["phone_calls"].to_json(orient="split"),
            frames["salesforce_clients"].to_json(orient="split"),
            config,
            max_interactions,
        )
except Exception as exc:
    st.error(f"Unable to run agents: {exc}")
    st.stop()

insights_df = result["insights"]
performance_df = result["performance"]
campaign_df = result["campaign"]

metric_cols = st.columns(5)
metric_cols[0].metric("Customer Interactions", f"{len(insights_df):,}")
metric_cols[1].metric("Agents Scored", f"{performance_df['AgentName'].nunique() if not performance_df.empty else 0:,}")
metric_cols[2].metric("Campaign Opportunities", f"{len(campaign_df):,}")
metric_cols[3].metric("Unresolved Cases", f"{(insights_df['Resolved'].astype(str).str.lower() != 'yes').sum():,}")
metric_cols[4].metric("High Risk Signals", f"{(insights_df.get('RiskLevel', pd.Series(dtype=str)).astype(str) == 'High').sum():,}")

tab_insights, tab_performance, tab_campaign = st.tabs(
    ["Insights Agent", "Performance Agent", "Sales Campaign Agent"]
)

with tab_insights:
    show_tab_header(
        "Insights Agent",
        "Transforms raw customer transcripts into a boardroom-ready view of sentiment, intent, risk, and next best action.",
    )
    show_llm_note()
    show_bullets(AGENT_BULLETS["insights"])
    chart_cols = st.columns(3)
    if "Sentiment" in insights_df.columns:
        chart_cols[0].bar_chart(insights_df["Sentiment"].value_counts())
    if "Intent" in insights_df.columns:
        chart_cols[1].bar_chart(insights_df["Intent"].value_counts())
    if "RiskLevel" in insights_df.columns:
        chart_cols[2].bar_chart(insights_df["RiskLevel"].value_counts())
    st.dataframe(insights_df, use_container_width=True, hide_index=True)
    st.download_button("Download insights CSV", dataframe_to_csv(insights_df), "output_insights.csv", "text/csv")

with tab_performance:
    show_tab_header(
        "Performance Agent",
        "Connects customer outcomes to agent execution so leadership can see where performance is creating or limiting growth.",
    )
    show_llm_note()
    show_bullets(AGENT_BULLETS["performance"])
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
    show_tab_header(
        "Sales Campaign Agent",
        "Turns unresolved and at-risk interactions into prioritized outreach actions for conversion, retention, and win-back teams.",
    )
    show_llm_note()
    show_bullets(AGENT_BULLETS["campaign"])
    chart_cols = st.columns(2)
    if not campaign_df.empty and "RecommendedChannel" in campaign_df.columns:
        chart_cols[0].bar_chart(campaign_df["RecommendedChannel"].value_counts())
    if not campaign_df.empty and "Sentiment" in campaign_df.columns:
        chart_cols[1].bar_chart(campaign_df["Sentiment"].value_counts())
    st.dataframe(campaign_df, use_container_width=True, hide_index=True)
    st.download_button(
        "Download campaign CSV",
        dataframe_to_csv(campaign_df),
        "output_sales_campaign_recommendations.csv",
        "text/csv",
    )
