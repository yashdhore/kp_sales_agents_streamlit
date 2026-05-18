from __future__ import annotations

from io import StringIO
from pathlib import Path

import pandas as pd
import streamlit as st

from agent_runner import find_agent_config, load_config
from agents import InsightsAgent, PerformanceAgent, SalesCampaignAgent
from agents.data_loader import GitHubCsvLoader, GitHubCsvSources
from agents.llm_client import InteractionLlmClient


st.set_page_config(page_title="Revenue Growth Intelligence Platform", page_icon="RG", layout="wide")

LOGO_PATH = Path("logo/AnalyticsAI Logo_B7.jpg")


st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.25rem;
        padding-bottom: 2rem;
        max-width: 1480px;
    }
    .stApp {
        background: linear-gradient(180deg, #f8fbfd 0%, #ffffff 42%);
    }
    h1 {
        color: #062b5f !important;
        border-left: 6px solid #18c99a;
        padding-left: 0.8rem;
    }
    h1, h2, h3 {
        color: #062b5f;
    }
    .main-subtitle {
        color: #006fd6;
        font-size: 1.04rem;
        font-weight: 600;
        margin-top: -0.25rem;
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
        border-left: 4px solid #18c99a;
        background: #f1fbf8;
        padding: 0.85rem 1rem;
        margin: 0.75rem 0 1.1rem;
        color: #263845;
    }
    .sample-note {
        border-left: 4px solid #006fd6;
        background: #eef7ff;
        padding: 0.8rem 0.95rem;
        margin: 0.65rem 0 1rem;
        color: #1f3344;
        font-size: 0.94rem;
    }
    .agent-copy {
        color: #2f3f4a;
        font-size: 1.02rem;
        line-height: 1.55;
        margin-bottom: 0.8rem;
    }
    .stButton > button, .stDownloadButton > button {
        background: linear-gradient(90deg, #006fd6 0%, #18c99a 100%) !important;
        color: #ffffff !important;
        border: 0 !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        background: linear-gradient(90deg, #0059ad 0%, #10a77f 100%) !important;
        color: #ffffff !important;
        border: 0 !important;
    }
    button[data-baseweb="tab"] {
        background: #e7f3ff;
        color: #062b5f;
        border-radius: 8px 8px 0 0;
        padding: 0.75rem 1.25rem;
        margin-right: 0.25rem;
        font-weight: 700;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(90deg, #006fd6 0%, #18c99a 100%);
        color: #ffffff;
    }
    button[data-baseweb="tab"] p {
        color: inherit;
        font-weight: 700;
    }
    .insight-card {
        background: #ffffff;
        border: 1px solid #d9e8f2;
        border-radius: 8px;
        padding: 1rem 1.15rem;
        margin: 0.75rem 0 1.15rem;
        box-shadow: 0 1px 2px rgba(6, 43, 95, 0.06);
    }
    .insight-card ul {
        margin: 0;
        padding-left: 1.15rem;
    }
    .insight-card li {
        margin: 0.35rem 0;
        color: #1f3344;
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


def load_uploaded_or_github(
    chats_file,
    calls_file,
    clients_file,
    chats_url: str,
    calls_url: str,
    clients_url: str,
) -> tuple[dict[str, pd.DataFrame], str]:
    uploaded_files = [chats_file, calls_file, clients_file]
    if any(uploaded_files):
        if not all(uploaded_files):
            raise ValueError("Upload all three local CSV files: chats, phone calls, and Salesforce clients.")
        return (
            {
                "chats": pd.read_csv(chats_file),
                "phone_calls": pd.read_csv(calls_file),
                "salesforce_clients": pd.read_csv(clients_file),
            },
            "local CSV uploads",
        )
    return load_github_data(chats_url, calls_url, clients_url), "sample GitHub CSVs"


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
        "The bullets below are derived from the currently loaded CSV data and enriched by the configured "
        "backend LLM, with deterministic fallback logic available when the LLM is not enabled.</div>",
        unsafe_allow_html=True,
    )


def show_sample_note(data_source_label: str) -> None:
    st.markdown(
        f"<div class='sample-note'><strong>Data context:</strong> This dashboard is currently using "
        f"<strong>{data_source_label}</strong>. The metrics, bullets, and LLM-enriched insights shown here "
        "are based only on the CSV files loaded in this session. Use the sidebar to upload local CSVs for a client-specific readout.</div>",
        unsafe_allow_html=True,
    )


def show_bullets(items: list[str]) -> None:
    html_items = "".join(f"<li>{item}</li>" for item in items)
    st.markdown(f"<div class='insight-card'><ul>{html_items}</ul></div>", unsafe_allow_html=True)


def top_value(df: pd.DataFrame, column: str, default: str = "N/A") -> tuple[str, int]:
    if df.empty or column not in df.columns:
        return default, 0
    counts = df[column].dropna().astype(str).value_counts()
    if counts.empty:
        return default, 0
    return counts.index[0], int(counts.iloc[0])


def pct(numerator: int | float, denominator: int | float) -> str:
    if not denominator:
        return "0%"
    return f"{(numerator / denominator) * 100:.0f}%"


def build_insight_bullets(insights: pd.DataFrame) -> list[str]:
    total = len(insights)
    top_intent, top_intent_count = top_value(insights, "Intent")
    top_sentiment, top_sentiment_count = top_value(insights, "Sentiment")
    high_risk = int((insights.get("RiskLevel", pd.Series(dtype=str)).astype(str) == "High").sum())
    unresolved = int((insights.get("Resolved", pd.Series(dtype=str)).astype(str).str.lower() != "yes").sum())
    top_channel, top_channel_count = top_value(insights, "SalesChannel")
    return [
        f"{top_intent} is the leading customer intent, representing {top_intent_count:,} of {total:,} analyzed interactions.",
        f"{top_sentiment} is the dominant sentiment signal across {top_sentiment_count:,} interactions, shaping the current customer experience narrative.",
        f"{unresolved:,} interactions remain unresolved, creating a {pct(unresolved, total)} service-to-growth recovery opportunity.",
        f"{high_risk:,} interactions are classified as high risk, giving leadership a focused queue for retention intervention.",
        f"{top_channel} is the highest-volume interaction channel with {top_channel_count:,} records, indicating where customer voice is currently concentrated.",
    ]


def build_performance_bullets(performance: pd.DataFrame, insights: pd.DataFrame) -> list[str]:
    total_agents = performance["AgentName"].nunique() if not performance.empty and "AgentName" in performance.columns else 0
    top_agent = "N/A"
    top_resolution = 0.0
    coaching_count = 0
    avg_resolution = 0.0
    sentiment_leader = "N/A"
    if not performance.empty:
        sorted_resolution = performance.sort_values("ResolutionRate", ascending=False)
        top_agent = str(sorted_resolution.iloc[0]["AgentName"])
        top_resolution = float(sorted_resolution.iloc[0]["ResolutionRate"])
        coaching_count = int((performance["Category"] == "Train").sum()) if "Category" in performance.columns else 0
        avg_resolution = float(performance["ResolutionRate"].mean()) if "ResolutionRate" in performance.columns else 0.0
        if "AvgSentimentScore" in performance.columns:
            sentiment_leader = str(performance.sort_values("AvgSentimentScore", ascending=False).iloc[0]["AgentName"])
    unresolved = int((insights.get("Resolved", pd.Series(dtype=str)).astype(str).str.lower() != "yes").sum())
    return [
        f"{total_agents:,} frontline agents are scored from the current interaction dataset.",
        f"{top_agent} has the strongest resolution signal at {top_resolution:.0%}, creating a benchmark for playbook replication.",
        f"The average resolution rate is {avg_resolution:.0%}, showing the current operating baseline for customer issue closure.",
        f"{coaching_count:,} agents fall into the coaching category, highlighting where enablement can lift customer outcomes.",
        f"{sentiment_leader} leads the sentiment quality signal, while {unresolved:,} unresolved cases show where execution can still protect revenue.",
    ]


def build_campaign_bullets(campaign: pd.DataFrame, insights: pd.DataFrame) -> list[str]:
    targets = len(campaign)
    top_channel, top_channel_count = top_value(campaign, "RecommendedChannel")
    top_intent, top_intent_count = top_value(campaign, "Intent")
    negative = int((campaign.get("Sentiment", pd.Series(dtype=str)).astype(str) == "Negative").sum())
    none_count = int((campaign.get("RecommendedChannel", pd.Series(dtype=str)).astype(str) == "None").sum())
    actionable = targets - none_count
    high_risk = int((insights.get("RiskLevel", pd.Series(dtype=str)).astype(str) == "High").sum())
    return [
        f"{targets:,} customer records qualify for campaign review from the loaded data.",
        f"{top_channel} is the leading recommended outreach path, assigned to {top_channel_count:,} customer opportunities.",
        f"{top_intent} is the top campaign-triggering intent, appearing in {top_intent_count:,} recommended outreach records.",
        f"{negative:,} campaign candidates carry negative sentiment, giving marketing a focused retention and recovery segment.",
        f"{actionable:,} records have an actionable channel recommendation, with {high_risk:,} high-risk signals available for prioritization.",
    ]


def build_executive_summary_bullets(insights: pd.DataFrame, performance: pd.DataFrame, campaign: pd.DataFrame) -> list[str]:
    total = len(insights)
    unresolved = int((insights.get("Resolved", pd.Series(dtype=str)).astype(str).str.lower() != "yes").sum())
    high_risk = int((insights.get("RiskLevel", pd.Series(dtype=str)).astype(str) == "High").sum())
    top_intent, top_intent_count = top_value(insights, "Intent")
    top_channel, top_channel_count = top_value(campaign, "RecommendedChannel")
    avg_resolution = float(performance["ResolutionRate"].mean()) if not performance.empty and "ResolutionRate" in performance.columns else 0.0
    return [
        f"The CMO-level signal is concentrated around {top_intent}, which accounts for {top_intent_count:,} of {total:,} analyzed interactions.",
        f"{unresolved:,} unresolved interactions create an immediate recovery opportunity across service, sales, and lifecycle marketing.",
        f"{high_risk:,} customer conversations are tagged high risk, making them candidates for priority retention or escalation workflows.",
        f"The current agent resolution baseline is {avg_resolution:.0%}, giving leadership a measurable operating lever for improving experience quality.",
        f"{top_channel} is the leading campaign action path with {top_channel_count:,} recommendations, translating customer voice into activation.",
    ]


def build_financial_bullets(insights: pd.DataFrame, campaign: pd.DataFrame) -> list[str]:
    lifetime_value = pd.to_numeric(insights.get("LifetimeValue", pd.Series(dtype=float)), errors="coerce")
    avg_ltv = float(lifetime_value.dropna().mean()) if not lifetime_value.dropna().empty else 0.0
    unresolved = int((insights.get("Resolved", pd.Series(dtype=str)).astype(str).str.lower() != "yes").sum())
    high_risk = int((insights.get("RiskLevel", pd.Series(dtype=str)).astype(str) == "High").sum())
    actionable = int((campaign.get("RecommendedChannel", pd.Series(dtype=str)).astype(str) != "None").sum())
    estimated_risk = high_risk * avg_ltv
    recovery_pool = unresolved * avg_ltv * 0.15
    return [
        f"Average customer lifetime value in the loaded CSVs is approximately ${avg_ltv:,.0f}, based on available Salesforce client records.",
        f"High-risk interactions imply an indicative revenue-at-risk pool of ${estimated_risk:,.0f} if those accounts are not recovered.",
        f"Unresolved interactions represent a modeled recovery opportunity of ${recovery_pool:,.0f}, assuming a 15% save or conversion impact.",
        f"{actionable:,} records have a concrete follow-up channel recommendation, giving marketing a measurable activation queue.",
        "These estimates are directional for executive prioritization and should be recalibrated with client-specific margin, churn, and conversion assumptions.",
    ]


def build_customer_journey_bullets(selected_row: pd.Series) -> list[str]:
    return [
        f"Customer {selected_row.get('ClientID', 'N/A')} entered through the {selected_row.get('SalesChannel', 'unknown')} channel with intent classified as {selected_row.get('Intent', 'Unknown')}.",
        f"The LLM-enriched sentiment is {selected_row.get('Sentiment', 'Unknown')} with risk level {selected_row.get('RiskLevel', 'Unknown')}.",
        f"Resolution status is {selected_row.get('Resolved', 'Unknown')}, which determines whether this record should move into campaign or service recovery.",
        f"The recommended next action is: {selected_row.get('NextBestAction', 'Review and follow up with the customer.')}",
        f"Executive readout: {selected_row.get('ExecutiveInsight', 'Use this interaction to validate the customer journey and campaign response path.')}",
    ]


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
    st.title("Revenue Growth Intelligence Platform")
    st.markdown(
        "<div class='main-subtitle'>AI agents that convert customer conversations into CMO-ready growth, "
        "performance, and campaign activation signals.</div>",
        unsafe_allow_html=True,
    )
with logo_col:
    show_logo(width=170)

with st.sidebar:
    st.header("Data Sources")
    st.markdown(
        "Default GitHub CSVs are sample files. Upload local CSVs below to replace the sample data for this session."
    )
    chats_file = st.file_uploader("Upload local chats CSV", type=["csv"])
    calls_file = st.file_uploader("Upload local phone calls CSV", type=["csv"])
    clients_file = st.file_uploader("Upload local Salesforce clients CSV", type=["csv"])
    st.divider()
    st.caption("Sample GitHub CSV URLs")
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
    with st.spinner("Reading CSVs and running AI agents..."):
        frames, data_source_label = load_uploaded_or_github(
            chats_file,
            calls_file,
            clients_file,
            chats_url,
            calls_url,
            clients_url,
        )
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

show_sample_note(data_source_label)

tab_summary, tab_financial, tab_journey, tab_insights, tab_performance, tab_campaign = st.tabs(
    [
        "Executive Summary",
        "Financial Lens",
        "Customer Journey",
        "Insights Agent",
        "Performance Agent",
        "Sales Campaign Agent",
    ]
)

with tab_summary:
    show_tab_header(
        "Executive Summary",
        "A CMO-ready view of the most important growth, retention, and campaign signals from the loaded interaction data.",
    )
    show_llm_note()
    show_bullets(build_executive_summary_bullets(insights_df, performance_df, campaign_df))
    summary_cols = st.columns(3)
    if "Intent" in insights_df.columns:
        summary_cols[0].bar_chart(insights_df["Intent"].value_counts().head(8))
    if "RiskLevel" in insights_df.columns:
        summary_cols[1].bar_chart(insights_df["RiskLevel"].value_counts())
    if "RecommendedChannel" in campaign_df.columns:
        summary_cols[2].bar_chart(campaign_df["RecommendedChannel"].value_counts())

with tab_financial:
    show_tab_header(
        "Financial Lens",
        "A directional revenue view that translates customer risk and campaign recommendations into an executive opportunity frame.",
    )
    show_llm_note()
    show_bullets(build_financial_bullets(insights_df, campaign_df))
    lifetime_value = pd.to_numeric(insights_df.get("LifetimeValue", pd.Series(dtype=float)), errors="coerce")
    finance_cols = st.columns(3)
    finance_cols[0].metric("Average Lifetime Value", f"${lifetime_value.dropna().mean():,.0f}" if not lifetime_value.dropna().empty else "N/A")
    finance_cols[1].metric("Actionable Campaign Records", f"{(campaign_df.get('RecommendedChannel', pd.Series(dtype=str)).astype(str) != 'None').sum():,}")
    finance_cols[2].metric("High Risk Records", f"{(insights_df.get('RiskLevel', pd.Series(dtype=str)).astype(str) == 'High').sum():,}")
    st.dataframe(
        campaign_df[["ClientID", "RecommendedChannel", "RecommendationScore", "Sentiment", "Intent", "Resolved"]]
        if not campaign_df.empty
        else campaign_df,
        use_container_width=True,
        hide_index=True,
    )

with tab_journey:
    show_tab_header(
        "Customer Journey",
        "A drill-through view showing how one customer interaction becomes insight, risk, and a recommended activation path.",
    )
    show_llm_note()
    journey_options = insights_df["ClientID"].dropna().astype(str).tolist() if "ClientID" in insights_df.columns else []
    selected_client = st.selectbox("Select customer record", journey_options) if journey_options else None
    if selected_client:
        selected_row = insights_df[insights_df["ClientID"].astype(str) == selected_client].iloc[0]
        show_bullets(build_customer_journey_bullets(selected_row))
        st.text_area("Transcript", value=str(selected_row.get("Transcript", "")), height=180)
        st.dataframe(pd.DataFrame([selected_row]), use_container_width=True, hide_index=True)

with tab_insights:
    show_tab_header(
        "Insights Agent",
        "Transforms raw customer transcripts into a boardroom-ready view of sentiment, intent, risk, and next best action.",
    )
    show_llm_note()
    show_bullets(build_insight_bullets(insights_df))
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
    show_bullets(build_performance_bullets(performance_df, insights_df))
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
    show_bullets(build_campaign_bullets(campaign_df, insights_df))
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
