# KP Sales Agents Streamlit

Streamlit app that runs three CSV-backed sales agents without Cosmos DB.

## Architecture

```text
GitHub raw CSV URLs
        |
        v
Streamlit UI
        |
        v
GitHubCsvLoader
        |
        v
InsightsAgent -> PerformanceAgent
             \-> SalesCampaignAgent
```

## Agents

- `InsightsAgent`: combines chat, phone call, and Salesforce client CSV data into interaction insights.
- `PerformanceAgent`: scores agent performance from the generated insights.
- `SalesCampaignAgent`: recommends the best follow-up channel for unresolved or at-risk customer interactions.

## Expected CSV Files

The app expects raw GitHub CSV URLs for:

- chats
- phone calls
- Salesforce clients

Use raw file URLs, not normal GitHub blob page URLs. A raw URL usually looks like:

```text
https://raw.githubusercontent.com/<owner>/<repo>/<branch>/data/input_chats.csv
```

## Run Locally

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

## Configuration

Default CSV URLs and agent settings live in `config/agents_config.yaml`.

Cosmos DB is not used in this version.
