# A2UI-BQ: Gemini Enterprise FinOps & Usage Pulse Agent

An end-to-end **A2A Protocol (`v0.3.0`) + A2UI (`v0.9`)** reference agent for **Gemini Enterprise** built with the Google Agent Development Kit (**ADK**), **BigQuery MCP (`https://bigquery.googleapis.com/mcp`)**, and **End-User Google OAuth 2.0**.

---

## ✨ Key Features

1. **Instant Personalized Turn 0 Welcome Screen (`<10ms`, Zero LLM Latency)**
   - Emits `createSurface` and `updateComponents` (`DataPart` with `mimeType: application/json+a2ui`) on Turn 0 (`text == ""`) **without** a companion Markdown `TextPart`, avoiding the `ucs-text-streamer.ts` typing-mode block so the 4 `MaterialCard` starter prompt boxes mount immediately on open.
   - Resolves the end-user's display name and email via Google OAuth 2.0 (`userinfo` / `tokeninfo`) and greets them (`👋 Hello <User>!`).

2. **Live BigQuery MCP (`https://bigquery.googleapis.com/mcp`) with End-User OAuth 2.0**
   - Captures the end-user's OAuth 2.0 bearer token (`ya29.*` from Gemini Enterprise `serverSideOauth2` or Entra ID WIF JWT) via [`OAuthCaptureMiddleware`](app/fast_api_app.py) and [`extract_user_oauth_token`](app/oauth_bq_mcp.py), ignoring Cloud Run's service-to-service IAM OIDC invoker token.
   - Executes read-only GoogleSQL queries (`execute_sql_readonly`) and schema discovery (`list_dataset_ids`, `list_table_ids`, `get_table_info`) against BigQuery MCP using the caller's identity:
     - `<GOOGLE_CLOUD_PROJECT>.gemini_enterprise.usage_metrics`
     - `<GOOGLE_CLOUD_PROJECT>.gemini_enterprise_audit_logs.discoveryengine_googleapis_com_gen_ai_client_inference_operation_details`
     - `<GOOGLE_CLOUD_PROJECT>.gemini_enterprise_audit_logs.discoveryengine_googleapis_com_gemini_enterprise_user_activity`

3. **Dual A2UI `v0.9` Visualization Layouts in the Same Agent**
   - **Mode A — Inline `MaterialTabs` (`build_inline_tabs_finops_a2ui_messages`)**:
     Renders inline KPI `MaterialCard`s + a 3-tab `MaterialTabs` view switching between:
     - `📊 Vega-Lite Chart` (`VegaChart` with `"spec": {"path": "/vegaSpec"}` populated via `updateDataModel`)
     - `📋 Sortable Data Table` (`GcbpTable` with server-side `fetchData` column sorting that re-renders the full dashboard with `activeTab: 1`)
     - `🔍 SQL & OAuth Telemetry` (authenticated principal badge, BigQuery job ID, bytes processed, and exact SQL query)
   - **Mode B — Side-Panel `Canvas` (`build_canvas_finops_a2ui_messages`, `autoOpen: true`)**:
     Opens Gemini Enterprise's right-hand resizable side panel with KPI cards, interactive `VegaChart`, sortable `GcbpTable`, and SQL telemetry.
   - **One-Click Mode Switching**: Every Inline `MaterialTabs` view includes an **`Open Side-Panel Canvas View`** button, and every `Canvas` view includes a **`Render as Inline MaterialTabs`** button.

4. **Clean 3-Step Synthetic Thinking Timeline**
   - Streams 3 distinct `TaskStatusUpdateEvent` / `TaskArtifactUpdateEvent` thinking updates (`metadata={"adk_thought": True}`) at a `1.5s` cadence (`🧠 [Synthetic Thought 1/3]`, `⚙️ [Synthetic Thought 2/3]`, `🎨 [Synthetic Thought 3/3]`), strictly separated from A2UI `DataPart` payloads.

---

## 📁 Project Structure

```text
├── app/
│   ├── agent.py                 # A2UIRootRouterAgent, A2UI v0.9 builders (MaterialTabs & Canvas), BigQuery MCP tools
│   ├── oauth_bq_mcp.py          # End-user Google OAuth 2.0 / Entra WIF extraction & BigQuery MCP JSON-RPC client
│   ├── fast_api_app.py          # FastAPI server + OAuthCaptureMiddleware + A2A route mounting
│   └── app_utils/
│       ├── a2a.py               # Custom A2A v0.3.0 <-> ADK converters, A2UI DataPart & thought separation
│       └── services.py          # ADK session & artifact service configuration
├── tests/
│   ├── unit/
│   │   └── test_agent.py        # Unit tests for A2UI surfaces, VegaChart data model binding, sorting, and OAuth
│   └── eval/                    # Agent evaluation datasets and configuration
├── AGENTS.md                    # Engineering guide for A2UI v0.9 + BigQuery MCP in Gemini Enterprise
├── sync_ge_metadata.py          # Registers / updates the agent and OAuth config in Gemini Enterprise
├── Dockerfile                   # Cloud Run container definition
├── .env.example                 # Environment variable template
└── pyproject.toml               # Project dependencies (managed via uv)
```

---

## 🚀 Quick Start

### 1. Configure Environment Variables

Copy `.env.example` to `.env` and fill in your Google Cloud project and Gemini Enterprise settings:

```bash
cp .env.example .env
```

### 2. Run Unit Tests

```bash
uv run pytest tests/unit/test_agent.py -v
```

### 3. Deploy to Cloud Run

```bash
gcloud run deploy a2ui-bq-agent \
  --source . \
  --project "$GOOGLE_CLOUD_PROJECT" \
  --region europe-west1 \
  --no-allow-unauthenticated \
  --session-affinity \
  --min-instances 0 \
  --max-instances 1 \
  --memory 2Gi \
  --cpu 2 \
  --set-env-vars="GOOGLE_CLOUD_PROJECT=${GOOGLE_CLOUD_PROJECT},GOOGLE_CLOUD_LOCATION=global,GOOGLE_GENAI_USE_VERTEXAI=True,AGENT_VERSION=0.3.2,APP_URL=${APP_URL}"
```

### 4. Register Agent & OAuth in Gemini Enterprise

Ensure a Discovery Engine OAuth authorization resource (`OAUTH_AUTH_ID`) exists in your target location, then run:

```bash
uv run python sync_ge_metadata.py
```
