# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Starter prompts, predefined BigQuery MCP SQL queries, and system instructions
for the A2UI-BQ Gemini Enterprise FinOps & Usage Pulse Agent."""

from __future__ import annotations

import os

USER_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT", "your-gcp-project-id")

A2UI_V09_CATALOG_ID = (
    "https://www.gstatic.com/vertexaisearch/a2ui/v0_9/"
    "gemini_enterprise_composite_catalog.json"
)

STARTER_PROMPT_1 = (
    "Show token usage & estimated cost by agent in inline MaterialTabs"
)
STARTER_PROMPT_2 = (
    "Open Gemini Enterprise daily activity & search pulse in Side-Panel Canvas"
)
STARTER_PROMPT_3 = (
    "Analyze Gemini Enterprise API & agent audit activity across services"
)
STARTER_PROMPT_4 = (
    "Discover all Gemini Enterprise BigQuery tables and summarize top cost drivers"
)
RETURN_WELCOME_PROMPT = (
    "Show the A2UI v0.9 welcome view and starter prompt cards"
)

STARTER_PROMPTS_LIST = [
    STARTER_PROMPT_1,
    STARTER_PROMPT_2,
    STARTER_PROMPT_3,
    STARTER_PROMPT_4,
]

# Read-only GoogleSQL queries executed via BigQuery MCP (`execute_sql_readonly`) using End-User OAuth
SQL_TOKEN_FINOPS_BY_AGENT = f"""
SELECT
  COALESCE(jsonPayload.gen_ai_agent_name, resource.labels.agent_id, 'default_assistant') AS agent_name,
  COUNT(*) AS inference_calls,
  SUM(CAST(COALESCE(jsonPayload.gen_ai_usage_input_tokens, 0) AS INT64)) AS input_tokens,
  SUM(CAST(COALESCE(jsonPayload.gen_ai_usage_cache_read_input_tokens, 0) AS INT64)) AS cached_tokens,
  SUM(CAST(COALESCE(jsonPayload.gen_ai_usage_output_tokens, 0) AS INT64)) AS output_tokens,
  SUM(CAST(COALESCE(jsonPayload.gen_ai_usage_reasoning_output_tokens, 0) AS INT64)) AS reasoning_tokens
FROM `{USER_PROJECT}.gemini_enterprise_audit_logs.discoveryengine_googleapis_com_gen_ai_client_inference_operation_details`
GROUP BY agent_name
ORDER BY input_tokens DESC
LIMIT 10
""".strip()

SQL_DAILY_USAGE_METRICS = f"""
SELECT
  date,
  SUM(CAST(COALESCE(search_count, 0) AS INT64)) AS searches,
  SUM(CAST(COALESCE(answer_count, 0) AS INT64)) AS answers,
  SUM(CAST(COALESCE(action_count, 0) AS INT64)) AS actions,
  MAX(CAST(COALESCE(daily_active_user_count, 0) AS INT64)) AS dau
FROM `{USER_PROJECT}.gemini_enterprise.usage_metrics`
WHERE date IS NOT NULL
GROUP BY date
HAVING searches > 0 OR answers > 0 OR actions > 0 OR dau > 0
ORDER BY date ASC
LIMIT 30
""".strip()

SQL_AUDIT_ACTIVITY_BY_METHOD = f"""
SELECT
  COALESCE(resource.labels.method, 'UnknownMethod') AS method,
  SPLIT(COALESCE(resource.labels.service, 'discoveryengine'), '.')[SAFE_OFFSET(3)] AS service_short,
  COUNT(*) AS event_count,
  CAST(MAX(DATE(timestamp)) AS STRING) AS latest_date
FROM `{USER_PROJECT}.gemini_enterprise_audit_logs.discoveryengine_googleapis_com_gemini_enterprise_user_activity`
GROUP BY method, service_short
ORDER BY event_count DESC
LIMIT 10
""".strip()


def build_specialist_instruction(project_id: str = USER_PROJECT) -> str:
    """Return the domain-specific system instruction for `gemini_a2ui_specialist`."""
    return (
        f"You are the `gemini_a2ui_specialist` for the Gemini Enterprise FinOps & Usage Pulse agent in project `{project_id}`.\n"
        "Use the BigQuery MCP tools (`bq_mcp_list_datasets_and_tables`, `bq_mcp_get_table_info`, `bq_mcp_execute_sql_readonly`) "
        "to inspect schemas and run read-only SQL queries.\n"
        "If the user asks to configure or create a FinOps budget alert threshold, call `bq_mcp_create_finops_budget_alert` "
        "(which enforces Human-in-the-Loop approval before executing).\n"
        f"Key tables in project `{project_id}`:\n"
        f"- `{project_id}.gemini_enterprise.usage_metrics` (daily search_count, answer_count, action_count, daily_active_user_count)\n"
        f"- `{project_id}.gemini_enterprise_audit_logs.discoveryengine_googleapis_com_gen_ai_client_inference_operation_details` (token consumption per agent)\n"
        f"- `{project_id}.gemini_enterprise_audit_logs.discoveryengine_googleapis_com_gemini_enterprise_user_activity` (API & agent audit events)\n\n"
        "Every BigQuery MCP call executes using the end-user's OAuth 2.0 identity. "
        "If a tool returns a `GuidedToolErrorRecovery` payload (`status='error'`), follow its `recovery_steps` and `suggested_fallback_tool` to self-correct. "
        "Always show the SQL query you ran and provide a concise, executive FinOps summary."
    )
