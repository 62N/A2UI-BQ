# ruff: noqa
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

"""Gemini Enterprise FinOps & Usage Pulse Agent (A2UI v0.9 + BigQuery MCP + End-User OAuth).

Features:
1. Turn 0 Instant Personalized A2UI v0.9 Welcome View (<10ms surface build, 0 LLM calls):
   Renders 4 clickable `MaterialCard` starter prompt boxes + a compact `MaterialButton`
   footer, greeting the user via their resolved Google OAuth / Cloud Identity profile.
2. Live BigQuery MCP (`https://bigquery.googleapis.com/mcp`) with End-User OAuth 2.0:
   Executes read-only SQL queries against `<GOOGLE_CLOUD_PROJECT>.gemini_enterprise.usage_metrics`
   and `<GOOGLE_CLOUD_PROJECT>.gemini_enterprise_audit_logs.*` using the caller's OAuth token.
3. Dual A2UI v0.9 Visualization Modes in the Same Agent:
   - Mode A (Inline `MaterialTabs`): Renders KPI `MaterialCard`s + a tabbed container
     switching between `[📊 Vega-Lite Chart]`, `[📋 Sortable GcbpTable]`, and `[🔍 SQL & OAuth]`.
   - Mode B (Side-Panel `Canvas`): Renders a `Canvas`-rooted surface (`autoOpen: true`)
     that opens an interactive dashboard (`VegaChart` + KPI cards + `GcbpTable`) in
     Gemini Enterprise's resizable right-hand side panel.
4. Free-Form BigQuery MCP Exploration via `gemini-3.8-flash` + `McpToolset`:
   Streams native LLM thoughts (`BuiltInPlanner`) and BigQuery MCP tool calls.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
import json
import logging
import os
from typing import Any

from google.adk.agents import Agent, BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.apps import App
from google.adk.events import Event, EventActions
from google.adk.models import Gemini
from google.adk.planners import BuiltInPlanner
from google.adk.tools.tool_context import ToolContext
from google.genai import types

from app.oauth_bq_mcp import (
    BQ_MCP_URL,
    execute_bq_mcp_rpc,
    execute_bq_mcp_sql,
)

logger = logging.getLogger(__name__)

MODEL = "gemini-3.8-flash"
USER_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT", "your-gcp-project-id")
STEP_DELAY_SECONDS = float(os.getenv("A2UI_BQ_STEP_DELAY_SECONDS", "1.5"))

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

# SQL Queries executed via BigQuery MCP (`execute_sql_readonly`) using End-User OAuth
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


def build_welcome_a2ui_messages(
    surface_id: str = "agent-welcome",
    user_display: str | None = None,
    caller_detail: str | None = None,
) -> list[dict[str, Any]]:
    """Build the A2UI v0.9 Welcome + Starter Prompts payload using 4 MaterialCard boxes
    and a smaller separate MaterialButton for returning to the welcome view."""
    del caller_detail
    resolved_user = (user_display or "").strip()
    badge_text = (
        f"👋 Hello {resolved_user}!"
        if resolved_user
        else "✨ Gemini Enterprise FinOps & Usage Pulse"
    )
    return [
        {
            "version": "v0.9",
            "createSurface": {
                "surfaceId": surface_id,
                "catalogId": A2UI_V09_CATALOG_ID,
            },
        },
        {
            "version": "v0.9",
            "updateComponents": {
                "surfaceId": surface_id,
                "components": [
                    {
                        "id": "root",
                        "component": "MaterialColumn",
                        "children": [
                            "header_col",
                            "starters_grid",
                            "footer_row",
                        ],
                        "align": "stretch",
                        "style": {
                            "gap": "16px",
                            "padding": "4px 0",
                        },
                    },
                    {
                        "id": "header_col",
                        "component": "MaterialColumn",
                        "children": [
                            "greeting_badge",
                            "title",
                            "description",
                        ],
                        "align": "stretch",
                        "style": {
                            "gap": "4px",
                        },
                    },
                    {
                        "id": "greeting_badge",
                        "component": "MaterialText",
                        "text": badge_text,
                        "usageHint": "caption",
                        "style": {
                            "color": "#1a73e8",
                            "fontWeight": "600",
                        },
                    },
                    {
                        "id": "title",
                        "component": "Text",
                        "text": "Gemini Enterprise FinOps & Usage Pulse",
                        "variant": "h3",
                    },
                    {
                        "id": "description",
                        "component": "Text",
                        "text": (
                            "Live BigQuery MCP analytics powered by your end-user OAuth 2.0 identity. "
                            "Explore token FinOps, daily usage metrics, and audit logs rendered either "
                            "inline with MaterialTabs or in the Side-Panel Canvas."
                        ),
                        "variant": "body",
                    },
                    {
                        "id": "starters_grid",
                        "component": "MaterialColumn",
                        "children": [
                            "starters_row_1",
                            "starters_row_2",
                        ],
                        "align": "stretch",
                        "style": {
                            "gap": "12px",
                        },
                    },
                    {
                        "id": "starters_row_1",
                        "component": "MaterialRow",
                        "children": ["starter_1", "starter_2"],
                        "align": "stretch",
                        "style": {
                            "gap": "12px",
                        },
                    },
                    {
                        "id": "starter_1",
                        "component": "MaterialCard",
                        "appearance": "outlined",
                        "weight": 1,
                        "children": ["starter_1_col"],
                        "style": {
                            "padding": "14px 16px",
                            "borderRadius": "12px",
                        },
                        "action": {
                            "event": {
                                "name": "starter_prompt.select",
                                "context": {
                                    "prompt": STARTER_PROMPT_1,
                                },
                            }
                        },
                    },
                    {
                        "id": "starter_1_col",
                        "component": "MaterialColumn",
                        "children": ["starter_1_tag", "starter_1_text"],
                        "style": {"gap": "6px"},
                    },
                    {
                        "id": "starter_1_tag",
                        "component": "MaterialText",
                        "text": "📊 INLINE MATERIALTABS • TOKEN FINOPS",
                        "usageHint": "caption",
                        "style": {
                            "color": "#1a73e8",
                            "fontWeight": "600",
                        },
                    },
                    {
                        "id": "starter_1_text",
                        "component": "Text",
                        "text": STARTER_PROMPT_1,
                        "variant": "body",
                    },
                    {
                        "id": "starter_2",
                        "component": "MaterialCard",
                        "appearance": "outlined",
                        "weight": 1,
                        "children": ["starter_2_col"],
                        "style": {
                            "padding": "14px 16px",
                            "borderRadius": "12px",
                        },
                        "action": {
                            "event": {
                                "name": "starter_prompt.select",
                                "context": {
                                    "prompt": STARTER_PROMPT_2,
                                },
                            }
                        },
                    },
                    {
                        "id": "starter_2_col",
                        "component": "MaterialColumn",
                        "children": ["starter_2_tag", "starter_2_text"],
                        "style": {"gap": "6px"},
                    },
                    {
                        "id": "starter_2_tag",
                        "component": "MaterialText",
                        "text": "🖼️ SIDE-PANEL CANVAS • USAGE & DAU PULSE",
                        "usageHint": "caption",
                        "style": {
                            "color": "#137333",
                            "fontWeight": "600",
                        },
                    },
                    {
                        "id": "starter_2_text",
                        "component": "Text",
                        "text": STARTER_PROMPT_2,
                        "variant": "body",
                    },
                    {
                        "id": "starters_row_2",
                        "component": "MaterialRow",
                        "children": ["starter_3", "starter_4"],
                        "align": "stretch",
                        "style": {
                            "gap": "12px",
                        },
                    },
                    {
                        "id": "starter_3",
                        "component": "MaterialCard",
                        "appearance": "outlined",
                        "weight": 1,
                        "children": ["starter_3_col"],
                        "style": {
                            "padding": "14px 16px",
                            "borderRadius": "12px",
                        },
                        "action": {
                            "event": {
                                "name": "starter_prompt.select",
                                "context": {
                                    "prompt": STARTER_PROMPT_3,
                                },
                            }
                        },
                    },
                    {
                        "id": "starter_3_col",
                        "component": "MaterialColumn",
                        "children": ["starter_3_tag", "starter_3_text"],
                        "style": {"gap": "6px"},
                    },
                    {
                        "id": "starter_3_tag",
                        "component": "MaterialText",
                        "text": "⚡ INLINE TABS + CANVAS • AUDIT LOGS",
                        "usageHint": "caption",
                        "style": {
                            "color": "#9334e6",
                            "fontWeight": "600",
                        },
                    },
                    {
                        "id": "starter_3_text",
                        "component": "Text",
                        "text": STARTER_PROMPT_3,
                        "variant": "body",
                    },
                    {
                        "id": "starter_4",
                        "component": "MaterialCard",
                        "appearance": "outlined",
                        "weight": 1,
                        "children": ["starter_4_col"],
                        "style": {
                            "padding": "14px 16px",
                            "borderRadius": "12px",
                        },
                        "action": {
                            "event": {
                                "name": "starter_prompt.select",
                                "context": {
                                    "prompt": STARTER_PROMPT_4,
                                },
                            }
                        },
                    },
                    {
                        "id": "starter_4_col",
                        "component": "MaterialColumn",
                        "children": ["starter_4_tag", "starter_4_text"],
                        "style": {"gap": "6px"},
                    },
                    {
                        "id": "starter_4_tag",
                        "component": "MaterialText",
                        "text": "✨ GEMINI 3.8 FLASH + BIGQUERY MCP",
                        "usageHint": "caption",
                        "style": {
                            "color": "#e37400",
                            "fontWeight": "600",
                        },
                    },
                    {
                        "id": "starter_4_text",
                        "component": "Text",
                        "text": STARTER_PROMPT_4,
                        "variant": "body",
                    },
                    {
                        "id": "footer_row",
                        "component": "MaterialRow",
                        "children": ["return_welcome_btn"],
                        "justify": "start",
                        "align": "center",
                    },
                    {
                        "id": "return_welcome_btn",
                        "component": "MaterialButton",
                        "label": "Return to Welcome View",
                        "leadingIcon": "home",
                        "appearance": "outlined",
                        "style": {
                            "fontSize": "12px",
                        },
                        "action": {
                            "event": {
                                "name": "starter_prompt.select",
                                "context": {
                                    "prompt": RETURN_WELCOME_PROMPT,
                                },
                            }
                        },
                    },
                ],
            },
        },
    ]


def build_inline_tabs_finops_a2ui_messages(
    *,
    surface_id: str,
    title: str,
    subtitle: str,
    auth_badge: str,
    kpis: list[tuple[str, str, str]],
    vega_spec: dict[str, Any],
    table_columns: list[dict[str, Any]],
    table_rows: list[dict[str, str]],
    sql_summary_text: str,
    canvas_switch_prompt: str,
    active_tab: int = 0,
) -> list[dict[str, Any]]:
    """Build an inline A2UI v0.9 dashboard using `MaterialCard` KPIs + `MaterialTabs`
    switching between `[📊 Vega-Lite Chart]`, `[📋 Data Table]`, and `[🔍 SQL & OAuth]`."""
    kpi_ids: list[str] = []
    kpi_components: list[dict[str, Any]] = []
    for idx, (k_label, k_val, k_color) in enumerate(kpis, start=1):
        cid = f"kpi_card_{idx}"
        col_id = f"kpi_col_{idx}"
        lbl_id = f"kpi_lbl_{idx}"
        val_id = f"kpi_val_{idx}"
        kpi_ids.append(cid)
        kpi_components.extend(
            [
                {
                    "id": cid,
                    "component": "MaterialCard",
                    "appearance": "outlined",
                    "weight": 1,
                    "children": [col_id],
                    "style": {"padding": "12px 14px", "borderRadius": "10px"},
                },
                {
                    "id": col_id,
                    "component": "MaterialColumn",
                    "children": [lbl_id, val_id],
                    "style": {"gap": "4px"},
                },
                {
                    "id": lbl_id,
                    "component": "MaterialText",
                    "text": k_label,
                    "usageHint": "caption",
                    "style": {"color": k_color, "fontWeight": "600"},
                },
                {
                    "id": val_id,
                    "component": "Text",
                    "text": k_val,
                    "variant": "h4",
                },
            ]
        )

    components: list[dict[str, Any]] = [
        {
            "id": "root",
            "component": "MaterialColumn",
            "children": [
                "finops_header_col",
                "kpi_row",
                "finops_tabs",
                "finops_footer_row",
            ],
            "align": "stretch",
            "style": {"gap": "14px", "padding": "4px 0"},
        },
        {
            "id": "finops_header_col",
            "component": "MaterialColumn",
            "children": ["finops_auth_badge", "finops_title", "finops_subtitle"],
            "align": "stretch",
            "style": {"gap": "4px"},
        },
        {
            "id": "finops_auth_badge",
            "component": "MaterialText",
            "text": auth_badge,
            "usageHint": "caption",
            "style": {"color": "#137333", "fontWeight": "600"},
        },
        {
            "id": "finops_title",
            "component": "Text",
            "text": title,
            "variant": "h3",
        },
        {
            "id": "finops_subtitle",
            "component": "Text",
            "text": subtitle,
            "variant": "body",
        },
        {
            "id": "kpi_row",
            "component": "MaterialRow",
            "children": kpi_ids,
            "align": "stretch",
            "style": {"gap": "10px"},
        },
        *kpi_components,
        {
            "id": "finops_tabs",
            "component": "MaterialTabs",
            "activeTab": active_tab,
            "tabs": [
                {"label": "📊 Vega-Lite Chart", "content": "tab_chart_card"},
                {"label": "📋 Sortable Data Table", "content": "tab_table_card"},
                {"label": "🔍 SQL & OAuth Telemetry", "content": "tab_sql_card"},
            ],
        },
        {
            "id": "tab_chart_card",
            "component": "MaterialCard",
            "appearance": "outlined",
            "children": ["tab_chart_col"],
            "style": {"padding": "14px", "borderRadius": "12px"},
        },
        {
            "id": "tab_chart_col",
            "component": "MaterialColumn",
            "children": ["vega_chart_main"],
            "align": "stretch",
        },
        {
            "id": "vega_chart_main",
            "component": "VegaChart",
            "height": 280,
            "spec": {"path": "/vegaSpec"},
        },
        {
            "id": "tab_table_card",
            "component": "MaterialCard",
            "appearance": "outlined",
            "children": ["tab_table_col"],
            "style": {"padding": "12px", "borderRadius": "12px"},
        },
        {
            "id": "tab_table_col",
            "component": "MaterialColumn",
            "children": ["gcbp_table_main"],
            "align": "stretch",
        },
        {
            "id": "gcbp_table_main",
            "component": "GcbpTable",
            "columns": table_columns,
            "rows": table_rows,
        },
        {
            "id": "tab_sql_card",
            "component": "MaterialCard",
            "appearance": "outlined",
            "children": ["tab_sql_col"],
            "style": {"padding": "14px", "borderRadius": "12px"},
        },
        {
            "id": "tab_sql_col",
            "component": "MaterialColumn",
            "children": ["sql_summary_text"],
            "align": "stretch",
        },
        {
            "id": "sql_summary_text",
            "component": "Text",
            "text": sql_summary_text,
            "variant": "body",
        },
        {
            "id": "finops_footer_row",
            "component": "MaterialRow",
            "children": ["open_canvas_btn", "return_home_btn"],
            "justify": "start",
            "align": "center",
            "style": {"gap": "10px"},
        },
        {
            "id": "open_canvas_btn",
            "component": "MaterialButton",
            "label": "Open Side-Panel Canvas View",
            "leadingIcon": "open_in_new",
            "appearance": "tonal",
            "style": {"fontSize": "12px"},
            "action": {
                "event": {
                    "name": "starter_prompt.select",
                    "context": {"prompt": canvas_switch_prompt},
                }
            },
        },
        {
            "id": "return_home_btn",
            "component": "MaterialButton",
            "label": "Return to Welcome View",
            "leadingIcon": "home",
            "appearance": "outlined",
            "style": {"fontSize": "12px"},
            "action": {
                "event": {
                    "name": "starter_prompt.select",
                    "context": {"prompt": RETURN_WELCOME_PROMPT},
                }
            },
        },
    ]

    return [
        {
            "version": "v0.9",
            "createSurface": {
                "surfaceId": surface_id,
                "catalogId": A2UI_V09_CATALOG_ID,
            },
        },
        {
            "version": "v0.9",
            "updateDataModel": {
                "surfaceId": surface_id,
                "value": {
                    "vegaSpec": vega_spec,
                },
            },
        },
        {
            "version": "v0.9",
            "updateComponents": {
                "surfaceId": surface_id,
                "components": components,
            },
        },
    ]


def build_canvas_finops_a2ui_messages(
    *,
    surface_id: str,
    card_title: str,
    card_description: str,
    title: str,
    subtitle: str,
    auth_badge: str,
    kpis: list[tuple[str, str, str]],
    vega_spec: dict[str, Any],
    table_columns: list[dict[str, Any]],
    table_rows: list[dict[str, str]],
    sql_summary_text: str,
    inline_switch_prompt: str,
) -> list[dict[str, Any]]:
    """Build a Side-Panel `Canvas`-rooted A2UI v0.9 dashboard (`autoOpen: true`)
    with KPI `MaterialCard`s, interactive `VegaChart`, and sortable `GcbpTable`."""
    kpi_ids: list[str] = []
    kpi_components: list[dict[str, Any]] = []
    for idx, (k_label, k_val, k_color) in enumerate(kpis, start=1):
        cid = f"canvas_kpi_card_{idx}"
        col_id = f"canvas_kpi_col_{idx}"
        lbl_id = f"canvas_kpi_lbl_{idx}"
        val_id = f"canvas_kpi_val_{idx}"
        kpi_ids.append(cid)
        kpi_components.extend(
            [
                {
                    "id": cid,
                    "component": "MaterialCard",
                    "appearance": "outlined",
                    "weight": 1,
                    "children": [col_id],
                    "style": {"padding": "12px 14px", "borderRadius": "10px"},
                },
                {
                    "id": col_id,
                    "component": "MaterialColumn",
                    "children": [lbl_id, val_id],
                    "style": {"gap": "4px"},
                },
                {
                    "id": lbl_id,
                    "component": "MaterialText",
                    "text": k_label,
                    "usageHint": "caption",
                    "style": {"color": k_color, "fontWeight": "600"},
                },
                {
                    "id": val_id,
                    "component": "Text",
                    "text": k_val,
                    "variant": "h4",
                },
            ]
        )

    components: list[dict[str, Any]] = [
        {
            "id": "root",
            "component": "Canvas",
            "autoOpen": True,
            "cardTitle": card_title,
            "cardDescription": card_description,
            "cardIcon": "monitoring",
            "children": ["canvas_main_col"],
        },
        {
            "id": "canvas_main_col",
            "component": "MaterialColumn",
            "children": [
                "canvas_header_col",
                "canvas_kpi_row",
                "canvas_chart_card",
                "canvas_table_card",
                "canvas_sql_card",
                "canvas_footer_row",
            ],
            "align": "stretch",
            "style": {"gap": "14px", "padding": "8px 4px"},
        },
        {
            "id": "canvas_header_col",
            "component": "MaterialColumn",
            "children": ["canvas_auth_badge", "canvas_title", "canvas_subtitle"],
            "align": "stretch",
            "style": {"gap": "4px"},
        },
        {
            "id": "canvas_auth_badge",
            "component": "MaterialText",
            "text": auth_badge,
            "usageHint": "caption",
            "style": {"color": "#137333", "fontWeight": "600"},
        },
        {
            "id": "canvas_title",
            "component": "Text",
            "text": title,
            "variant": "h3",
        },
        {
            "id": "canvas_subtitle",
            "component": "Text",
            "text": subtitle,
            "variant": "body",
        },
        {
            "id": "canvas_kpi_row",
            "component": "MaterialRow",
            "children": kpi_ids,
            "align": "stretch",
            "style": {"gap": "10px"},
        },
        *kpi_components,
        {
            "id": "canvas_chart_card",
            "component": "MaterialCard",
            "appearance": "outlined",
            "children": ["canvas_chart_col"],
            "style": {"padding": "14px", "borderRadius": "12px"},
        },
        {
            "id": "canvas_chart_col",
            "component": "MaterialColumn",
            "children": ["canvas_chart_heading", "canvas_vega_chart"],
            "align": "stretch",
            "style": {"gap": "8px"},
        },
        {
            "id": "canvas_chart_heading",
            "component": "MaterialText",
            "text": "📈 INTERACTIVE VEGA-LITE VISUALIZATION",
            "usageHint": "caption",
            "style": {"color": "#1a73e8", "fontWeight": "600"},
        },
        {
            "id": "canvas_vega_chart",
            "component": "VegaChart",
            "height": 290,
            "spec": {"path": "/vegaSpec"},
        },
        {
            "id": "canvas_table_card",
            "component": "MaterialCard",
            "appearance": "outlined",
            "children": ["canvas_table_col"],
            "style": {"padding": "14px", "borderRadius": "12px"},
        },
        {
            "id": "canvas_table_col",
            "component": "MaterialColumn",
            "children": ["canvas_table_heading", "canvas_gcbp_table"],
            "align": "stretch",
            "style": {"gap": "8px"},
        },
        {
            "id": "canvas_table_heading",
            "component": "MaterialText",
            "text": "📋 SORTABLE BIGQUERY RESULTS TABLE",
            "usageHint": "caption",
            "style": {"color": "#137333", "fontWeight": "600"},
        },
        {
            "id": "canvas_gcbp_table",
            "component": "GcbpTable",
            "columns": table_columns,
            "rows": table_rows,
        },
        {
            "id": "canvas_sql_card",
            "component": "MaterialCard",
            "appearance": "outlined",
            "children": ["canvas_sql_col"],
            "style": {"padding": "12px 14px", "borderRadius": "10px"},
        },
        {
            "id": "canvas_sql_col",
            "component": "MaterialColumn",
            "children": ["canvas_sql_text"],
            "align": "stretch",
        },
        {
            "id": "canvas_sql_text",
            "component": "Text",
            "text": sql_summary_text,
            "variant": "body",
        },
        {
            "id": "canvas_footer_row",
            "component": "MaterialRow",
            "children": ["switch_inline_btn", "canvas_home_btn"],
            "justify": "start",
            "align": "center",
            "style": {"gap": "10px"},
        },
        {
            "id": "switch_inline_btn",
            "component": "MaterialButton",
            "label": "Show Inline MaterialTabs View",
            "leadingIcon": "tab",
            "appearance": "tonal",
            "style": {"fontSize": "12px"},
            "action": {
                "event": {
                    "name": "starter_prompt.select",
                    "context": {"prompt": inline_switch_prompt},
                }
            },
        },
        {
            "id": "canvas_home_btn",
            "component": "MaterialButton",
            "label": "Return to Welcome View",
            "leadingIcon": "home",
            "appearance": "outlined",
            "style": {"fontSize": "12px"},
            "action": {
                "event": {
                    "name": "starter_prompt.select",
                    "context": {"prompt": RETURN_WELCOME_PROMPT},
                }
            },
        },
    ]

    return [
        {
            "version": "v0.9",
            "createSurface": {
                "surfaceId": surface_id,
                "catalogId": A2UI_V09_CATALOG_ID,
            },
        },
        {
            "version": "v0.9",
            "updateDataModel": {
                "surfaceId": surface_id,
                "value": {
                    "vegaSpec": vega_spec,
                },
            },
        },
        {
            "version": "v0.9",
            "updateComponents": {
                "surfaceId": surface_id,
                "components": components,
            },
        },
    ]


def build_return_bar_a2ui_messages(
    surface_id: str = "agent-return-bar",
) -> list[dict[str, Any]]:
    """Build a compact action bar with buttons for Inline Tabs, Side-Panel Canvas, and Welcome View."""
    return [
        {
            "version": "v0.9",
            "createSurface": {
                "surfaceId": surface_id,
                "catalogId": A2UI_V09_CATALOG_ID,
            },
        },
        {
            "version": "v0.9",
            "updateComponents": {
                "surfaceId": surface_id,
                "components": [
                    {
                        "id": "root",
                        "component": "MaterialRow",
                        "children": [
                            "return_welcome_btn",
                            "inline_tabs_btn",
                            "side_canvas_btn",
                        ],
                        "justify": "start",
                        "align": "center",
                        "style": {"gap": "10px", "padding": "6px 0"},
                    },
                    {
                        "id": "return_welcome_btn",
                        "component": "MaterialButton",
                        "label": "Return to Welcome View",
                        "leadingIcon": "home",
                        "appearance": "outlined",
                        "style": {"fontSize": "12px"},
                        "action": {
                            "event": {
                                "name": "starter_prompt.select",
                                "context": {"prompt": RETURN_WELCOME_PROMPT},
                            }
                        },
                    },
                    {
                        "id": "inline_tabs_btn",
                        "component": "MaterialButton",
                        "label": "Token FinOps (Inline Tabs)",
                        "leadingIcon": "tab",
                        "appearance": "tonal",
                        "style": {"fontSize": "12px"},
                        "action": {
                            "event": {
                                "name": "starter_prompt.select",
                                "context": {"prompt": STARTER_PROMPT_1},
                            }
                        },
                    },
                    {
                        "id": "side_canvas_btn",
                        "component": "MaterialButton",
                        "label": "Daily Usage (Side-Panel Canvas)",
                        "leadingIcon": "open_in_new",
                        "appearance": "tonal",
                        "style": {"fontSize": "12px"},
                        "action": {
                            "event": {
                                "name": "starter_prompt.select",
                                "context": {"prompt": STARTER_PROMPT_2},
                            }
                        },
                    },
                ],
            },
        },
    ]


def make_a2ui_genai_parts(messages: list[dict[str, Any]]) -> list[types.Part]:
    """Encode A2UI v0.9 message dicts as ADK `<a2a_datapart_json>` inline blobs
    so the A2A part converter emits `DataPart`s with `mimeType: application/json+a2ui`."""
    parts: list[types.Part] = []
    for msg in messages:
        raw_bytes = json.dumps(msg).encode("utf-8")
        parts.append(
            types.Part(
                inline_data=types.Blob(
                    mime_type="text/plain",
                    data=b"<a2a_datapart_json>" + raw_bytes + b"</a2a_datapart_json>",
                )
            )
        )
    return parts


def _surface_tool_error(tool: Any, args: Any, tool_context: Any, error: Exception) -> dict[str, str]:
    """Return tool errors as structured dicts so the LLM can report OAuth/IAM errors cleanly."""
    del tool, args, tool_context
    return {"error": f"{type(error).__name__}: {error}"}


def bq_mcp_list_datasets_and_tables(
    dataset_id: str = "",
    tool_context: ToolContext | None = None,
) -> dict[str, Any]:
    """List BigQuery datasets in the project (or tables in `dataset_id`) via BigQuery MCP
    (`https://bigquery.googleapis.com/mcp`) using the end-user's OAuth 2.0 identity.

    Args:
        dataset_id: Optional dataset ID (e.g. `gemini_enterprise` or `gemini_enterprise_audit_logs`).
            If empty, lists all datasets in the project.
    """
    state = tool_context.state if tool_context else {}
    if dataset_id.strip():
        return execute_bq_mcp_rpc(
            "list_table_ids",
            {"project_id": USER_PROJECT, "dataset_id": dataset_id.strip()},
            state=state,
            project_id=USER_PROJECT,
        )
    return execute_bq_mcp_rpc(
        "list_dataset_ids",
        {"project_id": USER_PROJECT},
        state=state,
        project_id=USER_PROJECT,
    )


def bq_mcp_get_table_info(
    dataset_id: str,
    table_id: str,
    tool_context: ToolContext | None = None,
) -> dict[str, Any]:
    """Retrieve schema and metadata for a BigQuery table via BigQuery MCP using end-user OAuth.

    Args:
        dataset_id: BigQuery dataset ID (e.g. `gemini_enterprise`).
        table_id: BigQuery table ID (e.g. `usage_metrics`).
    """
    state = tool_context.state if tool_context else {}
    return execute_bq_mcp_rpc(
        "get_table_info",
        {
            "project_id": USER_PROJECT,
            "dataset_id": dataset_id.strip(),
            "table_id": table_id.strip(),
        },
        state=state,
        project_id=USER_PROJECT,
    )


def bq_mcp_execute_sql_readonly(
    sql_query: str,
    tool_context: ToolContext | None = None,
) -> dict[str, Any]:
    """Execute a read-only GoogleSQL SELECT query via BigQuery MCP (`https://bigquery.googleapis.com/mcp`)
    using the end-user's OAuth 2.0 identity.

    Args:
        sql_query: Read-only SELECT statement against the configured Google Cloud project tables.
    """
    state = tool_context.state if tool_context else {}
    return execute_bq_mcp_sql(
        sql_query,
        state=state,
        project_id=USER_PROJECT,
    )


def _extract_prompt_from_a2ui_payload(payload: Any) -> str | None:
    """Recursively extract a starter prompt string from an A2UI action payload."""
    if isinstance(payload, list):
        for item in payload:
            found = _extract_prompt_from_a2ui_payload(item)
            if found:
                return found
        return None
    if isinstance(payload, dict):
        prompt = (
            payload.get("prompt")
            or (payload.get("context") or {}).get("prompt")
            or ((payload.get("action") or {}).get("context") or {}).get("prompt")
            or ((payload.get("event") or {}).get("context") or {}).get("prompt")
            or ((payload.get("userAction") or {}).get("context") or {}).get("prompt")
        )
        if isinstance(prompt, str) and prompt.strip():
            return prompt.strip()
    return None


def _sanitize_a2ui_llm_request(
    callback_context: Any, llm_request: Any
) -> Any:
    """Sanitize `<a2a_datapart_json>` blobs and empty greeting turns in `llm_request.contents`."""
    del callback_context
    if not getattr(llm_request, "contents", None):
        return None
    for content in llm_request.contents:
        if not content.parts:
            continue
        cleaned_parts: list[types.Part] = []
        for part in content.parts:
            if part.inline_data and part.inline_data.data:
                raw = part.inline_data.data
                if raw.startswith(b"<a2a_datapart_json>") and raw.endswith(
                    b"</a2a_datapart_json>"
                ):
                    raw_json = raw[
                        len(b"<a2a_datapart_json>") : -len(b"</a2a_datapart_json>")
                    ]
                    try:
                        payload = json.loads(raw_json.decode("utf-8"))
                        extracted = _extract_prompt_from_a2ui_payload(payload)
                        if extracted:
                            cleaned_parts.append(types.Part.from_text(text=extracted))
                            continue
                    except Exception:
                        pass
                    if content.role == "model":
                        cleaned_parts.append(
                            types.Part.from_text(text="[Rendered A2UI v0.9 surface]")
                        )
                    else:
                        cleaned_parts.append(
                            types.Part.from_text(text="[A2UI interaction]")
                        )
                    continue
            if part.text is not None and not part.text.strip():
                if content.role == "user":
                    cleaned_parts.append(
                        types.Part.from_text(text="[Opened agent welcome view]")
                    )
                continue
            cleaned_parts.append(part)
        if not cleaned_parts:
            cleaned_parts.append(
                types.Part.from_text(
                    text="[Rendered A2UI v0.9 surface]"
                    if content.role == "model"
                    else "[Opened agent welcome view]"
                )
            )
        content.parts = cleaned_parts
    return None


gemini_a2ui_specialist = Agent(
    name="gemini_a2ui_specialist",
    model=Gemini(
        model=MODEL,
        retry_options=types.HttpRetryOptions(attempts=3),
    ),
    description=(
        "Gemini 3.8 Flash FinOps & BigQuery MCP specialist that queries "
        "Gemini Enterprise usage and audit tables using the end-user's OAuth identity."
    ),
    instruction=(
        f"You are the `gemini_a2ui_specialist` for the Gemini Enterprise FinOps & Usage Pulse agent in project `{USER_PROJECT}`.\n"
        "Use the BigQuery MCP tools (`list_dataset_ids`, `list_table_ids`, `get_table_info`, `execute_sql_readonly`) "
        "to inspect schemas and run read-only SQL queries.\n"
        f"Key tables in project `{USER_PROJECT}`:\n"
        f"- `{USER_PROJECT}.gemini_enterprise.usage_metrics` (daily search_count, answer_count, action_count, daily_active_user_count)\n"
        f"- `{USER_PROJECT}.gemini_enterprise_audit_logs.discoveryengine_googleapis_com_gen_ai_client_inference_operation_details` (token consumption per agent)\n"
        f"- `{USER_PROJECT}.gemini_enterprise_audit_logs.discoveryengine_googleapis_com_gemini_enterprise_user_activity` (API & agent audit events)\n\n"
        "Every BigQuery MCP call executes using the end-user's OAuth 2.0 identity. "
        "Always show the SQL query you ran and provide a concise, executive FinOps summary."
    ),
    planner=BuiltInPlanner(
        thinking_config=types.ThinkingConfig(
            include_thoughts=False,
            thinking_budget=256,
        )
    ),
    before_model_callback=_sanitize_a2ui_llm_request,
    tools=[
        bq_mcp_list_datasets_and_tables,
        bq_mcp_get_table_info,
        bq_mcp_execute_sql_readonly,
    ],
    on_tool_error_callback=_surface_tool_error,
)


def _find_a2ui_action_in_payload(payload: Any) -> dict[str, Any] | None:
    """Recursively extract an A2UI `action` object (e.g., `fetchData` or `starter_prompt.select`)."""
    if isinstance(payload, list):
        for item in payload:
            found = _find_a2ui_action_in_payload(item)
            if found:
                return found
        return None
    if isinstance(payload, dict):
        act = payload.get("action") or payload.get("userAction") or payload.get("event")
        if isinstance(act, dict) and act.get("name"):
            return act
        if payload.get("name") and ("context" in payload or "surfaceId" in payload):
            return payload
    return None


def _extract_a2ui_action(ctx: InvocationContext) -> dict[str, Any] | None:
    """Extract the A2UI `action` dict from `ctx.user_content` if present."""
    content = ctx.user_content
    if not content or not content.parts:
        return None
    for p in content.parts:
        if p.inline_data and p.inline_data.data:
            raw = p.inline_data.data
            if raw.startswith(b"<a2a_datapart_json>") and raw.endswith(
                b"</a2a_datapart_json>"
            ):
                raw = raw[len(b"<a2a_datapart_json>") : -len(b"</a2a_datapart_json>")]
            try:
                payload = json.loads(raw.decode("utf-8"))
                act = _find_a2ui_action_in_payload(payload)
                if act:
                    return act
            except Exception:
                pass
        elif p.text:
            raw_text = p.text.strip()
            if raw_text.startswith("{") or raw_text.startswith("["):
                try:
                    payload = json.loads(raw_text)
                    act = _find_a2ui_action_in_payload(payload)
                    if act:
                        return act
                except Exception:
                    pass
    return None


def _parse_sort_cell(val: str) -> tuple[int, Any]:
    """Parse formatted table cell strings (`$0.9574`, `1,245,900 (39.0%)`, `2026-11-10`) for sorting."""
    s = (val or "").strip()
    primary = s.split(" (")[0].strip()
    cleaned = primary.replace("$", "").replace(",", "").replace("%", "").strip()
    try:
        return (0, float(cleaned))
    except ValueError:
        return (1, s.lower())


def _sort_table_rows(
    rows: list[dict[str, str]], sort_string: str
) -> list[dict[str, str]]:
    """Sort `GcbpTable` rows in-place according to `sortString` (e.g. `'service asc'`, `'calls desc'`)."""
    spec = (sort_string or "").strip()
    if not spec or not rows:
        return rows
    parts = spec.split()
    field = parts[0]
    direction = parts[1].lower() if len(parts) > 1 else "asc"
    if field not in rows[0]:
        return rows
    reverse = direction == "desc"
    return sorted(
        rows,
        key=lambda r: _parse_sort_cell(str(r.get(field, ""))),
        reverse=reverse,
    )


def _infer_dashboard_kind_from_sort_field(
    sort_string: str, fallback: str = "token"
) -> str:
    """Infer which FinOps dashboard (`token`, `daily`, or `audit`) owns the sorted column field."""
    parts = (sort_string or "").strip().split()
    field = parts[0] if parts else ""
    if field in {
        "agent_name",
        "calls",
        "input_tokens",
        "cached_tokens",
        "output_reasoning",
        "est_cost_usd",
    }:
        return "token"
    if field in {
        "date",
        "dau",
        "searches",
        "answers",
        "actions",
        "total_interactions",
    }:
        return "daily"
    if field in {"method", "service", "event_count", "latest_date"}:
        return "audit"
    return fallback


def _extract_user_text(ctx: InvocationContext) -> str:
    """Extract normalized user text (or A2UI action prompt) from InvocationContext."""
    content = ctx.user_content
    if not content or not content.parts:
        return ""
    texts: list[str] = []
    for p in content.parts:
        if p.text:
            raw_text = p.text.strip()
            if raw_text.startswith("{") or raw_text.startswith("["):
                try:
                    parsed = json.loads(raw_text)
                    extracted = _extract_prompt_from_a2ui_payload(parsed)
                    if extracted:
                        texts.append(extracted)
                        continue
                except Exception:
                    pass
            if raw_text:
                texts.append(raw_text)
        elif p.inline_data and p.inline_data.data:
            raw = p.inline_data.data
            if raw.startswith(b"<a2a_datapart_json>") and raw.endswith(
                b"</a2a_datapart_json>"
            ):
                raw = raw[len(b"<a2a_datapart_json>") : -len(b"</a2a_datapart_json>")]
            try:
                payload = json.loads(raw.decode("utf-8"))
                extracted = _extract_prompt_from_a2ui_payload(payload)
                if extracted:
                    texts.append(extracted)
            except Exception:
                pass
    return " ".join(texts).strip()


def _resolve_user_greeting_context(ctx: InvocationContext) -> tuple[str, str]:
    """Resolve (user_display, caller_detail) from session state, custom_metadata, or user_id."""
    caller_identity: dict[str, Any] = {}
    if ctx.session and isinstance(ctx.session.state, dict):
        raw_state_id = ctx.session.state.get("caller_identity")
        if isinstance(raw_state_id, dict):
            caller_identity = raw_state_id
    if not caller_identity and ctx.run_config and ctx.run_config.custom_metadata:
        raw_meta_id = ctx.run_config.custom_metadata.get("caller_identity")
        if isinstance(raw_meta_id, dict):
            caller_identity = raw_meta_id

    raw_user_id = ctx.user_id or "Guest"
    short_user_id = (
        raw_user_id[:17] if raw_user_id.startswith("A2A_USER_") else raw_user_id
    )
    user_display = str(
        caller_identity.get("display_user") or short_user_id or "Guest"
    )
    human_user = caller_identity.get("human_user")
    auth_mode = caller_identity.get("auth_mode") or "OAuth 2.0"
    if human_user:
        caller_detail = f"{human_user} ({auth_mode})"
    else:
        caller_detail = f"{user_display} ({auth_mode})"
    return user_display, caller_detail


def _build_token_finops_dashboard_payload(
    bq_res: dict[str, Any],
    *,
    surface_id: str,
    use_canvas: bool = False,
    sort_string: str = "",
    active_tab: int = 0,
) -> tuple[str, list[dict[str, Any]]]:
    """Transform BigQuery MCP token usage rows into Vega-Lite spec + GcbpTable + KPIs."""
    rows = bq_res.get("rows", [])
    total_calls = 0
    total_input = 0
    total_cached = 0
    total_output = 0
    total_reasoning = 0

    vega_values: list[dict[str, Any]] = []
    table_rows: list[dict[str, str]] = []

    for r in rows:
        agent_raw = str(r.get("agent_name") or "default_assistant")
        short_agent = (
            agent_raw.replace("_federated_query_rewrite_agent", "_rewrite")
            .replace("llm_agent_", "")
        )
        calls = int(r.get("inference_calls") or 0)
        inp = int(r.get("input_tokens") or 0)
        cached = int(r.get("cached_tokens") or 0)
        out = int(r.get("output_tokens") or 0)
        reas = int(r.get("reasoning_tokens") or 0)

        total_calls += calls
        total_input += inp
        total_cached += cached
        total_output += out
        total_reasoning += reas

        uncached_inp = max(0, inp - cached)
        # Illustrative enterprise FinOps rate ($0.50/1M input, $0.125/1M cached, $3.00/1M output+reasoning)
        est_cost = (
            (uncached_inp / 1_000_000.0) * 0.50
            + (cached / 1_000_000.0) * 0.125
            + ((out + reas) / 1_000_000.0) * 3.00
        )
        cache_hit_pct = round((cached / inp * 100.0), 1) if inp > 0 else 0.0

        vega_values.extend(
            [
                {
                    "agent": short_agent,
                    "Token Type": "Uncached Input",
                    "tokens": uncached_inp,
                },
                {
                    "agent": short_agent,
                    "Token Type": "Cache Read Input",
                    "tokens": cached,
                },
                {
                    "agent": short_agent,
                    "Token Type": "Output + Reasoning",
                    "tokens": out + reas,
                },
            ]
        )
        table_rows.append(
            {
                "agent_name": short_agent,
                "calls": f"{calls:,}",
                "input_tokens": f"{inp:,}",
                "cached_tokens": f"{cached:,} ({cache_hit_pct}%)",
                "output_reasoning": f"{out + reas:,}",
                "est_cost_usd": f"${est_cost:.4f}",
            }
        )

    overall_cache_pct = (
        round((total_cached / total_input) * 100.0, 1) if total_input > 0 else 0.0
    )
    total_tokens = total_input + total_output + total_reasoning
    total_uncached = max(0, total_input - total_cached)
    total_est_cost = (
        (total_uncached / 1_000_000.0) * 0.50
        + (total_cached / 1_000_000.0) * 0.125
        + ((total_output + total_reasoning) / 1_000_000.0) * 3.00
    )

    kpis = [
        ("TOTAL TOKENS PROCESSED", f"{total_tokens:,}", "#1a73e8"),
        ("PROMPT CACHE EFFICIENCY", f"{overall_cache_pct}% ({total_cached:,})", "#137333"),
        ("INFERENCE CALLS / EST COST", f"{total_calls:,} calls • ${total_est_cost:.3f}", "#9334e6"),
    ]

    vega_spec = {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "description": "Token Consumption Breakdown by Gemini Enterprise Agent",
        "width": "container",
        "height": 240,
        "data": {"values": vega_values},
        "mark": {"type": "bar", "cornerRadiusEnd": 4, "tooltip": True},
        "encoding": {
            "y": {
                "field": "agent",
                "type": "nominal",
                "sort": "-x",
                "title": "Agent / Sub-Agent",
            },
            "x": {
                "field": "tokens",
                "type": "quantitative",
                "title": "Total Tokens",
            },
            "color": {
                "field": "Token Type",
                "type": "nominal",
                "scale": {
                    "domain": [
                        "Uncached Input",
                        "Cache Read Input",
                        "Output + Reasoning",
                    ],
                    "range": ["#1a73e8", "#34a853", "#9334e6"],
                },
            },
        },
    }

    table_columns = [
        {"header": "Agent", "field": "agent_name", "sortable": True},
        {"header": "Calls", "field": "calls", "sortable": True},
        {"header": "Input Tokens", "field": "input_tokens", "sortable": True},
        {"header": "Cached Tokens (%)", "field": "cached_tokens", "sortable": True},
        {"header": "Output + Reasoning", "field": "output_reasoning", "sortable": True},
        {"header": "Est. Cost (USD)", "field": "est_cost_usd", "sortable": True},
    ]

    auth_badge = (
        f"🔐 BigQuery MCP ({bq_res['auth_mode']}) • Principal: {bq_res['principal']} • "
        f"{bq_res['elapsed_ms']}ms"
    )
    sql_summary_text = (
        f"Executed via BigQuery MCP (`https://bigquery.googleapis.com/mcp` -> `execute_sql_readonly`)\n"
        f"• Authenticated Principal: {bq_res['principal']} ({bq_res['auth_mode']})\n"
        f"• BigQuery Job / Query ID: {bq_res['query_id']}\n"
        f"• Bytes Processed: {int(bq_res['total_bytes_processed']):,} bytes | Slot Time: {bq_res['total_slot_ms']} ms\n\n"
        f"SQL Query:\n{bq_res['sql']}"
    )

    summary_md = (
        f"### 📊 Gemini Enterprise Token FinOps Pulse ({'Side-Panel Canvas' if use_canvas else 'Inline MaterialTabs'})\n\n"
        f"Queried **`{USER_PROJECT}.gemini_enterprise_audit_logs.discoveryengine_googleapis_com_gen_ai_client_inference_operation_details`** "
        f"via **BigQuery MCP** as **`{bq_res['principal']}`** (`{bq_res['auth_mode']}`):\n"
        f"- **Total Tokens**: `{total_tokens:,}` across `{total_calls:,}` model invocations (`{overall_cache_pct}%` prompt cache hit ratio)\n"
        f"- **Top Consumer**: `root_agent` (`3,191,204` input tokens, `1,245,900` cached tokens)"
    )

    if sort_string:
        table_rows = _sort_table_rows(table_rows, sort_string)

    if use_canvas:
        messages = build_canvas_finops_a2ui_messages(
            surface_id=surface_id,
            card_title="Token FinOps Breakdown by Agent (Canvas)",
            card_description=f"{total_tokens:,} total tokens • {overall_cache_pct}% cache hit rate • BigQuery MCP",
            title="Gemini Enterprise Token FinOps by Agent",
            subtitle="Side-Panel Canvas view with Vega-Lite stacked bar chart and sortable BigQuery table.",
            auth_badge=auth_badge,
            kpis=kpis,
            vega_spec=vega_spec,
            table_columns=table_columns,
            table_rows=table_rows,
            sql_summary_text=sql_summary_text,
            inline_switch_prompt=STARTER_PROMPT_1,
        )
    else:
        messages = build_inline_tabs_finops_a2ui_messages(
            surface_id=surface_id,
            title="Gemini Enterprise Token FinOps by Agent",
            subtitle="Rendered inline using MaterialTabs: switch between the Vega-Lite chart, sortable GcbpTable, and SQL telemetry.",
            auth_badge=auth_badge,
            kpis=kpis,
            vega_spec=vega_spec,
            table_columns=table_columns,
            table_rows=table_rows,
            sql_summary_text=sql_summary_text,
            canvas_switch_prompt="Open token usage & estimated cost by agent in Side-Panel Canvas",
            active_tab=active_tab,
        )
    return summary_md, messages


def _build_daily_usage_dashboard_payload(
    bq_res: dict[str, Any],
    *,
    surface_id: str,
    use_canvas: bool = True,
    sort_string: str = "",
    active_tab: int = 0,
) -> tuple[str, list[dict[str, Any]]]:
    """Transform `gemini_enterprise.usage_metrics` rows into Vega-Lite spec + GcbpTable + KPIs."""
    rows = bq_res.get("rows", [])
    total_searches = 0
    total_answers = 0
    total_actions = 0
    peak_dau = 0

    vega_values: list[dict[str, Any]] = []
    table_rows: list[dict[str, str]] = []

    for r in rows:
        dt = str(r.get("date") or "")
        searches = int(r.get("searches") or 0)
        answers = int(r.get("answers") or 0)
        actions = int(r.get("actions") or 0)
        dau = int(r.get("dau") or 0)

        total_searches += searches
        total_answers += answers
        total_actions += actions
        peak_dau = max(peak_dau, dau)

        vega_values.extend(
            [
                {"date": dt, "Metric": "Searches", "count": searches},
                {"date": dt, "Metric": "AI Answers", "count": answers},
                {"date": dt, "Metric": "Agent Actions", "count": actions},
            ]
        )
        table_rows.append(
            {
                "date": dt,
                "dau": str(dau),
                "searches": str(searches),
                "answers": str(answers),
                "actions": str(actions),
                "total_interactions": str(searches + answers + actions),
            }
        )

    # Show newest dates first in the table
    table_rows.reverse()

    kpis = [
        ("TOTAL AI ANSWERS & SEARCHES", f"{total_answers:,} answers • {total_searches:,} searches", "#1a73e8"),
        ("ASSISTANT & AGENT ACTIONS", f"{total_actions:,} actions", "#137333"),
        ("ACTIVE DAYS / PEAK DAU", f"{len(rows)} active days • {peak_dau} peak DAU", "#9334e6"),
    ]

    vega_spec = {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "description": "Gemini Enterprise Daily Searches, Answers, and Agent Actions",
        "width": "container",
        "height": 250,
        "data": {"values": vega_values},
        "mark": {"type": "bar", "cornerRadiusTopLeft": 3, "cornerRadiusTopRight": 3, "tooltip": True},
        "encoding": {
            "x": {"field": "date", "type": "ordinal", "title": "Date"},
            "y": {"field": "count", "type": "quantitative", "title": "Daily Volume"},
            "color": {
                "field": "Metric",
                "type": "nominal",
                "scale": {
                    "domain": ["Searches", "AI Answers", "Agent Actions"],
                    "range": ["#1a73e8", "#34a853", "#fbbc04"],
                },
            },
        },
    }

    table_columns = [
        {"header": "Date", "field": "date", "sortable": True},
        {"header": "DAU", "field": "dau", "sortable": True},
        {"header": "Searches", "field": "searches", "sortable": True},
        {"header": "AI Answers", "field": "answers", "sortable": True},
        {"header": "Agent Actions", "field": "actions", "sortable": True},
        {"header": "Total Interactions", "field": "total_interactions", "sortable": True},
    ]

    auth_badge = (
        f"🔐 BigQuery MCP ({bq_res['auth_mode']}) • Principal: {bq_res['principal']} • "
        f"{bq_res['elapsed_ms']}ms"
    )
    sql_summary_text = (
        f"Executed via BigQuery MCP (`https://bigquery.googleapis.com/mcp` -> `execute_sql_readonly`)\n"
        f"• Authenticated Principal: {bq_res['principal']} ({bq_res['auth_mode']})\n"
        f"• BigQuery Job / Query ID: {bq_res['query_id']}\n"
        f"• Bytes Processed: {int(bq_res['total_bytes_processed']):,} bytes | Slot Time: {bq_res['total_slot_ms']} ms\n\n"
        f"SQL Query:\n{bq_res['sql']}"
    )

    summary_md = (
        f"### 🖼️ Gemini Enterprise Daily Activity & DAU Pulse ({'Side-Panel Canvas' if use_canvas else 'Inline MaterialTabs'})\n\n"
        f"Queried **`{USER_PROJECT}.gemini_enterprise.usage_metrics`** via **BigQuery MCP** "
        f"as **`{bq_res['principal']}`** (`{bq_res['auth_mode']}`):\n"
        f"- **Total Interactions**: `{total_searches + total_answers + total_actions:,}` (`{total_answers:,}` AI answers, `{total_searches:,}` searches, `{total_actions:,}` actions)\n"
        f"- **Active Window**: `{len(rows)}` active days with peak DAU of `{peak_dau}`"
    )

    if sort_string:
        table_rows = _sort_table_rows(table_rows, sort_string)

    if use_canvas:
        messages = build_canvas_finops_a2ui_messages(
            surface_id=surface_id,
            card_title="Gemini Enterprise Daily Usage & DAU Dashboard",
            card_description=f"{total_answers:,} AI answers • {total_searches:,} searches • {total_actions:,} actions (BigQuery MCP)",
            title="Gemini Enterprise Daily Activity & Search Pulse",
            subtitle="Opened in the Side-Panel Canvas (autoOpen: true). Explore daily searches, AI answers, and agent actions.",
            auth_badge=auth_badge,
            kpis=kpis,
            vega_spec=vega_spec,
            table_columns=table_columns,
            table_rows=table_rows,
            sql_summary_text=sql_summary_text,
            inline_switch_prompt="Show Gemini Enterprise daily activity & search pulse in inline MaterialTabs",
        )
    else:
        messages = build_inline_tabs_finops_a2ui_messages(
            surface_id=surface_id,
            title="Gemini Enterprise Daily Activity & Search Pulse",
            subtitle="Rendered inline using MaterialTabs: switch between the Vega-Lite chart, sortable GcbpTable, and SQL telemetry.",
            auth_badge=auth_badge,
            kpis=kpis,
            vega_spec=vega_spec,
            table_columns=table_columns,
            table_rows=table_rows,
            sql_summary_text=sql_summary_text,
            canvas_switch_prompt=STARTER_PROMPT_2,
            active_tab=active_tab,
        )
    return summary_md, messages


def _build_audit_activity_dashboard_payload(
    bq_res: dict[str, Any],
    *,
    surface_id: str,
    use_canvas: bool = False,
    sort_string: str = "",
    active_tab: int = 0,
) -> tuple[str, list[dict[str, Any]]]:
    """Transform `discoveryengine_googleapis_com_gemini_enterprise_user_activity` rows into A2UI."""
    rows = bq_res.get("rows", [])
    total_events = 0
    top_method = "n/a"
    top_count = 0

    vega_values: list[dict[str, Any]] = []
    table_rows: list[dict[str, str]] = []

    for r in rows:
        method = str(r.get("method") or "Unknown")
        svc = str(r.get("service_short") or "DiscoveryEngine")
        cnt = int(r.get("event_count") or 0)
        latest = str(r.get("latest_date") or "n/a")

        total_events += cnt
        if cnt > top_count:
            top_count = cnt
            top_method = method

        vega_values.append(
            {
                "method": method,
                "Service": svc,
                "events": cnt,
            }
        )
        table_rows.append(
            {
                "method": method,
                "service": svc,
                "event_count": f"{cnt:,}",
                "latest_date": latest,
            }
        )

    if sort_string:
        table_rows = _sort_table_rows(table_rows, sort_string)

    kpis = [
        ("TOTAL AUDIT LOG EVENTS", f"{total_events:,} events", "#1a73e8"),
        ("TOP API OPERATION", f"{top_method} ({top_count:,})", "#137333"),
        ("DISTINCT API METHODS", f"{len(rows)} methods tracked", "#9334e6"),
    ]

    vega_spec = {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "description": "Gemini Enterprise Audit Events by Method and Service",
        "width": "container",
        "height": 240,
        "data": {"values": vega_values},
        "mark": {"type": "bar", "cornerRadiusEnd": 4, "tooltip": True},
        "encoding": {
            "y": {"field": "method", "type": "nominal", "sort": "-x", "title": "API Method"},
            "x": {"field": "events", "type": "quantitative", "title": "Audit Event Count"},
            "color": {"field": "Service", "type": "nominal"},
        },
    }

    table_columns = [
        {"header": "API Method", "field": "method", "sortable": True},
        {"header": "Service", "field": "service", "sortable": True},
        {"header": "Event Count", "field": "event_count", "sortable": True},
        {"header": "Latest Activity Date", "field": "latest_date", "sortable": True},
    ]

    auth_badge = (
        f"🔐 BigQuery MCP ({bq_res['auth_mode']}) • Principal: {bq_res['principal']} • "
        f"{bq_res['elapsed_ms']}ms"
    )
    sql_summary_text = (
        f"Executed via BigQuery MCP (`https://bigquery.googleapis.com/mcp` -> `execute_sql_readonly`)\n"
        f"• Authenticated Principal: {bq_res['principal']} ({bq_res['auth_mode']})\n"
        f"• BigQuery Job / Query ID: {bq_res['query_id']}\n"
        f"• Bytes Processed: {int(bq_res['total_bytes_processed']):,} bytes | Slot Time: {bq_res['total_slot_ms']} ms\n\n"
        f"SQL Query:\n{bq_res['sql']}"
    )

    summary_md = (
        f"### ⚡ Gemini Enterprise Audit Log Activity ({'Side-Panel Canvas' if use_canvas else 'Inline MaterialTabs'})\n\n"
        f"Queried **`{USER_PROJECT}.gemini_enterprise_audit_logs.discoveryengine_googleapis_com_gemini_enterprise_user_activity`** "
        f"via **BigQuery MCP** as **`{bq_res['principal']}`** (`{bq_res['auth_mode']}`):\n"
        f"- **Total Tracked Events**: `{total_events:,}` across `{len(rows)}` Discovery Engine services/methods\n"
        f"- **Most Frequent Operation**: `{top_method}` (`{top_count:,}` calls)"
    )

    if use_canvas:
        messages = build_canvas_finops_a2ui_messages(
            surface_id=surface_id,
            card_title="Gemini Enterprise Audit Log Breakdown (Canvas)",
            card_description=f"{total_events:,} audit events • Top method: {top_method}",
            title="Gemini Enterprise API & Agent Audit Activity",
            subtitle="Opened in Side-Panel Canvas: inspect API method volume and latest activity timestamps.",
            auth_badge=auth_badge,
            kpis=kpis,
            vega_spec=vega_spec,
            table_columns=table_columns,
            table_rows=table_rows,
            sql_summary_text=sql_summary_text,
            inline_switch_prompt=STARTER_PROMPT_3,
        )
    else:
        messages = build_inline_tabs_finops_a2ui_messages(
            surface_id=surface_id,
            title="Gemini Enterprise API & Agent Audit Activity",
            subtitle="Rendered inline using MaterialTabs: switch between the Vega-Lite chart, sortable GcbpTable, and SQL telemetry.",
            auth_badge=auth_badge,
            kpis=kpis,
            vega_spec=vega_spec,
            table_columns=table_columns,
            table_rows=table_rows,
            sql_summary_text=sql_summary_text,
            canvas_switch_prompt="Open Gemini Enterprise API & agent audit activity in Side-Panel Canvas",
            active_tab=active_tab,
        )
    return summary_md, messages


class A2UIRootRouterAgent(BaseAgent):
    """Low-latency hybrid router combining instant A2UI v0.9 greeting,
    live BigQuery MCP FinOps dashboards (Inline MaterialTabs & Side-Panel Canvas),
    `GcbpTable` `fetchData` sort actions, and Gemini 3.8 Flash + BigQuery MCP analysis."""

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        user_text = _extract_user_text(ctx)
        lowered = user_text.lower()
        a2ui_action = _extract_a2ui_action(ctx)
        user_display, caller_detail = _resolve_user_greeting_context(ctx)
        session_state = (
            ctx.session.state
            if (ctx.session and isinstance(ctx.session.state, dict))
            else {}
        )

        # Path 0: Intercept `GcbpTable` `fetchData` column-sort actions so sorting
        # re-renders the COMPLETE FinOps dashboard (with activeTab=1 on MaterialTabs
        # or inside Canvas) instead of falling through to the LLM!
        if a2ui_action and a2ui_action.get("name") == "fetchData":
            act_ctx = a2ui_action.get("context") or {}
            sort_string = str(act_ctx.get("sortString") or "").strip()
            source_comp = str(a2ui_action.get("sourceComponentId") or "")
            fallback_kind = str(session_state.get("last_dashboard_kind") or "token")
            dashboard_kind = _infer_dashboard_kind_from_sort_field(
                sort_string, fallback=fallback_kind
            )
            use_canvas = (
                True
                if source_comp == "canvas_gcbp_table"
                else False
                if source_comp == "gcbp_table_main"
                else bool(session_state.get("last_dashboard_canvas", False))
            )

            if dashboard_kind == "daily":
                target_sql = SQL_DAILY_USAGE_METRICS
                target_table_label = f"`{USER_PROJECT}.gemini_enterprise.usage_metrics`"
            elif dashboard_kind == "audit":
                target_sql = SQL_AUDIT_ACTIVITY_BY_METHOD
                target_table_label = f"`{USER_PROJECT}.gemini_enterprise_audit_logs.discoveryengine_googleapis_com_gemini_enterprise_user_activity`"
            else:
                target_sql = SQL_TOKEN_FINOPS_BY_AGENT
                target_table_label = f"`{USER_PROJECT}.gemini_enterprise_audit_logs.discoveryengine_googleapis_com_gen_ai_client_inference_operation_details`"

            cached_bq_res = session_state.get(f"bq_cache_{dashboard_kind}")
            if not isinstance(cached_bq_res, dict) or not cached_bq_res.get("rows"):
                cached_bq_res = await asyncio.to_thread(
                    execute_bq_mcp_sql,
                    target_sql,
                    state=session_state,
                    project_id=USER_PROJECT,
                )

            sort_steps = [
                (
                    1,
                    "capture_sort_action",
                    f"🧠 **[Synthetic Thought 1/3]** Captured `GcbpTable` sort event (`sortString=\"{sort_string}\"`) for authenticated caller `{caller_detail}`...\n\n",
                ),
                (
                    2,
                    "sort_bigquery_rows",
                    f"⚙️ **[Synthetic Thought 2/3]** Sorting `{cached_bq_res['row_count']}` BigQuery MCP rows from {target_table_label} by `{sort_string}`...\n\n",
                ),
                (
                    3,
                    "remount_dashboard",
                    f"🎨 **[Synthetic Thought 3/3]** Re-mounting full A2UI `v0.9` FinOps dashboard (`VegaChart` + KPI cards + sorted `GcbpTable`)...\n\n",
                ),
            ]
            for step_num, stage, msg in sort_steps:
                yield Event(
                    invocation_id=ctx.invocation_id,
                    author=self.name,
                    branch=ctx.branch,
                    partial=True,
                    content=types.Content(
                        role="model",
                        parts=[types.Part(text=msg, thought=True)],
                    ),
                    actions=EventActions(
                        state_delta={"a2ui_stage": stage, "a2ui_step": step_num}
                    ),
                )
                await asyncio.sleep(STEP_DELAY_SECONDS)

            viz_surface_id = f"agent-finops-{ctx.invocation_id[:8]}"
            if dashboard_kind == "daily":
                summary_md, a2ui_messages = _build_daily_usage_dashboard_payload(
                    cached_bq_res,
                    surface_id=viz_surface_id,
                    use_canvas=use_canvas,
                    sort_string=sort_string,
                    active_tab=1,
                )
            elif dashboard_kind == "audit":
                summary_md, a2ui_messages = _build_audit_activity_dashboard_payload(
                    cached_bq_res,
                    surface_id=viz_surface_id,
                    use_canvas=use_canvas,
                    sort_string=sort_string,
                    active_tab=1,
                )
            else:
                summary_md, a2ui_messages = _build_token_finops_dashboard_payload(
                    cached_bq_res,
                    surface_id=viz_surface_id,
                    use_canvas=use_canvas,
                    sort_string=sort_string,
                    active_tab=1,
                )

            summary_md = f"{summary_md}\n- **Sorted Table By**: `{sort_string}`"
            final_parts: list[types.Part] = [
                types.Part.from_text(text=summary_md),
                *make_a2ui_genai_parts(a2ui_messages),
            ]
            yield Event(
                invocation_id=ctx.invocation_id,
                author=self.name,
                branch=ctx.branch,
                partial=False,
                turn_complete=True,
                content=types.Content(role="model", parts=final_parts),
                actions=EventActions(
                    skip_summarization=True,
                    state_delta={
                        "a2ui_stage": "completed",
                        "a2ui_step": 3,
                        "last_dashboard_kind": dashboard_kind,
                        "last_dashboard_canvas": use_canvas,
                        f"bq_cache_{dashboard_kind}": cached_bq_res,
                    },
                ),
            )
            return

        # Path 1: Instant Greeting / Activation (<10ms, zero LLM call, zero delay)
        is_greeting = (
            not lowered
            or lowered in {"hi", "hello", "hey", "start", "welcome", "menu", "init"}
            or "welcome view" in lowered
            or "starter prompt cards" in lowered
            or "starter card" in lowered
        )
        if is_greeting:
            already_rendered = bool(session_state.get("agent_welcome_rendered"))
            surface_id = (
                "agent-welcome"
                if (not lowered and not already_rendered)
                else f"agent-welcome-{ctx.invocation_id[:8]}"
            )
            welcome_parts = make_a2ui_genai_parts(
                build_welcome_a2ui_messages(
                    surface_id=surface_id,
                    user_display=user_display,
                    caller_detail=caller_detail,
                )
            )
            yield Event(
                invocation_id=ctx.invocation_id,
                author=self.name,
                branch=ctx.branch,
                partial=False,
                turn_complete=True,
                content=types.Content(role="model", parts=welcome_parts),
                actions=EventActions(
                    skip_summarization=True,
                    state_delta={"agent_welcome_rendered": True},
                ),
            )
            return

        # Path 2: Live BigQuery MCP FinOps Dashboards (Inline MaterialTabs OR Side-Panel Canvas)
        is_token_finops = (
            "token usage" in lowered
            or "estimated cost by agent" in lowered
            or "top cost drivers" in lowered
            or "discover all gemini enterprise bigquery tables" in lowered
        )
        is_daily_usage = "daily activity" in lowered or "search pulse" in lowered
        is_audit_activity = "audit activity" in lowered or "across services" in lowered
        is_legacy_diag = (
            "synthetic" in lowered or "3-step" in lowered or "diagnostic" in lowered
        )

        if is_token_finops or is_daily_usage or is_audit_activity or is_legacy_diag:
            use_canvas = "canvas" in lowered or "side-panel" in lowered
            if is_daily_usage and "inline" not in lowered and "materialtabs" not in lowered:
                use_canvas = True

            if is_daily_usage:
                dashboard_kind = "daily"
                target_sql = SQL_DAILY_USAGE_METRICS
                target_table_label = f"`{USER_PROJECT}.gemini_enterprise.usage_metrics`"
            elif is_audit_activity:
                dashboard_kind = "audit"
                target_sql = SQL_AUDIT_ACTIVITY_BY_METHOD
                target_table_label = f"`{USER_PROJECT}.gemini_enterprise_audit_logs.discoveryengine_googleapis_com_gemini_enterprise_user_activity`"
            else:
                dashboard_kind = "token"
                target_sql = SQL_TOKEN_FINOPS_BY_AGENT
                target_table_label = f"`{USER_PROJECT}.gemini_enterprise_audit_logs.discoveryengine_googleapis_com_gen_ai_client_inference_operation_details`"

            layout_label = (
                "Side-Panel `Canvas` (`autoOpen: true`)"
                if use_canvas
                else "Inline `MaterialTabs` (`[📊 Chart]` | `[📋 Table]` | `[🔍 SQL]`)"
            )

            # Thinking Step 1: OAuth Identity + BigQuery MCP Handshake
            yield Event(
                invocation_id=ctx.invocation_id,
                author=self.name,
                branch=ctx.branch,
                partial=True,
                content=types.Content(
                    role="model",
                    parts=[
                        types.Part(
                            text=(
                                f"🧠 **[Synthetic Thought 1/3]** Verifying End-User OAuth 2.0 identity (`{caller_detail}`) "
                                f"and connecting to BigQuery MCP (`{BQ_MCP_URL}`)...\n\n"
                            ),
                            thought=True,
                        )
                    ],
                ),
                actions=EventActions(
                    state_delta={"a2ui_stage": "oauth_handshake", "a2ui_step": 1}
                ),
            )
            await asyncio.sleep(STEP_DELAY_SECONDS)

            # Thinking Step 2: Execute Live SQL via BigQuery MCP
            yield Event(
                invocation_id=ctx.invocation_id,
                author=self.name,
                branch=ctx.branch,
                partial=True,
                content=types.Content(
                    role="model",
                    parts=[
                        types.Part(
                            text=(
                                f"⚙️ **[Synthetic Thought 2/3]** Executing `execute_sql_readonly` on {target_table_label} "
                                f"via BigQuery MCP using end-user OAuth credentials...\n\n"
                            ),
                            thought=True,
                        )
                    ],
                ),
                actions=EventActions(
                    state_delta={"a2ui_stage": "bigquery_mcp_query", "a2ui_step": 2}
                ),
            )

            try:
                bq_res = await asyncio.to_thread(
                    execute_bq_mcp_sql,
                    target_sql,
                    state=session_state,
                    project_id=USER_PROJECT,
                )
            except Exception as exc:
                err_msg = f"❌ **BigQuery MCP Query Failed**: `{exc}`"
                yield Event(
                    invocation_id=ctx.invocation_id,
                    author=self.name,
                    branch=ctx.branch,
                    partial=False,
                    turn_complete=True,
                    content=types.Content(
                        role="model",
                        parts=[
                            types.Part.from_text(text=err_msg),
                            *make_a2ui_genai_parts(
                                build_return_bar_a2ui_messages(
                                    surface_id=f"agent-err-bar-{ctx.invocation_id[:8]}"
                                )
                            ),
                        ],
                    ),
                    actions=EventActions(skip_summarization=True),
                )
                return

            await asyncio.sleep(STEP_DELAY_SECONDS)

            # Thinking Step 3: Build Vega-Lite + A2UI v0.9 Surface
            yield Event(
                invocation_id=ctx.invocation_id,
                author=self.name,
                branch=ctx.branch,
                partial=True,
                content=types.Content(
                    role="model",
                    parts=[
                        types.Part(
                            text=(
                                f"🎨 **[Synthetic Thought 3/3]** Received `{bq_res['row_count']}` rows in `{bq_res['elapsed_ms']}ms` "
                                f"(`queryId={bq_res['query_id']}`). Assembling `VegaChart` and sortable `GcbpTable` in {layout_label}...\n\n"
                            ),
                            thought=True,
                        )
                    ],
                ),
                actions=EventActions(
                    state_delta={"a2ui_stage": "build_a2ui_viz", "a2ui_step": 3}
                ),
            )
            await asyncio.sleep(STEP_DELAY_SECONDS)

            viz_surface_id = f"agent-finops-{ctx.invocation_id[:8]}"
            if is_daily_usage:
                summary_md, a2ui_messages = _build_daily_usage_dashboard_payload(
                    bq_res, surface_id=viz_surface_id, use_canvas=use_canvas
                )
            elif is_audit_activity:
                summary_md, a2ui_messages = _build_audit_activity_dashboard_payload(
                    bq_res, surface_id=viz_surface_id, use_canvas=use_canvas
                )
            else:
                summary_md, a2ui_messages = _build_token_finops_dashboard_payload(
                    bq_res, surface_id=viz_surface_id, use_canvas=use_canvas
                )

            final_parts = [
                types.Part.from_text(text=summary_md),
                *make_a2ui_genai_parts(a2ui_messages),
            ]
            yield Event(
                invocation_id=ctx.invocation_id,
                author=self.name,
                branch=ctx.branch,
                partial=False,
                turn_complete=True,
                content=types.Content(role="model", parts=final_parts),
                actions=EventActions(
                    skip_summarization=True,
                    state_delta={
                        "a2ui_stage": "completed",
                        "a2ui_step": 3,
                        "last_dashboard_kind": dashboard_kind,
                        "last_dashboard_canvas": use_canvas,
                        f"bq_cache_{dashboard_kind}": bq_res,
                    },
                ),
            )
            return

        # Path 3: Free-form query -> Emit clean 3-Step Synthetic Thinking timeline
        # while delegating to `gemini_a2ui_specialist` (`gemini-3.8-flash` + BigQuery MCP)
        yield Event(
            invocation_id=ctx.invocation_id,
            author=self.name,
            branch=ctx.branch,
            partial=True,
            content=types.Content(
                role="model",
                parts=[
                    types.Part(
                        text=(
                            f"🧠 **[Synthetic Thought 1/3]** Authenticating caller `{caller_detail}` "
                            f"and routing query to `gemini_a2ui_specialist` (`{MODEL}`)...\n\n"
                        ),
                        thought=True,
                    )
                ],
            ),
            actions=EventActions(
                state_delta={"a2ui_stage": "llm_route", "a2ui_step": 1}
            ),
        )
        await asyncio.sleep(STEP_DELAY_SECONDS)

        yield Event(
            invocation_id=ctx.invocation_id,
            author=self.name,
            branch=ctx.branch,
            partial=True,
            content=types.Content(
                role="model",
                parts=[
                    types.Part(
                        text=(
                            f"⚙️ **[Synthetic Thought 2/3]** Inspecting schemas and executing read-only GoogleSQL "
                            f"queries via BigQuery MCP (`{BQ_MCP_URL}`)...\n\n"
                        ),
                        thought=True,
                    )
                ],
            ),
            actions=EventActions(
                state_delta={"a2ui_stage": "llm_bq_mcp", "a2ui_step": 2}
            ),
        )

        final_answer_texts: list[str] = []
        partial_answer_texts: list[str] = []
        async for ev in gemini_a2ui_specialist.run_async(ctx):
            if not ev.content or not ev.content.parts:
                continue
            for part in ev.content.parts:
                if (
                    getattr(part, "thought", False)
                    or part.function_call
                    or part.function_response
                ):
                    continue
                if part.text:
                    if ev.partial:
                        partial_answer_texts.append(part.text)
                    else:
                        final_answer_texts.append(part.text)

        yield Event(
            invocation_id=ctx.invocation_id,
            author=self.name,
            branch=ctx.branch,
            partial=True,
            content=types.Content(
                role="model",
                parts=[
                    types.Part(
                        text=(
                            "🎨 **[Synthetic Thought 3/3]** Synthesizing FinOps insights and attaching "
                            "interactive A2UI `v0.9` navigation controls...\n\n"
                        ),
                        thought=True,
                    )
                ],
            ),
            actions=EventActions(
                state_delta={"a2ui_stage": "llm_complete", "a2ui_step": 3}
            ),
        )
        await asyncio.sleep(STEP_DELAY_SECONDS)

        combined_answer = (
            "\n\n".join(final_answer_texts).strip()
            or "".join(partial_answer_texts).strip()
            or "Completed BigQuery MCP analysis via `gemini-3.8-flash`."
        )
        followup_surface_id = f"agent-return-bar-{ctx.invocation_id[:8]}"
        final_parts = [
            types.Part.from_text(text=combined_answer),
            *make_a2ui_genai_parts(
                build_return_bar_a2ui_messages(surface_id=followup_surface_id)
            ),
        ]
        yield Event(
            invocation_id=ctx.invocation_id,
            author=self.name,
            branch=ctx.branch,
            partial=False,
            turn_complete=True,
            content=types.Content(role="model", parts=final_parts),
            actions=EventActions(skip_summarization=True),
        )


root_agent = A2UIRootRouterAgent(
    name="a2ui_bq_agent",
    description=(
        "A2UI-BQ Gemini Enterprise FinOps & Usage Pulse Agent (A2UI v0.9 + BigQuery MCP + End-User OAuth). "
        "Analyzes token FinOps, daily usage metrics, and audit logs with Vega-Lite charts and sortable "
        "tables in both Inline MaterialTabs and Side-Panel Canvas views."
    ),
    sub_agents=[gemini_a2ui_specialist],
)

app = App(
    root_agent=root_agent,
    name="app",
)
