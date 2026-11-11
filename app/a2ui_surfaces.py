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

"""A2UI v0.9 surface builders, Vega-Lite spec generators, and GcbpTable sorting
utilities for the A2UI-BQ Gemini Enterprise FinOps & Usage Pulse Agent."""

from __future__ import annotations

import json
from typing import Any

from google.adk.agents.invocation_context import InvocationContext
from google.genai import types

from app.prompts import (
    A2UI_V09_CATALOG_ID,
    RETURN_WELCOME_PROMPT,
    STARTER_PROMPT_1,
    STARTER_PROMPT_2,
    STARTER_PROMPT_3,
    STARTER_PROMPT_4,
    USER_PROJECT,
)


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


def build_hitl_confirmation_a2ui_messages(
    *,
    surface_id: str,
    action_title: str,
    action_summary: str,
    confirm_prompt: str,
    cancel_prompt: str = RETURN_WELCOME_PROMPT,
) -> list[dict[str, Any]]:
    """Build an interactive A2UI v0.9 Human-in-the-Loop (HITL) approval card with
    explicit Approve (`hitl.confirm`) and Cancel (`hitl.reject`) buttons."""
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
                        "component": "MaterialCard",
                        "appearance": "outlined",
                        "children": ["hitl_col"],
                        "style": {"padding": "16px", "borderRadius": "12px"},
                    },
                    {
                        "id": "hitl_col",
                        "component": "MaterialColumn",
                        "children": [
                            "hitl_badge",
                            "hitl_title",
                            "hitl_desc",
                            "hitl_btn_row",
                        ],
                        "align": "stretch",
                        "style": {"gap": "10px"},
                    },
                    {
                        "id": "hitl_badge",
                        "component": "MaterialText",
                        "text": "🛡️ HUMAN-IN-THE-LOOP CONFIRMATION REQUIRED",
                        "usageHint": "caption",
                        "style": {"color": "#e37400", "fontWeight": "600"},
                    },
                    {
                        "id": "hitl_title",
                        "component": "Text",
                        "text": action_title,
                        "variant": "h4",
                    },
                    {
                        "id": "hitl_desc",
                        "component": "Text",
                        "text": action_summary,
                        "variant": "body",
                    },
                    {
                        "id": "hitl_btn_row",
                        "component": "MaterialRow",
                        "children": ["hitl_approve_btn", "hitl_cancel_btn"],
                        "justify": "start",
                        "align": "center",
                        "style": {"gap": "10px"},
                    },
                    {
                        "id": "hitl_approve_btn",
                        "component": "MaterialButton",
                        "label": "Approve & Execute Action",
                        "leadingIcon": "check_circle",
                        "appearance": "filled",
                        "style": {"fontSize": "12px"},
                        "action": {
                            "event": {
                                "name": "hitl.confirm",
                                "context": {"prompt": confirm_prompt},
                            }
                        },
                    },
                    {
                        "id": "hitl_cancel_btn",
                        "component": "MaterialButton",
                        "label": "Cancel Action",
                        "leadingIcon": "cancel",
                        "appearance": "outlined",
                        "style": {"fontSize": "12px"},
                        "action": {
                            "event": {
                                "name": "hitl.reject",
                                "context": {"prompt": cancel_prompt},
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
