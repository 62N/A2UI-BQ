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

Modular architecture:
- `app/prompts.py`: Starter prompts, predefined BigQuery MCP SQL queries, and system instructions.
- `app/tools.py`: Strict Pydantic I/O JSON schemas, BigQuery MCP tools, HITL confirmation tool,
  SQL guardrails, and guided error recovery.
- `app/a2ui_surfaces.py`: A2UI v0.9 surface builders (`MaterialTabs`, `Canvas`, `VegaChart`,
  `GcbpTable`, HITL confirmation card) and table sorting utilities.
- `app/observability.py`: Google Cloud `StructuredLogHandler`, `PIIRedactionFilter`,
  ADK OpenTelemetry (`google.adk.telemetry.tracer`), and `INTENT_VS_OUTCOME` auditing.
- `app/agent.py`: Memory Bank consolidation, context compaction, `gemini_a2ui_specialist`,
  `A2UIRootRouterAgent`, and `App` configuration.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
import json
import logging
import os
from typing import Any

from google.adk.agents import Agent, BaseAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.context_cache_config import ContextCacheConfig
from google.adk.agents.invocation_context import InvocationContext
from google.adk.apps import App
from google.adk.apps.app import EventsCompactionConfig, ResumabilityConfig
from google.adk.events import Event, EventActions
from google.adk.models import Gemini
from google.adk.planners import BuiltInPlanner
from google.adk.plugins.context_filter_plugin import ContextFilterPlugin
from google.adk.plugins.logging_plugin import LoggingPlugin
from google.adk.plugins.reflect_retry_tool_plugin import ReflectAndRetryToolPlugin
from google.adk.tools.load_memory_tool import LoadMemoryTool
from google.adk.tools.preload_memory_tool import PreloadMemoryTool
from google.genai import types

from app.a2ui_surfaces import (
    _build_audit_activity_dashboard_payload,
    _build_daily_usage_dashboard_payload,
    _build_token_finops_dashboard_payload,
    _extract_a2ui_action,
    _extract_prompt_from_a2ui_payload,
    _extract_user_text,
    _find_a2ui_action_in_payload,
    _infer_dashboard_kind_from_sort_field,
    _parse_sort_cell,
    _resolve_user_greeting_context,
    _sort_table_rows,
    build_canvas_finops_a2ui_messages,
    build_hitl_confirmation_a2ui_messages,
    build_inline_tabs_finops_a2ui_messages,
    build_return_bar_a2ui_messages,
    build_welcome_a2ui_messages,
    make_a2ui_genai_parts,
)
from app.app_utils import services
from app.oauth_bq_mcp import (
    BQ_MCP_URL,
    execute_bq_mcp_rpc,
    execute_bq_mcp_sql,
)
from app.observability import (
    AdkFinOpsTelemetryPlugin,
    log_structured_event,
    redact_pii_text,
    redact_principal,
    scrub_pii,
    trace_span,
    track_intent_outcome,
)
from app.prompts import (
    A2UI_V09_CATALOG_ID,
    RETURN_WELCOME_PROMPT,
    SQL_AUDIT_ACTIVITY_BY_METHOD,
    SQL_DAILY_USAGE_METRICS,
    SQL_TOKEN_FINOPS_BY_AGENT,
    STARTER_PROMPT_1,
    STARTER_PROMPT_2,
    STARTER_PROMPT_3,
    STARTER_PROMPT_4,
    STARTER_PROMPTS_LIST,
    USER_PROJECT,
    build_specialist_instruction,
)
from app.tools import (
    DatasetAndTableListRequest,
    DatasetAndTableListResponse,
    ExecuteReadonlySqlRequest,
    ExecuteReadonlySqlResponse,
    FinOpsBudgetAlertActionRequest,
    FinOpsBudgetAlertActionResponse,
    GuidedToolErrorRecovery,
    TableSchemaInfoRequest,
    TableSchemaInfoResponse,
    _classify_and_build_error_recovery,
    _surface_tool_error,
    bq_mcp_create_finops_budget_alert,
    bq_mcp_execute_sql_readonly,
    bq_mcp_get_table_info,
    bq_mcp_list_datasets_and_tables,
    enforce_hitl_and_sql_guardrails,
    finops_budget_alert_hitl_tool,
)

logger = logging.getLogger(__name__)

MODEL = "gemini-3.8-flash"
STEP_DELAY_SECONDS = float(os.getenv("A2UI_BQ_STEP_DELAY_SECONDS", "1.5"))

_BACKGROUND_MEMORY_TASKS: set[asyncio.Task[Any]] = set()
MAX_HISTORY_TURNS = 8
KEEP_RECENT_TURNS = 4


def compact_conversation_history(
    contents: list[types.Content],
    *,
    max_turns: int = MAX_HISTORY_TURNS,
    keep_recent_turns: int = KEEP_RECENT_TURNS,
) -> list[types.Content]:
    """Explicitly compact older conversation turns into a single semantic summary turn
    when turn count exceeds `max_turns`, preserving recent context and FinOps state."""
    if len(contents) <= max_turns:
        return contents

    with trace_span(
        "a2ui_bq.memory.compact_history",
        attributes={
            "memory.original_turns": len(contents),
            "memory.max_turns": max_turns,
        },
    ):
        keep_tail = max(1, min(keep_recent_turns, len(contents) - 1))
        older_turns = contents[:-keep_tail]
        recent_turns = contents[-keep_tail:]

        digest_lines = [
            "[Compacted Session Context Summary — Older turns condensed]"
        ]
        for idx, turn in enumerate(older_turns[-10:]):
            role = turn.role or "user"
            snippets: list[str] = []
            for p in turn.parts or []:
                if p.text:
                    clean = " ".join(redact_pii_text(p.text).split())
                    if len(clean) > 140:
                        clean = clean[:137] + "..."
                    snippets.append(clean)
            if snippets:
                digest_lines.append(
                    f"- Turn {idx + 1} ({role}): {' | '.join(snippets)}"
                )

        compacted_turn = types.Content(
            role="user",
            parts=[types.Part.from_text(text="\n".join(digest_lines))],
        )
        compacted = [compacted_turn, *recent_turns]
        log_structured_event(
            "HISTORY_COMPACTION_APPLIED",
            f"Compacted conversation history from {len(contents)} turns to {len(compacted)} turns",
            component="memory",
            structured_data={
                "original_turns": len(contents),
                "compacted_turns": len(compacted),
            },
        )
        return compacted


async def consolidate_session_to_memory_bank(
    callback_context: CallbackContext,
) -> types.Content | None:
    """`after_agent_callback` that consolidates completed session turns into
    GCP Vertex AI Memory Bank (`VertexAiMemoryBankService` / `InMemoryMemoryService`)."""
    with trace_span("a2ui_bq.memory.add_session_to_memory"):
        try:
            await callback_context.add_session_to_memory()
            log_structured_event(
                "MEMORY_BANK_CONSOLIDATION_SUCCESS",
                "Consolidated session events into GCP Memory Service via CallbackContext",
                component="memory",
            )
        except Exception as exc:
            log_structured_event(
                "MEMORY_BANK_CONSOLIDATION_SKIPPED",
                f"Memory Bank consolidation skipped: {redact_pii_text(str(exc))}",
                component="memory",
            )
    return None


async def _consolidate_invocation_memory_bg(
    ctx: InvocationContext,
    intent: str,
    dashboard_kind: str | None = None,
    use_canvas: bool | None = None,
) -> None:
    """Background asynchronous task that consolidates session state and events
    into GCP Vertex AI Memory Bank (`services.get_memory_service()`)."""
    await asyncio.sleep(0)
    with trace_span(
        "a2ui_bq.memory.async_background_consolidation",
        attributes={
            "memory.intent": intent,
            "memory.dashboard_kind": dashboard_kind or "n/a",
        },
    ):
        try:
            if ctx.session and isinstance(ctx.session.state, dict):
                mem_meta = ctx.session.state.get("consolidated_finops_memory") or {}
                if not isinstance(mem_meta, dict):
                    mem_meta = {}
                mem_meta["consolidation_count"] = (
                    int(mem_meta.get("consolidation_count", 0)) + 1
                )
                mem_meta["last_intent"] = intent
                if dashboard_kind:
                    recent = list(mem_meta.get("recent_dashboards", []))
                    if not recent or recent[-1] != dashboard_kind:
                        recent.append(dashboard_kind)
                    mem_meta["recent_dashboards"] = recent[-10:]
                if use_canvas is not None:
                    mem_meta["preferred_layout"] = (
                        "canvas" if use_canvas else "inline_tabs"
                    )
                ctx.session.state["consolidated_finops_memory"] = mem_meta

            memory_service = ctx.memory_service or services.get_memory_service()
            if memory_service and ctx.session:
                await memory_service.add_session_to_memory(ctx.session)
                log_structured_event(
                    "ASYNC_MEMORY_CONSOLIDATION_COMPLETED",
                    f"Asynchronously consolidated session '{ctx.session.id}' to GCP Memory Service",
                    component="memory",
                    structured_data={
                        "intent": intent,
                        "dashboard_kind": dashboard_kind or "n/a",
                    },
                )
        except Exception as exc:
            log_structured_event(
                "ASYNC_MEMORY_CONSOLIDATION_ERROR",
                f"Async memory consolidation warning: {redact_pii_text(str(exc))}",
                component="memory",
                severity=logging.WARNING,
            )


def schedule_async_memory_consolidation(
    ctx: InvocationContext,
    *,
    intent: str,
    dashboard_kind: str | None = None,
    use_canvas: bool | None = None,
) -> asyncio.Task[None] | None:
    """Schedule non-blocking asynchronous memory consolidation on the event loop."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return None
    task = loop.create_task(
        _consolidate_invocation_memory_bg(
            ctx,
            intent=intent,
            dashboard_kind=dashboard_kind,
            use_canvas=use_canvas,
        )
    )
    _BACKGROUND_MEMORY_TASKS.add(task)
    task.add_done_callback(_BACKGROUND_MEMORY_TASKS.discard)
    return task


def _sanitize_a2ui_llm_request(
    callback_context: Any, llm_request: Any
) -> Any:
    """Sanitize `<a2a_datapart_json>` blobs, empty greeting turns, and compact long
    conversation histories in `llm_request.contents`."""
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

    llm_request.contents = compact_conversation_history(list(llm_request.contents))
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
    instruction=build_specialist_instruction(USER_PROJECT),
    planner=BuiltInPlanner(
        thinking_config=types.ThinkingConfig(
            include_thoughts=False,
            thinking_budget=256,
        )
    ),
    before_model_callback=_sanitize_a2ui_llm_request,
    before_tool_callback=enforce_hitl_and_sql_guardrails,
    after_agent_callback=consolidate_session_to_memory_bank,
    tools=[
        PreloadMemoryTool(),
        LoadMemoryTool(),
        bq_mcp_list_datasets_and_tables,
        bq_mcp_get_table_info,
        bq_mcp_execute_sql_readonly,
        finops_budget_alert_hitl_tool,
    ],
    on_tool_error_callback=_surface_tool_error,
)


class A2UIRootRouterAgent(BaseAgent):
    """Low-latency hybrid router combining instant A2UI v0.9 greeting,
    live BigQuery MCP FinOps dashboards (Inline MaterialTabs & Side-Panel Canvas),
    `GcbpTable` `fetchData` sort actions, HITL confirmation gates, and
    Gemini 3.8 Flash + BigQuery MCP analysis."""

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

        # Path 0a: Intercept Human-in-the-Loop (HITL) Confirmation / Rejection actions
        if a2ui_action and a2ui_action.get("name") in {"hitl.confirm", "hitl.reject"}:
            act_name = str(a2ui_action.get("name"))
            with track_intent_outcome(
                invocation_id=ctx.invocation_id,
                user_identity=user_display,
                classified_intent=f"hitl_{act_name.replace('.', '_')}",
                router_path="path_0a_hitl_decision",
                target_resource="finops_budget_alert",
            ) as outcome:
                pending = session_state.get("pending_hitl_action") or {
                    "target_agent_or_service": "root_agent",
                    "daily_budget_usd": 25.0,
                    "notification_channel": "finops-slack-alerts",
                }
                surface_id = f"agent-hitl-res-{ctx.invocation_id[:8]}"
                outcome["surface_id"] = surface_id
                if act_name == "hitl.confirm":
                    session_state["hitl_confirmed"] = True
                    res = bq_mcp_create_finops_budget_alert(
                        target_agent_or_service=str(
                            pending.get("target_agent_or_service", "root_agent")
                        ),
                        daily_budget_usd=float(pending.get("daily_budget_usd", 25.0)),
                        notification_channel=str(
                            pending.get("notification_channel", "finops-slack-alerts")
                        ),
                        tool_context=None,
                    )
                    res["status"] = "approved_and_created"
                    res["requires_human_confirmation"] = False
                    outcome["actual_action"] = "Approved and executed FinOps budget alert"
                    md_text = (
                        f"✅ **Human-in-the-Loop Action Approved & Executed**\n\n"
                        f"- **Target Agent/Service**: `{res['target_agent_or_service']}`\n"
                        f"- **Daily Budget Threshold**: `${res['daily_budget_usd']:.2f}/day`\n"
                        f"- **Notification Channel**: `{res['notification_channel']}`"
                    )
                else:
                    outcome["actual_action"] = "Rejected FinOps budget alert action"
                    md_text = "❌ **Human-in-the-Loop Action Cancelled** — No FinOps budget alert changes were applied."

                schedule_async_memory_consolidation(
                    ctx, intent=act_name, dashboard_kind="hitl_governance"
                )
                yield Event(
                    invocation_id=ctx.invocation_id,
                    author=self.name,
                    branch=ctx.branch,
                    partial=False,
                    turn_complete=True,
                    content=types.Content(
                        role="model",
                        parts=[
                            types.Part.from_text(text=md_text),
                            *make_a2ui_genai_parts(
                                build_return_bar_a2ui_messages(surface_id=surface_id)
                            ),
                        ],
                    ),
                    actions=EventActions(
                        skip_summarization=True,
                        state_delta={
                            "pending_hitl_action": None,
                            "hitl_confirmed": act_name == "hitl.confirm",
                        },
                    ),
                )
                return

        if "budget alert" in lowered and (
            "create" in lowered
            or "set" in lowered
            or "propose" in lowered
            or "schedule" in lowered
        ):
            with track_intent_outcome(
                invocation_id=ctx.invocation_id,
                user_identity=user_display,
                classified_intent="propose_finops_budget_alert",
                router_path="path_0b_hitl_gate",
                target_resource="finops_budget_alert",
            ) as outcome:
                hitl_surface_id = f"agent-hitl-{ctx.invocation_id[:8]}"
                outcome["status"] = "CONFIRMATION_REQUIRED"
                outcome["actual_action"] = "Rendered A2UI v0.9 HITL approval gate"
                outcome["surface_id"] = hitl_surface_id
                pending_payload = {
                    "target_agent_or_service": "root_agent",
                    "daily_budget_usd": 25.0,
                    "notification_channel": "finops-slack-alerts",
                }
                hitl_msgs = build_hitl_confirmation_a2ui_messages(
                    surface_id=hitl_surface_id,
                    action_title="Confirm FinOps Daily Budget Alert ($25.00/day)",
                    action_summary=(
                        f"You requested creating a FinOps budget alert for `root_agent` in `{USER_PROJECT}` "
                        "notifying `finops-slack-alerts` when daily spend exceeds $25.00. Please confirm to execute."
                    ),
                    confirm_prompt="Approve FinOps budget alert",
                )
                yield Event(
                    invocation_id=ctx.invocation_id,
                    author=self.name,
                    branch=ctx.branch,
                    partial=False,
                    turn_complete=True,
                    content=types.Content(
                        role="model",
                        parts=[
                            types.Part.from_text(
                                text="🛡️ **Confirmation Required**: Please review and approve the proposed FinOps budget alert below."
                            ),
                            *make_a2ui_genai_parts(hitl_msgs),
                        ],
                    ),
                    actions=EventActions(
                        skip_summarization=True,
                        state_delta={"pending_hitl_action": pending_payload},
                    ),
                )
                return

        # Path 0: Intercept `GcbpTable` `fetchData` column-sort actions
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

            with track_intent_outcome(
                invocation_id=ctx.invocation_id,
                user_identity=user_display,
                classified_intent=f"sort_table_{dashboard_kind}",
                router_path="path_0_table_sort",
                target_resource=target_table_label,
            ) as outcome:
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

                outcome["surface_id"] = viz_surface_id
                outcome["row_count"] = int(cached_bq_res.get("row_count", 0))
                outcome["actual_action"] = (
                    f"Sorted {dashboard_kind} table by '{sort_string}'"
                )
                schedule_async_memory_consolidation(
                    ctx,
                    intent=f"sort_{dashboard_kind}",
                    dashboard_kind=dashboard_kind,
                    use_canvas=use_canvas,
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
            with track_intent_outcome(
                invocation_id=ctx.invocation_id,
                user_identity=user_display,
                classified_intent="welcome_greeting",
                router_path="path_1_instant_greeting",
                target_resource="agent-welcome",
            ) as outcome:
                already_rendered = bool(session_state.get("agent_welcome_rendered"))
                surface_id = (
                    "agent-welcome"
                    if (not lowered and not already_rendered)
                    else f"agent-welcome-{ctx.invocation_id[:8]}"
                )
                outcome["surface_id"] = surface_id
                outcome["actual_action"] = (
                    f"Rendered A2UI v0.9 welcome surface ({surface_id})"
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
            if (
                is_daily_usage
                and "inline" not in lowered
                and "materialtabs" not in lowered
            ):
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

            with track_intent_outcome(
                invocation_id=ctx.invocation_id,
                user_identity=user_display,
                classified_intent=f"finops_dashboard_{dashboard_kind}",
                router_path="path_2_finops_dashboard",
                target_resource=target_table_label,
            ) as outcome:
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
                    outcome["status"] = "ERROR"
                    outcome["error_code"] = type(exc).__name__
                    outcome["actual_action"] = (
                        f"BigQuery MCP query failed: {type(exc).__name__}"
                    )
                    err_msg = (
                        f"❌ **BigQuery MCP Query Failed**: `{redact_pii_text(str(exc))}`"
                    )
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

                outcome["surface_id"] = viz_surface_id
                outcome["row_count"] = int(bq_res.get("row_count", 0))
                outcome["actual_action"] = (
                    f"Rendered {dashboard_kind} dashboard ({'Canvas' if use_canvas else 'MaterialTabs'})"
                )
                schedule_async_memory_consolidation(
                    ctx,
                    intent=f"dashboard_{dashboard_kind}",
                    dashboard_kind=dashboard_kind,
                    use_canvas=use_canvas,
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
        with track_intent_outcome(
            invocation_id=ctx.invocation_id,
            user_identity=user_display,
            classified_intent="freeform_bq_mcp_analysis",
            router_path="path_3_llm_specialist",
            target_resource=MODEL,
        ) as outcome:
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
            outcome["surface_id"] = followup_surface_id
            outcome["actual_action"] = (
                "Completed Gemini 3.8 Flash + BigQuery MCP specialist analysis"
            )
            schedule_async_memory_consolidation(
                ctx, intent="freeform_llm_analysis"
            )

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
    after_agent_callback=consolidate_session_to_memory_bank,
)

app = App(
    root_agent=root_agent,
    name="app",
    events_compaction_config=EventsCompactionConfig(
        token_threshold=32000,
        event_retention_size=5,
        compaction_interval=6,
        overlap_size=2,
    ),
    context_cache_config=ContextCacheConfig(
        min_tokens=2048,
        ttl_seconds=1800,
        cache_intervals=10,
    ),
    resumability_config=ResumabilityConfig(is_resumable=True),
    plugins=[
        LoggingPlugin(),
        AdkFinOpsTelemetryPlugin(),
        ReflectAndRetryToolPlugin(max_retries=2),
        ContextFilterPlugin(num_invocations_to_keep=8),
    ],
)
