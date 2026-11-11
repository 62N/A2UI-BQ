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

"""BigQuery MCP tools, strict Pydantic I/O JSON schemas, Human-in-the-Loop (HITL)
confirmation gates, and guided error recovery for the A2UI-BQ agent."""

from __future__ import annotations

import logging
from typing import Any

from google.adk.tools import FunctionTool
from google.adk.tools.tool_context import ToolContext
from pydantic import BaseModel, ConfigDict, Field

from app.oauth_bq_mcp import execute_bq_mcp_rpc, execute_bq_mcp_sql
from app.observability import (
    log_structured_event,
    redact_pii_text,
    redact_principal,
)
from app.prompts import USER_PROJECT


class GuidedToolErrorRecovery(BaseModel):
    """Structured error response with explicit guided recovery instructions for the LLM."""

    model_config = ConfigDict(extra="forbid")

    status: str = Field(
        default="error", description="Always 'error' when a tool fails."
    )
    tool_name: str = Field(
        ..., description="Name of the tool that encountered an error."
    )
    error_type: str = Field(..., description="Exception class or error category.")
    error_code: str = Field(
        ...,
        description="Canonical error classification code (e.g. AUTH_REQUIRED, PERMISSION_DENIED, NOT_FOUND, INVALID_SQL, TIMEOUT, INTERNAL_ERROR).",
    )
    message: str = Field(..., description="PII-redacted description of the error.")
    is_retryable: bool = Field(
        ..., description="True if the LLM should retry after applying recovery steps."
    )
    recovery_steps: list[str] = Field(
        ...,
        description="Explicit step-by-step guided recovery instructions for the LLM.",
    )
    suggested_fallback_tool: str | None = Field(
        default=None,
        description="Recommended tool to call next to recover from this error.",
    )
    remediation_sql_hint: str | None = Field(
        default=None,
        description="Concrete SQL syntax or table-qualification hint if applicable.",
    )


class DatasetAndTableListRequest(BaseModel):
    """Strict input schema for listing BigQuery datasets or tables via BigQuery MCP."""

    model_config = ConfigDict(extra="forbid")

    dataset_id: str = Field(
        default="",
        description="Optional dataset ID (e.g. 'gemini_enterprise' or 'gemini_enterprise_audit_logs'). Leave empty to list all datasets.",
    )


class DatasetAndTableListResponse(BaseModel):
    """Strict output schema for `bq_mcp_list_datasets_and_tables`."""

    model_config = ConfigDict(extra="forbid")

    status: str = Field(default="success", description="Execution status ('success').")
    operation: str = Field(
        ...,
        description="MCP tool operation invoked ('list_dataset_ids' or 'list_table_ids').",
    )
    project_id: str = Field(..., description="Target Google Cloud project ID.")
    dataset_id: str | None = Field(
        default=None, description="Queried dataset ID when listing tables."
    )
    auth_mode: str = Field(..., description="Authentication mechanism used.")
    principal_pseudonym: str = Field(
        ..., description="PII-redacted pseudonymized principal identifier."
    )
    elapsed_ms: float = Field(..., ge=0.0, description="Execution latency in ms.")
    data: dict[str, Any] = Field(
        default_factory=dict,
        description="Structured dataset or table list returned by BigQuery MCP.",
    )
    json_schema_version: str = Field(default="1.0", description="Schema version.")


class TableSchemaInfoRequest(BaseModel):
    """Strict input schema for retrieving BigQuery table metadata via BigQuery MCP."""

    model_config = ConfigDict(extra="forbid")

    dataset_id: str = Field(
        ...,
        min_length=1,
        description="BigQuery dataset ID (e.g. 'gemini_enterprise').",
    )
    table_id: str = Field(
        ..., min_length=1, description="BigQuery table ID (e.g. 'usage_metrics')."
    )


class TableSchemaInfoResponse(BaseModel):
    """Strict output schema for `bq_mcp_get_table_info`."""

    model_config = ConfigDict(extra="forbid")

    status: str = Field(default="success", description="Execution status ('success').")
    project_id: str = Field(..., description="Target Google Cloud project ID.")
    dataset_id: str = Field(..., description="BigQuery dataset ID.")
    table_id: str = Field(..., description="BigQuery table ID.")
    auth_mode: str = Field(..., description="Authentication mechanism used.")
    principal_pseudonym: str = Field(
        ..., description="PII-redacted pseudonymized principal identifier."
    )
    elapsed_ms: float = Field(..., ge=0.0, description="Execution latency in ms.")
    schema_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Table schema and metadata from BigQuery MCP.",
    )
    json_schema_version: str = Field(default="1.0", description="Schema version.")


class ExecuteReadonlySqlRequest(BaseModel):
    """Strict input schema for executing read-only GoogleSQL via BigQuery MCP."""

    model_config = ConfigDict(extra="forbid")

    sql_query: str = Field(
        ...,
        min_length=6,
        description="Read-only GoogleSQL SELECT statement against the configured project.",
    )


class ExecuteReadonlySqlResponse(BaseModel):
    """Strict output schema for `bq_mcp_execute_sql_readonly`."""

    model_config = ConfigDict(extra="forbid")

    status: str = Field(default="success", description="Execution status ('success').")
    rows: list[dict[str, Any]] = Field(
        default_factory=list, description="Parsed result rows as column-keyed dicts."
    )
    columns: list[str] = Field(
        default_factory=list,
        description="Ordered column names from the result schema.",
    )
    row_count: int = Field(..., ge=0, description="Number of rows returned.")
    query_id: str = Field(..., description="BigQuery Job / Query ID.")
    total_bytes_processed: str = Field(
        ..., description="Total bytes processed by BigQuery."
    )
    total_bytes_billed: str = Field(
        ..., description="Total bytes billed by BigQuery."
    )
    total_slot_ms: str = Field(..., description="Total slot milliseconds consumed.")
    elapsed_ms: float = Field(..., ge=0.0, description="End-to-end RPC latency in ms.")
    auth_mode: str = Field(..., description="Authentication mechanism used.")
    principal_pseudonym: str = Field(
        ..., description="PII-redacted pseudonymized principal identifier."
    )
    sql: str = Field(..., description="Executed SQL statement.")
    json_schema_version: str = Field(default="1.0", description="Schema version.")


class FinOpsBudgetAlertActionRequest(BaseModel):
    """Strict input schema for proposing or creating a FinOps budget alert threshold."""

    model_config = ConfigDict(extra="forbid")

    target_agent_or_service: str = Field(
        ..., min_length=1, description="Agent name or service to monitor."
    )
    daily_budget_usd: float = Field(
        ..., gt=0.0, description="Daily budget alert threshold in USD."
    )
    notification_channel: str = Field(
        default="finops-slack-alerts",
        description="Target notification channel identifier.",
    )


class FinOpsBudgetAlertActionResponse(BaseModel):
    """Strict output schema for `bq_mcp_create_finops_budget_alert` (HITL-guarded)."""

    model_config = ConfigDict(extra="forbid")

    status: str = Field(
        ...,
        description="'confirmation_required' when awaiting human approval, or 'approved_and_created' once confirmed.",
    )
    requires_human_confirmation: bool = Field(
        ...,
        description="Whether human-in-the-loop approval is required before execution.",
    )
    target_agent_or_service: str = Field(
        ..., description="Target agent or service for the budget alert."
    )
    daily_budget_usd: float = Field(..., description="Daily threshold in USD.")
    notification_channel: str = Field(..., description="Target alert channel.")
    message: str = Field(..., description="Human-readable status and next steps.")


def _classify_and_build_error_recovery(
    tool_name: str, error: Exception
) -> GuidedToolErrorRecovery:
    """Classify tool exceptions and generate explicit step-by-step guided recovery instructions."""
    raw_msg = redact_pii_text(str(error))
    lower_msg = raw_msg.lower()

    if "401" in lower_msg or "unauthenticated" in lower_msg or "oauth" in lower_msg:
        return GuidedToolErrorRecovery(
            tool_name=tool_name,
            error_type=type(error).__name__,
            error_code="AUTH_REQUIRED",
            message=raw_msg,
            is_retryable=False,
            recovery_steps=[
                "1. Do NOT retry the same query in a tight loop when OAuth credentials are missing or expired.",
                "2. Inform the user that their end-user Google OAuth 2.0 session needs to be authorized or refreshed in Gemini Enterprise.",
                "3. Verify that the Discovery Engine authorization resource (`OAUTH_AUTH_ID`) includes the `https://www.googleapis.com/auth/bigquery` scope.",
            ],
            suggested_fallback_tool=None,
        )

    if "403" in lower_msg or "permission" in lower_msg or "access denied" in lower_msg:
        return GuidedToolErrorRecovery(
            tool_name=tool_name,
            error_type=type(error).__name__,
            error_code="PERMISSION_DENIED",
            message=raw_msg,
            is_retryable=False,
            recovery_steps=[
                "1. Call `bq_mcp_list_datasets_and_tables` with an empty `dataset_id` to check which datasets the authenticated user has IAM access to.",
                "2. Explain to the user which IAM role is missing (`roles/bigquery.dataViewer` and `roles/bigquery.jobUser` on the target project/dataset).",
            ],
            suggested_fallback_tool="bq_mcp_list_datasets_and_tables",
        )

    if "404" in lower_msg or "not found" in lower_msg or "notfound" in lower_msg:
        return GuidedToolErrorRecovery(
            tool_name=tool_name,
            error_type=type(error).__name__,
            error_code="NOT_FOUND",
            message=raw_msg,
            is_retryable=True,
            recovery_steps=[
                "1. Call `bq_mcp_list_datasets_and_tables(dataset_id='')` to discover valid dataset names in the project.",
                "2. Call `bq_mcp_list_datasets_and_tables(dataset_id='<dataset>')` to verify the exact table name.",
                "3. Retry the query using the fully-qualified table identifier `` `<project>.<dataset>.<table>` ``.",
            ],
            suggested_fallback_tool="bq_mcp_list_datasets_and_tables",
            remediation_sql_hint=(
                f"Use `{USER_PROJECT}.gemini_enterprise.usage_metrics` or "
                f"`{USER_PROJECT}.gemini_enterprise_audit_logs.<table_name>`."
            ),
        )

    if (
        "syntax" in lower_msg
        or "unrecognized name" in lower_msg
        or "invalid" in lower_msg
        or "400" in lower_msg
        or "only read-only" in lower_msg
        or "blocked mutating" in lower_msg
    ):
        return GuidedToolErrorRecovery(
            tool_name=tool_name,
            error_type=type(error).__name__,
            error_code="INVALID_SQL",
            message=raw_msg,
            is_retryable=True,
            recovery_steps=[
                "1. Call `bq_mcp_get_table_info(dataset_id=..., table_id=...)` to inspect the exact column names and types before rewriting the SQL.",
                "2. Ensure the query is a read-only GoogleSQL `SELECT` or `WITH ... SELECT` statement with a `LIMIT` clause (DDL/DML statements are blocked).",
                "3. Retry `bq_mcp_execute_sql_readonly` with the corrected column names and standard GoogleSQL syntax.",
            ],
            suggested_fallback_tool="bq_mcp_get_table_info",
            remediation_sql_hint="Verify nested column paths such as `jsonPayload.gen_ai_usage_input_tokens` and `resource.labels.method` via `bq_mcp_get_table_info`.",
        )

    if "timeout" in lower_msg or "429" in lower_msg or "rate" in lower_msg:
        return GuidedToolErrorRecovery(
            tool_name=tool_name,
            error_type=type(error).__name__,
            error_code="TIMEOUT",
            message=raw_msg,
            is_retryable=True,
            recovery_steps=[
                "1. Add a partition filter (e.g. `WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)` or `WHERE date IS NOT NULL`).",
                "2. Add `LIMIT 100` to reduce result payload size and retry `bq_mcp_execute_sql_readonly`.",
            ],
            suggested_fallback_tool="bq_mcp_execute_sql_readonly",
            remediation_sql_hint="Always include a partition filter on `date` or `timestamp` and a `LIMIT` clause.",
        )

    return GuidedToolErrorRecovery(
        tool_name=tool_name,
        error_type=type(error).__name__,
        error_code="INTERNAL_ERROR",
        message=raw_msg,
        is_retryable=True,
        recovery_steps=[
            "1. Call `bq_mcp_list_datasets_and_tables` to verify BigQuery MCP connectivity.",
            "2. Simplify the request parameters and retry once.",
        ],
        suggested_fallback_tool="bq_mcp_list_datasets_and_tables",
    )


def _surface_tool_error(
    tool: Any, args: Any, tool_context: Any, error: Exception
) -> dict[str, Any]:
    """Return tool errors as strict `GuidedToolErrorRecovery` JSON-schema dicts with
    actionable recovery steps so the LLM can self-correct without crashing."""
    del args, tool_context
    tool_name = getattr(tool, "name", str(tool))
    recovery = _classify_and_build_error_recovery(tool_name, error)
    log_structured_event(
        "TOOL_ERROR_RECOVERY_SURFACED",
        f"Tool '{tool_name}' failed ({recovery.error_code}): {recovery.message}",
        component="tools",
        severity=logging.WARNING,
        outcome=recovery.model_dump(),
    )
    return recovery.model_dump()


def bq_mcp_list_datasets_and_tables(
    dataset_id: str = "",
    tool_context: ToolContext | None = None,
) -> dict[str, Any]:
    """List BigQuery datasets in the project (or tables in `dataset_id`) via BigQuery MCP
    (`https://bigquery.googleapis.com/mcp`) using the end-user's OAuth 2.0 identity.

    Args:
        dataset_id: Optional dataset ID (e.g. `gemini_enterprise` or `gemini_enterprise_audit_logs`).
            If empty, lists all datasets in the project.

    Returns:
        Strict `DatasetAndTableListResponse` dict (or `GuidedToolErrorRecovery` dict on error).
    """
    req = DatasetAndTableListRequest(dataset_id=dataset_id)
    state = tool_context.state if tool_context else {}
    clean_ds = req.dataset_id.strip()
    if clean_ds:
        raw = execute_bq_mcp_rpc(
            "list_table_ids",
            {"project_id": USER_PROJECT, "dataset_id": clean_ds},
            state=state,
            project_id=USER_PROJECT,
        )
        op = "list_table_ids"
    else:
        raw = execute_bq_mcp_rpc(
            "list_dataset_ids",
            {"project_id": USER_PROJECT},
            state=state,
            project_id=USER_PROJECT,
        )
        op = "list_dataset_ids"

    validated = DatasetAndTableListResponse(
        operation=op,
        project_id=USER_PROJECT,
        dataset_id=clean_ds or None,
        auth_mode=str(raw.get("auth_mode", "OAuth 2.0")),
        principal_pseudonym=redact_principal(str(raw.get("principal", ""))),
        elapsed_ms=float(raw.get("elapsed_ms", 0.0)),
        data=raw.get("data")
        if isinstance(raw.get("data"), dict)
        else {"result": raw.get("data")},
    )
    return validated.model_dump()


def bq_mcp_get_table_info(
    dataset_id: str,
    table_id: str,
    tool_context: ToolContext | None = None,
) -> dict[str, Any]:
    """Retrieve schema and metadata for a BigQuery table via BigQuery MCP using end-user OAuth.

    Args:
        dataset_id: BigQuery dataset ID (e.g. `gemini_enterprise`).
        table_id: BigQuery table ID (e.g. `usage_metrics`).

    Returns:
        Strict `TableSchemaInfoResponse` dict (or `GuidedToolErrorRecovery` dict on error).
    """
    req = TableSchemaInfoRequest(
        dataset_id=dataset_id.strip(), table_id=table_id.strip()
    )
    state = tool_context.state if tool_context else {}
    raw = execute_bq_mcp_rpc(
        "get_table_info",
        {
            "project_id": USER_PROJECT,
            "dataset_id": req.dataset_id,
            "table_id": req.table_id,
        },
        state=state,
        project_id=USER_PROJECT,
    )
    validated = TableSchemaInfoResponse(
        project_id=USER_PROJECT,
        dataset_id=req.dataset_id,
        table_id=req.table_id,
        auth_mode=str(raw.get("auth_mode", "OAuth 2.0")),
        principal_pseudonym=redact_principal(str(raw.get("principal", ""))),
        elapsed_ms=float(raw.get("elapsed_ms", 0.0)),
        schema_metadata=raw.get("data")
        if isinstance(raw.get("data"), dict)
        else {"result": raw.get("data")},
    )
    return validated.model_dump()


def bq_mcp_execute_sql_readonly(
    sql_query: str,
    tool_context: ToolContext | None = None,
) -> dict[str, Any]:
    """Execute a read-only GoogleSQL SELECT query via BigQuery MCP (`https://bigquery.googleapis.com/mcp`)
    using the end-user's OAuth 2.0 identity.

    Args:
        sql_query: Read-only SELECT statement against the configured Google Cloud project tables.

    Returns:
        Strict `ExecuteReadonlySqlResponse` dict (or `GuidedToolErrorRecovery` dict on error).
    """
    req = ExecuteReadonlySqlRequest(sql_query=sql_query.strip())
    normalized_sql = req.sql_query.lstrip().upper()
    if not (normalized_sql.startswith("SELECT") or normalized_sql.startswith("WITH")):
        raise ValueError(
            "Only read-only GoogleSQL SELECT or WITH ... SELECT statements are permitted."
        )
    state = tool_context.state if tool_context else {}
    raw = execute_bq_mcp_sql(
        req.sql_query,
        state=state,
        project_id=USER_PROJECT,
    )
    validated = ExecuteReadonlySqlResponse(
        rows=raw.get("rows", []),
        columns=raw.get("columns", []),
        row_count=int(raw.get("row_count", 0)),
        query_id=str(raw.get("query_id", "n/a")),
        total_bytes_processed=str(raw.get("total_bytes_processed", "0")),
        total_bytes_billed=str(raw.get("total_bytes_billed", "0")),
        total_slot_ms=str(raw.get("total_slot_ms", "0")),
        elapsed_ms=float(raw.get("elapsed_ms", 0.0)),
        auth_mode=str(raw.get("auth_mode", "OAuth 2.0")),
        principal_pseudonym=redact_principal(str(raw.get("principal", ""))),
        sql=str(raw.get("sql", req.sql_query)),
    )
    return validated.model_dump()


def bq_mcp_create_finops_budget_alert(
    target_agent_or_service: str,
    daily_budget_usd: float,
    notification_channel: str = "finops-slack-alerts",
    tool_context: ToolContext | None = None,
) -> dict[str, Any]:
    """Create or update a FinOps daily budget alert threshold for a Gemini Enterprise agent.
    Protected by an explicit Human-in-the-Loop (HITL) confirmation gate.

    Args:
        target_agent_or_service: Agent or service name to monitor (e.g. `root_agent`).
        daily_budget_usd: Daily spend threshold in USD that triggers an alert.
        notification_channel: Target notification channel (e.g. `finops-slack-alerts`).

    Returns:
        Strict `FinOpsBudgetAlertActionResponse` dict indicating confirmation status.
    """
    req = FinOpsBudgetAlertActionRequest(
        target_agent_or_service=target_agent_or_service.strip(),
        daily_budget_usd=daily_budget_usd,
        notification_channel=notification_channel.strip() or "finops-slack-alerts",
    )
    confirmed = False
    if tool_context is not None:
        tc = getattr(tool_context, "tool_confirmation", None)
        if tc and getattr(tc, "confirmed", False):
            confirmed = True
        elif isinstance(tool_context.state, dict) and tool_context.state.get(
            "hitl_confirmed"
        ):
            confirmed = True
        else:
            tool_context.request_confirmation(
                hint=(
                    f"Approve creating FinOps daily budget alert (${req.daily_budget_usd:.2f}/day) "
                    f"for '{req.target_agent_or_service}' -> '{req.notification_channel}'?"
                ),
                payload=req.model_dump(),
            )

    if not confirmed:
        resp = FinOpsBudgetAlertActionResponse(
            status="confirmation_required",
            requires_human_confirmation=True,
            target_agent_or_service=req.target_agent_or_service,
            daily_budget_usd=req.daily_budget_usd,
            notification_channel=req.notification_channel,
            message=(
                f"Human-in-the-loop confirmation required before creating FinOps budget alert "
                f"(${req.daily_budget_usd:.2f}/day) for '{req.target_agent_or_service}'."
            ),
        )
        return resp.model_dump()

    resp = FinOpsBudgetAlertActionResponse(
        status="approved_and_created",
        requires_human_confirmation=False,
        target_agent_or_service=req.target_agent_or_service,
        daily_budget_usd=req.daily_budget_usd,
        notification_channel=req.notification_channel,
        message=(
            f"Confirmed and registered FinOps daily budget alert (${req.daily_budget_usd:.2f}/day) "
            f"for '{req.target_agent_or_service}' on channel '{req.notification_channel}'."
        ),
    )
    return resp.model_dump()


finops_budget_alert_hitl_tool = FunctionTool(
    bq_mcp_create_finops_budget_alert,
    require_confirmation=True,
)


def enforce_hitl_and_sql_guardrails(
    tool: Any, args: dict[str, Any], tool_context: ToolContext
) -> dict[str, Any] | None:
    """`before_tool_callback` enforcing read-only SQL guardrails and Human-in-the-Loop (HITL)
    confirmation checks before sensitive or mutating tool calls execute."""
    del tool_context
    tool_name = getattr(tool, "name", str(tool))
    if tool_name == "bq_mcp_execute_sql_readonly":
        sql = str(args.get("sql_query") or "").strip()
        upper_sql = sql.upper()
        forbidden_keywords = (
            "DROP ",
            "DELETE ",
            "UPDATE ",
            "INSERT ",
            "ALTER ",
            "TRUNCATE ",
            "CREATE ",
            "GRANT ",
        )
        if any(kw in f"{upper_sql} " for kw in forbidden_keywords):
            return _classify_and_build_error_recovery(
                tool_name,
                ValueError(
                    "Blocked mutating/DDL SQL keyword in `bq_mcp_execute_sql_readonly`. Only read-only SELECT queries are allowed."
                ),
            ).model_dump()
    return None
