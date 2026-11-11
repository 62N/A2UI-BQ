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

"""Google Cloud Structured Logging, ADK OpenTelemetry GenAI Tracing, PII Redaction, and Intent-vs-Outcome Auditing.

Integrates native ADK and Google Cloud observability primitives:
1. OpenTelemetry GenAI Semantic Conventions & ADK Telemetry (`google.adk.telemetry.tracer`):
   - Defaults `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT="NO_CONTENT"` and
     `ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS="false"` so raw prompt/response PII is never leaked in spans.
   - Emits child spans under ADK's `invoke_agent` / `call_llm` / `execute_tool` trace hierarchy.
2. Google Cloud Structured Logging (`google.cloud.logging_v2.handlers.StructuredLogHandler` &
   `google.cloud.logging.Client`) with `PIIRedactionFilter` so every log record is emitted as
   Cloud Logging `jsonPayload` with emails, human names, and OAuth tokens scrubbed.
3. Native ADK `BasePlugin` (`AdkFinOpsTelemetryPlugin`) that hooks `before_run_callback`,
   `after_run_callback`, `before_tool_callback`, `after_tool_callback`, and `on_tool_error_callback`
   alongside ADK's built-in `LoggingPlugin`, `ReflectAndRetryToolPlugin`, `ContextFilterPlugin`,
   and optional `BigQueryAgentAnalyticsPlugin`.
"""

from __future__ import annotations

from collections.abc import Generator, Mapping
import contextlib
import hashlib
import json
import logging
import os
import re
import time
import uuid
from typing import Any, Literal

from google.adk.plugins.base_plugin import BasePlugin
from google.adk.telemetry import tracer as adk_tracer
from google.cloud.logging_v2.handlers import StructuredLogHandler
from opentelemetry import trace
from opentelemetry.trace import Span, Status, StatusCode
from pydantic import BaseModel, ConfigDict, Field

SERVICE_NAME = "a2ui-bq-finops-agent"

# Enforce OpenTelemetry GenAI privacy defaults so ADK spans do not capture raw PII message content
os.environ.setdefault("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "NO_CONTENT")
os.environ.setdefault("ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS", "false")
os.environ.setdefault("OTEL_SEMCONV_STABILITY_OPT_IN", "gen_ai_latest_experimental")

# Regex patterns for PII and credential scrubbing
_EMAIL_RE = re.compile(r"\b([A-Za-z0-9._%+-]+)@([A-Za-z0-9.-]+\.[A-Za-z]{2,})\b")
_GOOGLE_TOKEN_RE = re.compile(r"\bya29\.[A-Za-z0-9._\-]+\b")
_JWT_RE = re.compile(r"\beyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]*\b")
_BEARER_RE = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._\-]+\b")

_SENSITIVE_TOKEN_KEYS = frozenset(
    {
        "user_oauth_token",
        "access_token",
        "refresh_token",
        "id_token",
        "subject_token",
        "authorization",
        "x-forwarded-authorization",
        "x-goog-oauth-token",
        "x-vertex-ai-oauth-token",
        "token",
    }
)

_PII_IDENTITY_KEYS = frozenset(
    {
        "email",
        "user_email",
        "human_user",
        "display_user",
        "auth_principal",
        "principal",
        "name",
        "given_name",
        "family_name",
        "upn",
        "preferred_username",
        "user_name",
        "username",
        "displayname",
    }
)


def redact_principal(value: str | None) -> str:
    """Pseudonymize a user email or display name into a deterministic non-PII identifier."""
    if not value or not isinstance(value, str):
        return "anonymous"
    cleaned = value.strip()
    if not cleaned or cleaned.lower() in {"guest", "anonymous", "n/a"}:
        return "anonymous"
    digest = hashlib.sha256(cleaned.lower().encode("utf-8")).hexdigest()[:10]
    if "@" in cleaned:
        _, domain = cleaned.split("@", 1)
        domain_parts = domain.split(".")
        masked_domain = (
            f"{domain_parts[0][:1]}***.{domain_parts[-1]}"
            if len(domain_parts) >= 2
            else "***"
        )
        return f"user_{digest}@{masked_domain}"
    return f"user_{digest}"


def redact_pii_text(text: str) -> str:
    """Scrub OAuth tokens, JWTs, and email addresses from arbitrary strings."""
    if not text or not isinstance(text, str):
        return text
    scrubbed = _GOOGLE_TOKEN_RE.sub("[REDACTED_OAUTH_TOKEN]", text)
    scrubbed = _JWT_RE.sub("[REDACTED_JWT]", scrubbed)
    scrubbed = _BEARER_RE.sub("Bearer [REDACTED_TOKEN]", scrubbed)
    scrubbed = _EMAIL_RE.sub(lambda m: redact_principal(m.group(0)), scrubbed)
    return scrubbed


def scrub_pii(obj: Any, *, _key: str | None = None) -> Any:
    """Recursively scrub PII (emails, human names, OAuth tokens) from dicts, lists, and strings."""
    norm_key = (_key or "").lower()
    if norm_key in _SENSITIVE_TOKEN_KEYS:
        return "[REDACTED_TOKEN]" if obj else None
    if norm_key in _PII_IDENTITY_KEYS and isinstance(obj, str):
        return redact_principal(obj)

    if isinstance(obj, str):
        return redact_pii_text(obj)
    if isinstance(obj, Mapping):
        return {str(k): scrub_pii(v, _key=str(k)) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [scrub_pii(item, _key=_key) for item in obj]
    return obj


class PIIRedactionFilter(logging.Filter):
    """Logging filter that scrubs PII from `record.msg`, `record.args`, and `json_fields`
    before Google Cloud's `StructuredLogHandler` serializes the log entry."""

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = redact_pii_text(record.msg)
        if record.args:
            if isinstance(record.args, Mapping):
                record.args = {
                    k: scrub_pii(v, _key=str(k)) for k, v in record.args.items()
                }
            elif isinstance(record.args, tuple):
                record.args = tuple(scrub_pii(a) for a in record.args)

        json_fields = getattr(record, "json_fields", None)
        if isinstance(json_fields, Mapping):
            record.json_fields = scrub_pii(dict(json_fields))
        return True


_LOGGING_CONFIGURED = False


def configure_structured_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure application logging using Google Cloud's official `StructuredLogHandler`
    and `PIIRedactionFilter`."""
    global _LOGGING_CONFIGURED
    app_logger = logging.getLogger("app")
    app_logger.setLevel(level)

    if not _LOGGING_CONFIGURED:
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
        handler = StructuredLogHandler(
            project_id=project_id,
            labels={"service": SERVICE_NAME, "pii_redacted": "true"},
        )
        handler.addFilter(PIIRedactionFilter())
        app_logger.handlers = [handler]
        app_logger.propagate = False
        _LOGGING_CONFIGURED = True
    return app_logger


def get_tracer() -> trace.Tracer:
    """Return ADK's OpenTelemetry tracer (`google.adk.telemetry.tracer`)."""
    return adk_tracer


@contextlib.contextmanager
def trace_span(
    span_name: str,
    attributes: Mapping[str, Any] | None = None,
) -> Generator[Span, None, None]:
    """Start a child span using `google.adk.telemetry.tracer` with automatic PII scrubbing."""
    with adk_tracer.start_as_current_span(span_name) as span:
        if attributes:
            for k, v in attributes.items():
                if v is not None:
                    scrubbed_val = scrub_pii(v, _key=k)
                    if isinstance(scrubbed_val, (str, bool, int, float)):
                        span.set_attribute(k, scrubbed_val)
                    else:
                        span.set_attribute(k, json.dumps(scrubbed_val, default=str))
        try:
            yield span
            span.set_status(Status(StatusCode.OK))
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR, redact_pii_text(str(exc))))
            raise


class Feedback(BaseModel):
    """Pydantic model for end-user feedback routed to Cloud Logging & BigQuery sinks."""

    model_config = ConfigDict(extra="forbid")

    score: int | float = Field(..., description="Numeric user rating or score.")
    text: str | None = Field(
        default="", description="Optional user feedback comment (PII-redacted before logging)."
    )
    log_type: Literal["feedback"] = "feedback"
    service_name: Literal["a2ui-bq-finops-agent"] = "a2ui-bq-finops-agent"
    user_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))


class IntentOutcomeRecord(BaseModel):
    """Strict schema capturing user intent versus execution outcome for auditability."""

    model_config = ConfigDict(extra="forbid")

    invocation_id: str = Field(..., description="Unique ADK invocation ID.")
    user_pseudonym: str = Field(
        ..., description="PII-redacted pseudonymized identifier of the caller."
    )
    classified_intent: str = Field(
        ...,
        description="Classified intent category (e.g. greeting, token_finops, daily_usage, audit_activity, table_sort, hitl_confirmation, llm_specialist).",
    )
    router_path: str = Field(
        ..., description="Selected execution path in A2UIRootRouterAgent."
    )
    target_resource: str = Field(
        default="n/a",
        description="Target BigQuery table, MCP tool, or A2UI surface.",
    )
    outcome_status: str = Field(
        ...,
        description="Execution outcome status: SUCCESS, ERROR, or CONFIRMATION_REQUIRED.",
    )
    actual_action: str = Field(
        ..., description="Summary of the action actually performed."
    )
    surface_id: str = Field(
        default="n/a", description="Rendered A2UI v0.9 surfaceId if applicable."
    )
    row_count: int = Field(
        default=0, ge=0, description="Number of BigQuery rows processed."
    )
    latency_ms: float = Field(
        default=0.0, ge=0.0, description="End-to-end execution latency in milliseconds."
    )
    error_code: str | None = Field(
        default=None, description="Error code or exception class if outcome_status is ERROR."
    )


def log_structured_event(
    event_type: str,
    message: str,
    *,
    component: str = "agent",
    severity: int = logging.INFO,
    intent: Mapping[str, Any] | None = None,
    outcome: Mapping[str, Any] | None = None,
    structured_data: Mapping[str, Any] | None = None,
    latency_ms: float | None = None,
) -> dict[str, Any]:
    """Emit a PII-scrubbed structured JSON log event via Google Cloud `StructuredLogHandler`."""
    app_logger = configure_structured_logging()
    scrubbed_msg = redact_pii_text(message)
    scrubbed_intent = scrub_pii(dict(intent)) if intent else None
    scrubbed_outcome = scrub_pii(dict(outcome)) if outcome else None
    scrubbed_data = scrub_pii(dict(structured_data)) if structured_data else None

    json_fields: dict[str, Any] = {
        "event_type": event_type,
        "service_name": SERVICE_NAME,
        "component": component,
        "pii_redacted": True,
    }
    if scrubbed_intent is not None:
        json_fields["intent"] = scrubbed_intent
    if scrubbed_outcome is not None:
        json_fields["outcome"] = scrubbed_outcome
    if scrubbed_data is not None:
        json_fields["structured_data"] = scrubbed_data
    if latency_ms is not None:
        json_fields["latency_ms"] = round(latency_ms, 2)

    app_logger.log(
        severity,
        scrubbed_msg,
        extra={"json_fields": json_fields},
    )
    return {"message": scrubbed_msg, **json_fields}


def log_intent_vs_outcome(record: IntentOutcomeRecord) -> dict[str, Any]:
    """Log an explicit Intent-vs-Outcome audit event and annotate the active ADK OpenTelemetry span."""
    intent_dict = {
        "invocation_id": record.invocation_id,
        "user_pseudonym": redact_principal(record.user_pseudonym),
        "classified_intent": record.classified_intent,
        "router_path": record.router_path,
        "target_resource": record.target_resource,
    }
    outcome_dict = {
        "status": record.outcome_status,
        "actual_action": redact_pii_text(record.actual_action),
        "surface_id": record.surface_id,
        "row_count": record.row_count,
        "latency_ms": round(record.latency_ms, 2),
        "error_code": record.error_code,
    }

    span = trace.get_current_span()
    if span and span.get_span_context().is_valid:
        span.set_attribute("gen_ai.agent.intent", record.classified_intent)
        span.set_attribute("gen_ai.agent.router_path", record.router_path)
        span.set_attribute("gen_ai.agent.target_resource", record.target_resource)
        span.set_attribute("gen_ai.agent.outcome.status", record.outcome_status)
        span.set_attribute("gen_ai.agent.outcome.surface_id", record.surface_id)
        span.set_attribute("gen_ai.agent.outcome.row_count", record.row_count)
        span.set_attribute("gen_ai.agent.outcome.latency_ms", round(record.latency_ms, 2))

    severity = logging.ERROR if record.outcome_status == "ERROR" else logging.INFO
    return log_structured_event(
        "INTENT_VS_OUTCOME",
        (
            f"Intent '{record.classified_intent}' ({record.router_path}) -> "
            f"Outcome '{record.outcome_status}' ({record.actual_action}) in {record.latency_ms:.1f}ms"
        ),
        component="router",
        severity=severity,
        intent=intent_dict,
        outcome=outcome_dict,
        latency_ms=round(record.latency_ms, 2),
    )


@contextlib.contextmanager
def track_intent_outcome(
    *,
    invocation_id: str,
    user_identity: str | None,
    classified_intent: str,
    router_path: str,
    target_resource: str = "n/a",
) -> Generator[dict[str, Any], None, None]:
    """Context manager that wraps an execution path in an ADK OpenTelemetry span and emits
    an explicit `INTENT_VS_OUTCOME` structured log entry upon completion or error."""
    t0 = time.perf_counter()
    pseudonym = redact_principal(user_identity)
    outcome_tracker: dict[str, Any] = {
        "status": "SUCCESS",
        "actual_action": classified_intent,
        "surface_id": "n/a",
        "row_count": 0,
        "error_code": None,
    }
    with trace_span(
        f"a2ui_bq.{router_path}",
        attributes={
            "gen_ai.conversation.id": invocation_id,
            "enduser.id": pseudonym,
            "gen_ai.agent.intent": classified_intent,
            "gen_ai.agent.router_path": router_path,
            "gen_ai.agent.target_resource": target_resource,
        },
    ):
        try:
            yield outcome_tracker
        except Exception as exc:
            outcome_tracker["status"] = "ERROR"
            outcome_tracker["error_code"] = type(exc).__name__
            outcome_tracker["actual_action"] = f"Failed with {type(exc).__name__}"
            raise
        finally:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            log_intent_vs_outcome(
                IntentOutcomeRecord(
                    invocation_id=invocation_id,
                    user_pseudonym=pseudonym,
                    classified_intent=classified_intent,
                    router_path=router_path,
                    target_resource=target_resource,
                    outcome_status=str(outcome_tracker["status"]),
                    actual_action=str(outcome_tracker["actual_action"]),
                    surface_id=str(outcome_tracker["surface_id"]),
                    row_count=int(outcome_tracker["row_count"] or 0),
                    latency_ms=elapsed_ms,
                    error_code=outcome_tracker.get("error_code"),
                )
            )


class AdkFinOpsTelemetryPlugin(BasePlugin):
    """Runner-wide ADK `BasePlugin` that records PII-scrubbed structured tool & agent
    telemetry and correlates intent vs. outcome across all sub-agents."""

    def __init__(self, name: str = "adk_finops_telemetry_plugin") -> None:
        super().__init__(name=name)

    async def before_tool_callback(
        self, *, tool: Any, tool_args: dict[str, Any], tool_context: Any
    ) -> dict[str, Any] | None:
        tool_name = getattr(tool, "name", str(tool))
        log_structured_event(
            "TOOL_INVOCATION_INTENT",
            f"Invoking tool '{tool_name}'",
            component="adk_plugin",
            intent={
                "tool_name": tool_name,
                "arguments": scrub_pii(tool_args),
            },
        )
        return None

    async def after_tool_callback(
        self,
        *,
        tool: Any,
        tool_args: dict[str, Any],
        tool_context: Any,
        result: dict[str, Any],
    ) -> dict[str, Any] | None:
        tool_name = getattr(tool, "name", str(tool))
        status = (
            "ERROR"
            if isinstance(result, Mapping) and result.get("status") == "error"
            else "SUCCESS"
        )
        log_structured_event(
            "TOOL_INVOCATION_OUTCOME",
            f"Tool '{tool_name}' completed with status={status}",
            component="adk_plugin",
            intent={"tool_name": tool_name, "arguments": scrub_pii(tool_args)},
            outcome={
                "status": status,
                "row_count": result.get("row_count", 0)
                if isinstance(result, Mapping)
                else 0,
            },
        )
        return None
