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

"""Attach A2A + A2UI v0.9 endpoints to the FastAPI app for Gemini Enterprise."""

from __future__ import annotations

import base64
import json
import logging
import os
from typing import TYPE_CHECKING, Any

logger = logging.getLogger(__name__)

from a2a.server.agent_execution import RequestContext
from a2a.server.events import Event as A2AEvent
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import (
    add_a2a_routes_to_fastapi,
    create_agent_card_routes,
    create_jsonrpc_routes,
)
from a2a.server.tasks import TaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentExtension,
    AgentInterface,
    Part as A2APart,
    TaskArtifactUpdateEvent,
)
from a2a.utils.constants import AGENT_CARD_WELL_KNOWN_PATH
from google.adk.a2a import _compat
from google.adk.a2a.converters.from_adk_event import (
    A2AUpdateEvent,
    convert_event_to_a2a_events as convert_event_v2,
)
from google.adk.a2a.converters.part_converter import (
    A2A_DATA_PART_END_TAG,
    A2A_DATA_PART_START_TAG,
    A2A_DATA_PART_TEXT_MIME_TYPE,
    A2APartToGenAIPartConverter,
    GenAIPartToA2APartConverter,
    convert_a2a_part_to_genai_part,
    convert_genai_part_to_a2a_part,
)
from google.adk.a2a.converters.request_converter import (
    AgentRunRequest,
    convert_a2a_request_to_agent_run_request,
)
from google.adk.a2a.executor.a2a_agent_executor import (
    A2aAgentExecutor,
    A2aAgentExecutorConfig,
)
from google.adk.a2a.executor.config import ExecuteInterceptor
from google.adk.a2a.utils.agent_card_builder import AgentCardBuilder
from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.run_config import RunConfig, StreamingMode
from google.genai import types as genai_types

from app.agent import (
    A2UI_V09_CATALOG_ID,
    STARTER_PROMPTS_LIST,
)
from app.oauth_bq_mcp import (
    current_oauth_token_var,
    extract_candidate_user_token,
    is_entra_jwt,
    is_google_oauth_access_token,
    resolve_google_oauth_userinfo,
)
from app.observability import log_structured_event, scrub_pii

if TYPE_CHECKING:
    from fastapi import FastAPI
    from google.adk.agents import BaseAgent
    from google.adk.events.event import Event
    from google.adk.runners import Runner

_ADK_AGENT_EXECUTOR_EXTENSION_URI = (
    "https://google.github.io/adk-docs/a2a/a2a-extension/"
)
_STARTER_PROMPTS_EXTENSION_URI = (
    "https://www.googleapis.com/gemini-enterprise/a2a/extensions/starter_prompts/v1"
)
_AUTHORIZATIONS_EXTENSION_URI = (
    "https://www.googleapis.com/gemini-enterprise/a2a/extensions/authorizations/v1"
)
_A2UI_V09_EXTENSION_URI = "https://a2ui.org/a2a-extension/a2ui/v0.9"
_A2UI_MIME_TYPE = "application/json+a2ui"


def _add_v0_3_compat_interface(card: AgentCard) -> AgentCard:
    """Advertise a v0.3 JSON-RPC interface for Gemini Enterprise registration."""
    if card.supported_interfaces and not any(
        i.protocol_binding == "JSONRPC" and i.protocol_version == "0.3"
        for i in card.supported_interfaces
    ):
        card.supported_interfaces.append(
            AgentInterface(
                protocol_binding="JSONRPC",
                protocol_version="0.3",
                url=card.supported_interfaces[0].url,
            )
        )
    return card


def _default_capabilities() -> AgentCapabilities:
    """Returns default A2A capabilities with ADK v2, starter_prompts/v1, and A2UI v0.9."""
    return AgentCapabilities(
        streaming=True,
        extensions=[
            AgentExtension(
                uri=_ADK_AGENT_EXECUTOR_EXTENSION_URI,
                description="Ability to use the new agent executor implementation",
            ),
            AgentExtension(
                uri=_STARTER_PROMPTS_EXTENSION_URI,
                description="Gemini Enterprise A2A starter prompts extension",
                params={"prompts": list(STARTER_PROMPTS_LIST)},
            ),
            AgentExtension(
                uri=_AUTHORIZATIONS_EXTENSION_URI,
                description="Google Gemini Enterprise authorization extension",
                params={},
            ),
            AgentExtension(
                uri=_A2UI_V09_EXTENSION_URI,
                description="Ability to render A2UI v0.9",
                params={
                    "supportedCatalogIds": [A2UI_V09_CATALOG_ID],
                    "acceptsInlineCatalogs": True,
                },
            ),
        ],
    )


def _set_repeated_strings(field: object, values: list[str]) -> None:
    """Replace contents of a protobuf repeated string field in-place."""
    del field[:]  # type: ignore[index]
    field.extend(values)  # type: ignore[attr-defined]


def _enrich_a2ui_agent_card(card: AgentCard) -> AgentCard:
    """Populate description, input/output modes, skill examples, and extensions."""
    card.description = (
        "A2UI-BQ Gemini Enterprise FinOps & Usage Pulse Agent (A2UI v0.9 + BigQuery MCP + End-User OAuth). "
        "Features <10ms instant A2UI v0.9 MaterialCard greeting, 3-step thinking timelines, "
        "and interactive Vega-Lite & GcbpTable visualizations in Inline MaterialTabs and Side-Panel Canvas."
    )
    _set_repeated_strings(
        card.default_input_modes,
        ["text/plain", "application/json", _A2UI_MIME_TYPE, "application/a2ui+json"],
    )
    _set_repeated_strings(
        card.default_output_modes,
        ["text/plain", "application/json", _A2UI_MIME_TYPE, "application/a2ui+json"],
    )
    if card.skills:
        _set_repeated_strings(card.skills[0].examples, list(STARTER_PROMPTS_LIST))
    return card


def _a2ui_genai_part_to_a2a_part(part: genai_types.Part) -> A2APart | None:
    """Convert `<a2a_datapart_json>` inline blobs containing A2UI payloads into
    A2A `DataPart`s with `metadata={'mimeType': 'application/json+a2ui'}`."""
    if (
        part.inline_data
        and part.inline_data.mime_type == A2A_DATA_PART_TEXT_MIME_TYPE
        and part.inline_data.data is not None
        and part.inline_data.data.startswith(A2A_DATA_PART_START_TAG)
        and part.inline_data.data.endswith(A2A_DATA_PART_END_TAG)
    ):
        raw_json = part.inline_data.data[
            len(A2A_DATA_PART_START_TAG) : -len(A2A_DATA_PART_END_TAG)
        ]
        try:
            data_dict = json.loads(raw_json.decode("utf-8"))
            if isinstance(data_dict, dict) and any(
                k in data_dict
                for k in (
                    "createSurface",
                    "updateComponents",
                    "updateDataModel",
                    "beginRendering",
                    "surfaceUpdate",
                    "dataModelUpdate",
                    "deleteSurface",
                )
            ):
                return _compat.make_data_part(
                    data=data_dict,
                    metadata={"mimeType": _A2UI_MIME_TYPE},
                )
        except Exception:
            pass
    return convert_genai_part_to_a2a_part(part)


def _decode_jwt_payload_unverified(token: str) -> dict[str, Any]:
    """Decode the middle base64url payload segment of a JWT (if present)."""
    raw = token.strip()
    if raw.lower().startswith("bearer "):
        raw = raw[7:].strip()
    parts = raw.split(".")
    if len(parts) < 2:
        return {}
    payload_b64 = parts[1]
    padding = "=" * (-len(payload_b64) % 4)
    try:
        decoded = base64.urlsafe_b64decode(payload_b64 + padding).decode("utf-8")
        data = json.loads(decoded)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _extract_caller_identity(
    request: RequestContext,
    default_user_id: str | None,
) -> dict[str, Any]:
    """Extract human or service caller identity from A2A RequestContext, headers, and JWT."""
    a2a_user_id = default_user_id or f"A2A_USER_{request.context_id}"
    call_user = (
        request.call_context.user
        if request.call_context and request.call_context.user
        else None
    )
    authenticated_user = (
        call_user.user_name
        if call_user and call_user.is_authenticated and call_user.user_name
        else None
    )

    headers: dict[str, str] = {}
    if request.call_context and isinstance(request.call_context.state, dict):
        raw_headers = request.call_context.state.get("headers")
        if isinstance(raw_headers, dict):
            headers = {str(k).lower(): str(v) for k, v in raw_headers.items()}

    header_user = (
        headers.get("x-goog-authenticated-user-email")
        or headers.get("x-forwarded-user")
        or headers.get("x-user-email")
        or headers.get("x-user-name")
        or headers.get("x-end-user-id")
    )
    if header_user and ":" in header_user:
        header_user = header_user.split(":", 1)[-1]

    jwt_claims: dict[str, Any] = {}
    for hdr_key in (
        "x-forwarded-authorization",
        "x-goog-iap-jwt-assertion",
        "authorization",
    ):
        if headers.get(hdr_key):
            jwt_claims = _decode_jwt_payload_unverified(headers[hdr_key])
            if jwt_claims:
                break

    jwt_email = jwt_claims.get("email") or jwt_claims.get("upn")
    jwt_name = (
        jwt_claims.get("name")
        or jwt_claims.get("given_name")
        or jwt_claims.get("preferred_username")
    )
    is_service_account = bool(
        isinstance(jwt_email, str) and jwt_email.endswith(".gserviceaccount.com")
    )
    human_jwt_user = (
        (jwt_name or jwt_email) if not is_service_account else None
    )

    req_meta = _compat.meta_to_dict(request.metadata) or {}
    msg_meta = (
        _compat.meta_to_dict(request.message.metadata)
        if request.message and request.message.metadata
        else {}
    )
    combined_meta = {**req_meta, **msg_meta}
    meta_user = None
    for k in (
        "user",
        "userName",
        "user_name",
        "displayName",
        "display_name",
        "email",
        "userEmail",
        "user_email",
        "userId",
        "user_id",
    ):
        val = combined_meta.get(k)
        if isinstance(val, str) and val.strip():
            meta_user = val.strip()
            break

    short_a2a_id = (
        a2a_user_id[:17] if a2a_user_id.startswith("A2A_USER_") else a2a_user_id
    )

    # Check for end-user OAuth 2.0 token (Google `ya29.*` or Entra WIF JWT)
    req_state = (
        request.call_context.state
        if request.call_context and isinstance(request.call_context.state, dict)
        else {}
    )
    user_oauth_token = extract_candidate_user_token(
        headers=headers,
        state=req_state,
        metadata=combined_meta,
    )
    oauth_user_email: str | None = None
    oauth_user_name: str | None = None
    auth_mode = "Cloud Run IAM (Service Agent)"
    if user_oauth_token:
        current_oauth_token_var.set(user_oauth_token)
        if is_google_oauth_access_token(user_oauth_token):
            auth_mode = "End-User Google OAuth 2.0"
            uinfo = resolve_google_oauth_userinfo(user_oauth_token)
            oauth_user_email = uinfo.get("email")
            oauth_user_name = uinfo.get("name") or uinfo.get("given_name")
        elif is_entra_jwt(user_oauth_token):
            auth_mode = "End-User Entra WIF (STS)"
            eclaims = _decode_jwt_payload_unverified(user_oauth_token)
            oauth_user_email = (
                eclaims.get("email")
                or eclaims.get("preferred_username")
                or eclaims.get("upn")
            )
            oauth_user_name = eclaims.get("name") or eclaims.get("given_name")

    human_user = (
        oauth_user_email
        or oauth_user_name
        or authenticated_user
        or header_user
        or meta_user
        or human_jwt_user
    )
    display_user = oauth_user_name or oauth_user_email or human_user or short_a2a_id

    identity_info = {
        "display_user": display_user,
        "human_user": human_user,
        "a2a_user_id": a2a_user_id,
        "auth_principal": oauth_user_email or jwt_email,
        "auth_mode": auth_mode,
        "has_user_oauth_token": bool(user_oauth_token),
        "user_oauth_token": user_oauth_token,
        "is_service_account": is_service_account and not bool(user_oauth_token),
        "header_keys": sorted(
            k
            for k in headers
            if k not in {"authorization", "x-forwarded-authorization"}
        ),
        "meta_keys": sorted(combined_meta.keys()),
    }
    log_copy = scrub_pii(
        {k: v for k, v in identity_info.items() if k != "user_oauth_token"}
    )
    log_structured_event(
        "A2A_CALLER_IDENTITY_RESOLVED",
        "Resolved and scrubbed A2A caller identity",
        component="a2a",
        structured_data=log_copy,
    )
    return identity_info


def _sse_request_converter(
    request: RequestContext,
    part_converter: A2APartToGenAIPartConverter = convert_a2a_part_to_genai_part,
) -> AgentRunRequest:
    """Enable `StreamingMode.SSE` and attach resolved caller identity."""
    run_req = convert_a2a_request_to_agent_run_request(request, part_converter)
    caller_identity = _extract_caller_identity(request, run_req.user_id)
    if caller_identity.get("human_user"):
        run_req.user_id = str(caller_identity["human_user"])

    if run_req.state_delta is None:
        run_req.state_delta = {}
    run_req.state_delta["caller_identity"] = caller_identity

    if run_req.run_config is None:
        run_req.run_config = RunConfig(
            streaming_mode=StreamingMode.SSE,
            custom_metadata={"caller_identity": caller_identity},
        )
    else:
        run_req.run_config.streaming_mode = StreamingMode.SSE
        if run_req.run_config.custom_metadata is None:
            run_req.run_config.custom_metadata = {}
        run_req.run_config.custom_metadata["caller_identity"] = caller_identity
    return run_req


def _format_tool_part_as_thought_text(
    part: genai_types.Part, author: str
) -> genai_types.Part | None:
    """Convert `function_call` / `function_response` into `Part(text=..., thought=True)`."""
    if part.function_call:
        fc = part.function_call
        name = fc.name or "tool"
        args = fc.args or {}
        args_str = ", ".join(f"{k}={v!r}" for k, v in args.items())
        return genai_types.Part(
            text=f"🔧 **[Tool Call]** `{author}` invoking `{name}({args_str})`...\n\n",
            thought=True,
        )
    if part.function_response:
        fr = part.function_response
        name = fr.name or "tool"
        return genai_types.Part(
            text=f"📥 **[Tool Result]** `{name}` completed and returned data to `{author}`.\n\n",
            thought=True,
        )
    return None


def _is_a2ui_blob_part(part: genai_types.Part) -> bool:
    """Return True if the GenAI Part is an `<a2a_datapart_json>` inline blob."""
    return bool(
        part.inline_data
        and part.inline_data.data
        and part.inline_data.data.startswith(A2A_DATA_PART_START_TAG)
        and part.inline_data.data.endswith(A2A_DATA_PART_END_TAG)
    )


def _split_event_into_thought_and_answer_events(
    event: Event,
    streamed_thought_keys: set[tuple[str, str]],
    streamed_answer_keys: set[tuple[str, str]],
) -> list[tuple[Event, bool]]:
    """Split an ADK `Event` into pure thought-only and pure answer/A2UI `Event`s."""
    if not event or not event.content or not event.content.parts:
        return [(event, False)]

    author = event.author or "agent"
    inv_key = (event.invocation_id or "", author)
    is_partial = bool(event.partial)

    native_thought_parts: list[genai_types.Part] = []
    tool_thought_parts: list[genai_types.Part] = []
    answer_parts: list[genai_types.Part] = []
    a2ui_parts: list[genai_types.Part] = []

    for part in event.content.parts:
        tool_thought = _format_tool_part_as_thought_text(part, author=author)
        if tool_thought is not None:
            tool_thought_parts.append(tool_thought)
        elif _is_a2ui_blob_part(part):
            a2ui_parts.append(part)
        elif getattr(part, "thought", False):
            if part.text:
                native_thought_parts.append(
                    genai_types.Part(text=part.text, thought=True)
                )
        elif part.text:
            answer_parts.append(genai_types.Part(text=part.text, thought=False))

    if is_partial:
        if native_thought_parts:
            streamed_thought_keys.add(inv_key)
        if answer_parts:
            streamed_answer_keys.add(inv_key)
    else:
        if inv_key in streamed_thought_keys:
            native_thought_parts = []
        if inv_key in streamed_answer_keys:
            answer_parts = []

    all_thought_parts = native_thought_parts + tool_thought_parts
    all_output_parts = answer_parts + a2ui_parts
    results: list[tuple[Event, bool]] = []

    if all_thought_parts:
        thought_event = event.model_copy(deep=True)
        thought_event.content = genai_types.Content(
            role="model",
            parts=all_thought_parts,
        )
        thought_event.partial = True
        results.append((thought_event, True))

    if all_output_parts:
        answer_event = event.model_copy(deep=True)
        answer_event.content = genai_types.Content(
            role="model",
            parts=all_output_parts,
        )
        results.append((answer_event, False))

    return results


class _GEV2A2UIEventConverter:
    """Custom `adk_event_converter` for A2UI v0.9 + GE thinking timeline."""

    def __init__(self) -> None:
        self._streamed_thought_keys: set[tuple[str, str]] = set()
        self._streamed_answer_keys: set[tuple[str, str]] = set()
        self._seen_artifacts: set[str] = set()

    def __call__(
        self,
        event: Event,
        agents_artifacts: dict[str, str] | None,
        task_id: str | None = None,
        context_id: str | None = None,
        part_converter: GenAIPartToA2APartConverter = _a2ui_genai_part_to_a2a_part,
    ) -> list[A2AUpdateEvent]:
        if agents_artifacts is None:
            agents_artifacts = {}

        split_events = _split_event_into_thought_and_answer_events(
            event,
            self._streamed_thought_keys,
            self._streamed_answer_keys,
        )

        out: list[A2AUpdateEvent] = []
        for sub_event, is_thought in split_events:
            if not sub_event.content or not sub_event.content.parts:
                continue
            orig_author = sub_event.author or "agent"
            artifact_key = (
                f"{orig_author}__thought" if is_thought else f"{orig_author}__answer"
            )
            if artifact_key in agents_artifacts:
                agents_artifacts[orig_author] = agents_artifacts[artifact_key]
            else:
                agents_artifacts.pop(orig_author, None)

            converted = convert_event_v2(
                sub_event,
                agents_artifacts,
                task_id,
                context_id,
                _a2ui_genai_part_to_a2a_part,
            )

            if orig_author in agents_artifacts:
                agents_artifacts[artifact_key] = agents_artifacts[orig_author]
            else:
                agents_artifacts.pop(artifact_key, None)

            for a2a_ev in converted:
                if isinstance(a2a_ev, TaskArtifactUpdateEvent):
                    aid = a2a_ev.artifact.artifact_id
                    if aid not in self._seen_artifacts:
                        self._seen_artifacts.add(aid)
                        a2a_ev.append = False
                    else:
                        a2a_ev.append = True
                out.append(a2a_ev)

        return out


async def _activate_a2ui_extension(context: RequestContext) -> RequestContext:
    """Activate A2UI v0.9 extension when requested by Gemini Enterprise."""
    if _A2UI_V09_EXTENSION_URI in context.requested_extensions:
        _compat.add_activated_extension(context, _A2UI_V09_EXTENSION_URI)
    return context


async def attach_a2a_routes(
    app: FastAPI,
    *,
    agent: BaseAgent,
    runner: Runner,
    task_store: TaskStore,
    rpc_path: str,
    capabilities: AgentCapabilities | None = None,
    agent_version: str | None = None,
    app_url: str | None = None,
) -> None:
    """Register A2A + A2UI v0.9 routes under ``rpc_path``."""
    resolved_app_url = app_url or os.getenv("APP_URL", "http://0.0.0.0:8000")
    resolved_agent_version = agent_version or os.getenv("AGENT_VERSION", "0.2.0")
    resolved_capabilities = capabilities or _default_capabilities()

    agent_card = await AgentCardBuilder(
        agent=agent,
        capabilities=resolved_capabilities,
        rpc_url=f"{resolved_app_url}{rpc_path}",
        agent_version=resolved_agent_version,
    ).build()

    agent_card = _enrich_a2ui_agent_card(agent_card)
    agent_card = _add_v0_3_compat_interface(agent_card)

    config = A2aAgentExecutorConfig(
        request_converter=_sse_request_converter,
        gen_ai_part_converter=_a2ui_genai_part_to_a2a_part,
        adk_event_converter=_GEV2A2UIEventConverter(),
        execute_interceptors=[
            ExecuteInterceptor(before_agent=_activate_a2ui_extension)
        ],
    )
    executor = A2aAgentExecutor(
        runner=runner,
        config=config,
        force_new_version=True,
    )

    request_handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=task_store,
        agent_card=agent_card,
    )

    add_a2a_routes_to_fastapi(
        app,
        agent_card_routes=create_agent_card_routes(
            agent_card,
            card_url=f"{rpc_path}{AGENT_CARD_WELL_KNOWN_PATH}",
        ),
        jsonrpc_routes=create_jsonrpc_routes(
            request_handler,
            rpc_url=rpc_path,
            enable_v0_3_compat=True,
        ),
    )
