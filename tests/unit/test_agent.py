# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for the A2UI-BQ Gemini Enterprise FinOps & Usage Pulse agent."""

import json
import time

from a2a.types import TaskArtifactUpdateEvent, TaskStatusUpdateEvent
from fastapi.testclient import TestClient
from google.adk.a2a.converters.part_converter import (
    A2A_DATA_PART_END_TAG,
    A2A_DATA_PART_START_TAG,
)
from google.adk.events.event import Event
from google.genai import types as genai_types
from google.protobuf.json_format import MessageToDict

from app.agent import (
    A2UI_V09_CATALOG_ID,
    STARTER_PROMPTS_LIST,
    _build_daily_usage_dashboard_payload,
    _build_token_finops_dashboard_payload,
    build_welcome_a2ui_messages,
    make_a2ui_genai_parts,
    root_agent,
)
from app.app_utils.a2a import (
    _A2UI_MIME_TYPE,
    _A2UI_V09_EXTENSION_URI,
    _AUTHORIZATIONS_EXTENSION_URI,
    _STARTER_PROMPTS_EXTENSION_URI,
    _a2ui_genai_part_to_a2a_part,
    _GEV2A2UIEventConverter,
)
from app.fast_api_app import app
from app.oauth_bq_mcp import extract_candidate_user_token


def test_agent_card_includes_a2ui_and_starter_prompts() -> None:
    """Verify the AgentCard exposes A2UI v0.9 and starter_prompts/v1 extensions."""
    with TestClient(app) as client:
        resp = client.get("/a2a/app/.well-known/agent-card.json")
        assert resp.status_code == 200
        card = resp.json()

        assert _A2UI_MIME_TYPE in card.get("defaultOutputModes", [])

        extensions = card.get("capabilities", {}).get("extensions", [])
        ext_by_uri = {ext["uri"]: ext for ext in extensions}

        assert (
            "https://google.github.io/adk-docs/a2a/a2a-extension/"
            in ext_by_uri
        )
        assert _STARTER_PROMPTS_EXTENSION_URI in ext_by_uri
        assert _AUTHORIZATIONS_EXTENSION_URI in ext_by_uri
        assert ext_by_uri[_STARTER_PROMPTS_EXTENSION_URI]["params"]["prompts"] == (
            STARTER_PROMPTS_LIST
        )

        assert _A2UI_V09_EXTENSION_URI in ext_by_uri
        a2ui_params = ext_by_uri[_A2UI_V09_EXTENSION_URI]["params"]
        assert A2UI_V09_CATALOG_ID in a2ui_params["supportedCatalogIds"]
        assert a2ui_params.get("acceptsInlineCatalogs") is True


def test_instant_greeting_latency_and_a2ui_datapart_conversion() -> None:
    """Verify Path 1 (instant greeting) runs in <50ms and emits ONLY A2UI DataParts."""
    converter = _GEV2A2UIEventConverter()

    t0 = time.perf_counter()
    parts = make_a2ui_genai_parts(build_welcome_a2ui_messages())
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert elapsed_ms < 50.0

    event = Event(
        author=root_agent.name,
        content=genai_types.Content(role="model", parts=parts),
    )
    a2a_events = converter(
        event,
        {},
        "task-greeting-1",
        "ctx-greeting-1",
        _a2ui_genai_part_to_a2a_part,
    )
    assert len(a2a_events) == 1
    assert isinstance(a2a_events[0], TaskArtifactUpdateEvent)

    artifact_parts = a2a_events[0].artifact.parts
    data_parts = [p for p in artifact_parts if p.HasField("data")]
    text_parts = [p for p in artifact_parts if p.HasField("text")]

    assert len(data_parts) == 2
    assert len(text_parts) == 0
    for dp in data_parts:
        assert dict(dp.metadata).get("mimeType") == _A2UI_MIME_TYPE
    dp0_dict = MessageToDict(data_parts[0].data)
    dp1_dict = MessageToDict(data_parts[1].data)
    assert dp0_dict["version"] == "v0.9"
    assert dp0_dict["createSurface"]["surfaceId"] == "agent-welcome"
    assert dp1_dict["updateComponents"]["surfaceId"] == "agent-welcome"


def test_v03_message_stream_multi_turn_greeting_unique_surface_ids() -> None:
    """Verify Turn 1 ('') uses 'agent-welcome' and Turn 2 ('Show the A2UI...') uses a unique surfaceId with 0 TextParts."""
    turn1_payload = {
        "jsonrpc": "2.0",
        "id": "test-ge-greeting-1",
        "method": "message/stream",
        "params": {
            "message": {
                "messageId": "msg-greeting-1",
                "contextId": "ctx-multi-turn-1",
                "role": "user",
                "parts": [{"kind": "text", "text": ""}],
            },
            "metadata": {
                "a2uiClientCapabilities": {
                    "v0.9": {
                        "supportedCatalogIds": [A2UI_V09_CATALOG_ID],
                    }
                },
                "X-A2A-Extensions": _A2UI_V09_EXTENSION_URI,
            },
        },
    }
    turn2_payload = {
        "jsonrpc": "2.0",
        "id": "test-ge-greeting-2",
        "method": "message/stream",
        "params": {
            "message": {
                "messageId": "msg-greeting-2",
                "contextId": "ctx-multi-turn-1",
                "role": "user",
                "parts": [
                    {
                        "kind": "text",
                        "text": "Show the A2UI v0.9 welcome view and starter prompt cards",
                    }
                ],
            },
        },
    }
    with TestClient(app) as client:
        resp1 = client.post("/a2a/app", json=turn1_payload)
        assert resp1.status_code == 200
        sse1 = [
            json.loads(line[len("data: ") :])
            for line in resp1.text.splitlines()
            if line.startswith("data: ")
        ]
        artifacts1 = [
            ev["result"]
            for ev in sse1
            if ev.get("result", {}).get("kind") == "artifact-update"
        ]
        assert len(artifacts1) == 1
        parts1 = artifacts1[0]["artifact"]["parts"]
        assert len(parts1) == 2
        assert all(p.get("kind") == "data" for p in parts1)
        assert parts1[0]["data"]["createSurface"]["surfaceId"] == "agent-welcome"
        comps1 = {
            c["id"]: c for c in parts1[1]["data"]["updateComponents"]["components"]
        }
        assert "Hello " in comps1["greeting_badge"]["text"]
        assert not comps1["title"]["text"].startswith("Hello ")
        assert not comps1["description"]["text"].startswith("Hello ")

        resp2 = client.post("/a2a/app", json=turn2_payload)
        assert resp2.status_code == 200
        sse2 = [
            json.loads(line[len("data: ") :])
            for line in resp2.text.splitlines()
            if line.startswith("data: ")
        ]
        artifacts2 = [
            ev["result"]
            for ev in sse2
            if ev.get("result", {}).get("kind") == "artifact-update"
        ]
        assert len(artifacts2) == 1
        parts2 = artifacts2[0]["artifact"]["parts"]
        assert len(parts2) == 2
        assert all(p.get("kind") == "data" for p in parts2)
        surf2 = parts2[0]["data"]["createSurface"]["surfaceId"]
        assert surf2.startswith("agent-welcome-")
        assert surf2 != "agent-welcome"
        assert parts2[1]["data"]["updateComponents"]["surfaceId"] == surf2


def test_synthetic_thinking_separation_from_a2ui_payload() -> None:
    """Verify synthetic thoughts and A2UI payloads are never mixed in the same Event."""
    converter = _GEV2A2UIEventConverter()

    a2ui_msg = {
        "version": "v0.9",
        "createSurface": {
            "surfaceId": "agent-diagnostic-report",
            "catalogId": A2UI_V09_CATALOG_ID,
        },
    }
    raw_bytes = (
        A2A_DATA_PART_START_TAG
        + json.dumps(a2ui_msg).encode("utf-8")
        + A2A_DATA_PART_END_TAG
    )

    mixed_event = Event(
        author=root_agent.name,
        content=genai_types.Content(
            role="model",
            parts=[
                genai_types.Part(
                    text="**[Step 1/3 — A2UI Surface Allocator]** Initializing...",
                    thought=True,
                ),
                genai_types.Part(
                    inline_data=genai_types.Blob(
                        mime_type="text/plain",
                        data=raw_bytes,
                    )
                ),
                genai_types.Part(
                    text="Diagnostic complete.",
                    thought=False,
                ),
            ],
        ),
    )

    a2a_events = converter(
        mixed_event,
        {},
        "task-diag-1",
        "ctx-diag-1",
        _a2ui_genai_part_to_a2a_part,
    )

    # Must produce 2 separate TaskArtifactUpdateEvents:
    # 1. Thought-only TextPart with adk_thought=True
    # 2. Answer-only event with A2UI DataPart + companion TextPart
    assert len(a2a_events) == 2
    thought_evt, answer_evt = a2a_events[0], a2a_events[1]
    assert isinstance(thought_evt, TaskArtifactUpdateEvent)
    assert isinstance(answer_evt, TaskArtifactUpdateEvent)
    assert thought_evt.artifact.artifact_id != answer_evt.artifact.artifact_id

    assert len(thought_evt.artifact.parts) == 1
    t_part = thought_evt.artifact.parts[0]
    assert t_part.HasField("text")
    assert dict(t_part.metadata).get("adk_thought") is True

    ans_parts = answer_evt.artifact.parts
    assert all(not dict(p.metadata).get("adk_thought") for p in ans_parts)
    assert any(
        p.HasField("data") and dict(p.metadata).get("mimeType") == _A2UI_MIME_TYPE
        for p in ans_parts
    )


def test_inline_tabs_and_side_panel_canvas_a2ui_surfaces() -> None:
    """Verify both Inline MaterialTabs and Side-Panel Canvas FinOps surfaces emit
    createSurface + updateDataModel (/vegaSpec) + updateComponents DataParts."""
    converter = _GEV2A2UIEventConverter()
    mock_bq = {
        "rows": [
            {
                "agent_name": "root_agent",
                "inference_calls": 85,
                "input_tokens": 3191204,
                "cached_tokens": 1245900,
                "output_tokens": 20440,
                "reasoning_tokens": 8606,
                "date": "2026-03-10",
                "searches": 1,
                "answers": 41,
                "actions": 37,
                "dau": 1,
            }
        ],
        "row_count": 1,
        "query_id": "job-finops-1",
        "total_bytes_processed": "10485760",
        "total_slot_ms": "77",
        "elapsed_ms": 95.0,
        "auth_mode": "End-User Google OAuth 2.0",
        "principal": "analyst@example.com",
        "sql": "SELECT * FROM usage_metrics",
    }

    # 1. Inline MaterialTabs surface
    _, tabs_msgs = _build_token_finops_dashboard_payload(
        mock_bq, surface_id="finops-tabs-1", use_canvas=False
    )
    tabs_parts = make_a2ui_genai_parts(tabs_msgs)
    tabs_ev = Event(
        author=root_agent.name,
        content=genai_types.Content(role="model", parts=tabs_parts),
    )
    a2a_tabs = converter(
        tabs_ev, {}, "t-tabs", "c-tabs", _a2ui_genai_part_to_a2a_part
    )
    assert len(a2a_tabs) == 1
    t_parts = a2a_tabs[0].artifact.parts
    assert len(t_parts) == 3
    t_dicts = [MessageToDict(p.data) for p in t_parts]
    assert "createSurface" in t_dicts[0]
    assert "updateDataModel" in t_dicts[1]
    assert "vegaSpec" in t_dicts[1]["updateDataModel"]["value"]
    assert "updateComponents" in t_dicts[2]
    tabs_comps = {
        c["id"]: c for c in t_dicts[2]["updateComponents"]["components"]
    }
    assert tabs_comps["finops_tabs"]["component"] == "MaterialTabs"
    assert tabs_comps["vega_chart_main"]["component"] == "VegaChart"
    assert tabs_comps["vega_chart_main"]["spec"] == {"path": "/vegaSpec"}
    assert tabs_comps["gcbp_table_main"]["component"] == "GcbpTable"

    # 2. Side-Panel Canvas surface
    _, canvas_msgs = _build_daily_usage_dashboard_payload(
        mock_bq, surface_id="finops-canvas-1", use_canvas=True
    )
    canvas_parts = make_a2ui_genai_parts(canvas_msgs)
    canvas_ev = Event(
        author=root_agent.name,
        content=genai_types.Content(role="model", parts=canvas_parts),
    )
    a2a_canvas = converter(
        canvas_ev, {}, "t-canvas", "c-canvas", _a2ui_genai_part_to_a2a_part
    )
    assert len(a2a_canvas) == 1
    c_parts = a2a_canvas[0].artifact.parts
    assert len(c_parts) == 3
    c_dicts = [MessageToDict(p.data) for p in c_parts]
    canvas_comps = {
        c["id"]: c for c in c_dicts[2]["updateComponents"]["components"]
    }
    assert canvas_comps["root"]["component"] == "Canvas"
    assert canvas_comps["root"]["autoOpen"] is True
    assert canvas_comps["canvas_vega_chart"]["component"] == "VegaChart"
    assert canvas_comps["canvas_gcbp_table"]["component"] == "GcbpTable"


def test_oauth_token_extraction() -> None:
    """Verify Google OAuth ya29.* tokens are extracted from headers or state."""
    tok = extract_candidate_user_token(
        headers={"authorization": "Bearer ya29.a0AfH6SMBx-test-user-token"},
    )
    assert tok == "ya29.a0AfH6SMBx-test-user-token"


def test_gcbp_table_fetch_data_sorting_keeps_full_dashboard() -> None:
    """Verify GcbpTable sorting sorts numeric/currency/string cells properly and keeps MaterialTabs activeTab=1."""
    mock_bq = {
        "rows": [
            {
                "agent_name": "root_agent",
                "inference_calls": 85,
                "input_tokens": 3191204,
                "cached_tokens": 1245900,
                "output_tokens": 20440,
                "reasoning_tokens": 8606,
            },
            {
                "agent_name": "alpha_agent",
                "inference_calls": 12,
                "input_tokens": 150000,
                "cached_tokens": 50000,
                "output_tokens": 2000,
                "reasoning_tokens": 500,
            },
        ],
        "row_count": 2,
        "query_id": "job-sort-1",
        "total_bytes_processed": "10485760",
        "total_slot_ms": "77",
        "elapsed_ms": 95.0,
        "auth_mode": "End-User Google OAuth 2.0",
        "principal": "analyst@example.com",
        "sql": "SELECT * FROM usage_metrics",
    }
    _, msgs = _build_token_finops_dashboard_payload(
        mock_bq,
        surface_id="finops-sorted-1",
        use_canvas=False,
        sort_string="calls asc",
        active_tab=1,
    )
    comps = {c["id"]: c for c in msgs[2]["updateComponents"]["components"]}
    assert comps["finops_tabs"]["activeTab"] == 1
    rows = comps["gcbp_table_main"]["rows"]
    assert [r["agent_name"] for r in rows] == ["alpha_agent", "root_agent"]


