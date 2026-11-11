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

import contextlib
import os
from collections.abc import AsyncIterator

from a2a.server.tasks import InMemoryTaskStore
from dotenv import load_dotenv
from fastapi import FastAPI
from google.adk.cli.fast_api import get_fast_api_app
from google.adk.runners import Runner

from app.app_utils import services
from app.app_utils.a2a import attach_a2a_routes
from app.oauth_bq_mcp import (
    current_oauth_token_var,
    is_entra_jwt,
    is_google_oauth_access_token,
    is_service_account_oidc_jwt,
)
from app.observability import (
    Feedback,
    configure_structured_logging,
    log_structured_event,
    redact_pii_text,
    trace_span,
)

load_dotenv()
configure_structured_logging()
allow_origins = (
    os.getenv("ALLOW_ORIGINS", "").split(",") if os.getenv("ALLOW_ORIGINS") else None
)
otel_to_cloud = True

AGENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class OAuthCaptureMiddleware:
    """Pure-ASGI middleware that captures end-user OAuth bearer tokens (`ya29.*` or Entra JWT)
    from incoming HTTP headers into `current_oauth_token_var` while ignoring Cloud Run's
    `gcp-sa-discoveryengine` IAM invoker OIDC token."""

    def __init__(self, app: object) -> None:
        self.app = app

    async def __call__(self, scope: dict, receive: object, send: object) -> None:
        token: str | None = None
        if scope.get("type") == "http":
            for raw_name, raw_val in scope.get("headers", []):
                name = raw_name.decode("latin-1").lower()
                if name in (
                    "authorization",
                    "x-forwarded-authorization",
                    "x-goog-oauth-token",
                    "x-vertex-ai-oauth-token",
                ):
                    val = raw_val.decode("latin-1").strip()
                    if val.lower().startswith("bearer "):
                        val = val[7:].strip()
                    if is_google_oauth_access_token(val) or is_entra_jwt(val):
                        token = val
                        break
                    if val and not is_service_account_oidc_jwt(val):
                        token = val
        reset = current_oauth_token_var.set(token)
        try:
            if scope.get("type") == "http":
                path = scope.get("path", "")
                method = scope.get("method", "GET")
                with trace_span(
                    "a2ui_bq.http.request",
                    attributes={
                        "http.method": method,
                        "http.route": path,
                        "oauth.has_user_token": bool(token),
                    },
                ):
                    await self.app(scope, receive, send)  # type: ignore[misc]
            else:
                await self.app(scope, receive, send)  # type: ignore[misc]
        finally:
            current_oauth_token_var.reset(reset)


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    from app.agent import app as adk_app
    from app.agent import root_agent

    runner = Runner(
        app=adk_app,
        session_service=services.get_session_service(),
        artifact_service=services.get_artifact_service(),
        memory_service=services.get_memory_service(),
        auto_create_session=True,
    )
    app.state.runner = runner
    app.state.agent_app_name = adk_app.name
    await attach_a2a_routes(
        app,
        agent=root_agent,
        runner=runner,
        task_store=InMemoryTaskStore(),
        rpc_path=f"/a2a/{adk_app.name}",
    )
    yield


app: FastAPI = get_fast_api_app(
    agents_dir=AGENT_DIR,
    web=True,
    artifact_service_uri=services.ARTIFACT_SERVICE_URI,
    memory_service_uri=services.MEMORY_SERVICE_URI,
    allow_origins=allow_origins,
    session_service_uri=services.SESSION_SERVICE_URI,
    otel_to_cloud=otel_to_cloud,
    lifespan=lifespan,
)
app.title = "a2ui-bq-agent"
app.description = "API for interacting with the A2UI-BQ Gemini Enterprise FinOps Agent"
app.add_middleware(OAuthCaptureMiddleware)


@app.post("/feedback")
def collect_feedback(feedback: Feedback) -> dict[str, str]:
    """Collect and log PII-redacted user feedback to Cloud Logging."""
    sanitized = feedback.model_copy(
        update={"text": redact_pii_text(feedback.text or "")}
    )
    log_structured_event(
        "USER_FEEDBACK",
        f"Received feedback score={sanitized.score}",
        component="feedback",
        structured_data=sanitized.model_dump(),
    )
    return {"status": "success"}


# Main execution
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
