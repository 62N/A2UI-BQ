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

"""End-user OAuth 2.0 (Google OAuth / Cloud Identity & Entra WIF) + BigQuery MCP client.

Supports:
1. Google OAuth 2.0 access tokens (`ya29.*`) injected by Gemini Enterprise via
   `serverSideOauth2` (`projects/<PROJECT_NUMBER>/locations/<LOCATION>/authorizations/<AUTH_ID>`).
2. Microsoft Entra ID JWTs exchanged via Google Cloud STS Workforce Identity Federation.
3. User identity resolution (`email`, `name`) via Google OAuth `userinfo` / `tokeninfo`
   (cached per token hash) so the A2UI welcome screen and FinOps dashboards show the
   authenticated end-user identity.
4. Direct JSON-RPC calls and ADK `McpToolset` `header_provider` for
   `https://bigquery.googleapis.com/mcp` (`execute_sql_readonly`, `list_dataset_ids`, etc.).
"""

from __future__ import annotations

import base64
from collections.abc import Callable, Mapping
import contextvars
import hashlib
import json
import logging
import os
import threading
import time
from typing import Any

import google.auth
import google.auth.transport.requests
from google.adk.agents.readonly_context import ReadonlyContext
import requests

logger = logging.getLogger(__name__)

BQ_MCP_URL = "https://bigquery.googleapis.com/mcp"
_STS_URL = "https://sts.googleapis.com/v1/token"
_GOOGLE_USERINFO_URL = "https://openidconnect.googleapis.com/v1/userinfo"
_GOOGLE_TOKENINFO_URL = "https://oauth2.googleapis.com/tokeninfo"
_DEFAULT_SCOPE = "https://www.googleapis.com/auth/cloud-platform"

# Per-request ContextVars populated by ASGI middleware and/or A2A request converter
current_oauth_token_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "current_oauth_token_var", default=None
)
current_user_identity_var: contextvars.ContextVar[dict[str, Any] | None] = (
    contextvars.ContextVar("current_user_identity_var", default=None)
)

_userinfo_cache: dict[str, tuple[dict[str, Any], float]] = {}
_sts_cache: dict[str, tuple[str, float]] = {}
_cache_lock = threading.Lock()


def _decode_jwt_claims(token: str) -> dict[str, Any] | None:
    """Decode JWT payload without signature verification."""
    parts = token.strip().split(".")
    if len(parts) != 3:
        return None
    payload_b64 = parts[1]
    pad = "=" * (-len(payload_b64) % 4)
    try:
        raw = base64.urlsafe_b64decode(payload_b64 + pad)
        data = json.loads(raw.decode("utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def is_google_oauth_access_token(value: Any) -> bool:
    """Return True if value looks like a Google OAuth 2.0 access token (`ya29.*`)."""
    return isinstance(value, str) and value.strip().startswith("ya29.")


def is_entra_jwt(value: Any) -> bool:
    """Return True if value is a JWT issued by Microsoft Entra ID."""
    if not isinstance(value, str) or value.count(".") != 2:
        return False
    claims = _decode_jwt_claims(value)
    if not claims:
        return False
    iss = str(claims.get("iss") or "")
    return "login.microsoftonline.com" in iss or "sts.windows.net" in iss


def is_service_account_oidc_jwt(value: Any) -> bool:
    """Return True if value is a Google OIDC JWT for a GCP service account
    (e.g. Cloud Run IAM invoker token from `gcp-sa-discoveryengine`)."""
    if not isinstance(value, str) or value.count(".") != 2:
        return False
    claims = _decode_jwt_claims(value)
    if not claims:
        return False
    email = str(claims.get("email") or "")
    return email.endswith(".gserviceaccount.com")


def extract_candidate_user_token(
    *,
    headers: Mapping[str, str] | None = None,
    state: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> str | None:
    """Locate the end-user OAuth token from HTTP headers, A2A metadata, or ADK session state."""
    # 1. Check ContextVar first if already captured on this request
    ctx_tok = current_oauth_token_var.get()
    if ctx_tok and (is_google_oauth_access_token(ctx_tok) or is_entra_jwt(ctx_tok)):
        return ctx_tok

    # 2. Scan state and metadata for Google `ya29.*` or Entra JWT
    for mapping in (state, metadata):
        found = _scan_mapping_for_user_token(mapping)
        if found:
            return found

    # 3. Check HTTP Authorization / X-Forwarded-Authorization headers
    if headers:
        for hdr_name in (
            "authorization",
            "x-forwarded-authorization",
            "x-goog-oauth-token",
            "x-vertex-ai-oauth-token",
        ):
            raw_val = headers.get(hdr_name) or headers.get(hdr_name.title())
            if not raw_val:
                continue
            token = raw_val.strip()
            if token.lower().startswith("bearer "):
                token = token[7:].strip()
            if is_google_oauth_access_token(token) or is_entra_jwt(token):
                return token
            # If it's a non-service-account JWT/token, also allow it
            if token and not is_service_account_oidc_jwt(token):
                return token

    return ctx_tok


def _scan_mapping_for_user_token(obj: Any) -> str | None:
    """Recursively scan a dict/list for a `ya29.*` Google OAuth token or Entra JWT."""
    if isinstance(obj, str):
        val = obj.strip()
        if val.lower().startswith("bearer "):
            val = val[7:].strip()
        if is_google_oauth_access_token(val) or is_entra_jwt(val):
            return val
        return None
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            # Skip known internal keys that don't hold tokens
            if k in {"header_keys", "caller_identity"}:
                continue
            found = _scan_mapping_for_user_token(v)
            if found:
                return found
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            found = _scan_mapping_for_user_token(item)
            if found:
                return found
    return None


def resolve_google_oauth_userinfo(access_token: str) -> dict[str, Any]:
    """Resolve user email & display name from a Google OAuth `ya29.*` access token."""
    token_hash = hashlib.sha256(access_token.encode("utf-8")).hexdigest()[:16]
    now = time.time()
    with _cache_lock:
        cached = _userinfo_cache.get(token_hash)
        if cached and cached[1] > now:
            return cached[0]

    info: dict[str, Any] = {}
    try:
        resp = requests.get(
            _GOOGLE_USERINFO_URL,
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=2.5,
        )
        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, dict):
                info.update(data)
    except Exception as exc:
        logger.debug("Google userinfo lookup failed: %s", exc)

    if not info.get("email"):
        try:
            resp2 = requests.get(
                _GOOGLE_TOKENINFO_URL,
                params={"access_token": access_token},
                timeout=2.5,
            )
            if resp2.status_code == 200:
                data2 = resp2.json()
                if isinstance(data2, dict):
                    info.update(data2)
        except Exception as exc:
            logger.debug("Google tokeninfo lookup failed: %s", exc)

    with _cache_lock:
        _userinfo_cache[token_hash] = (info, now + 1800)
    return info


def exchange_entra_jwt_for_google_token(
    entra_jwt: str,
    wif_provider: str,
    user_project: str,
    scope: str = _DEFAULT_SCOPE,
) -> str:
    """Exchange a Microsoft Entra ID JWT for a Google WIF access token via STS."""
    claims = _decode_jwt_claims(entra_jwt) or {}
    subject = str(claims.get("oid") or claims.get("sub") or entra_jwt[:32])
    now = time.time()
    with _cache_lock:
        cached = _sts_cache.get(subject)
        if cached and cached[1] - now > 60:
            return cached[0]

    resp = requests.post(
        _STS_URL,
        data={
            "grant_type": "urn:ietf:params:oauth:grant-type:token-exchange",
            "audience": wif_provider,
            "requested_token_type": "urn:ietf:params:oauth:token-type:access_token",
            "scope": scope,
            "subject_token_type": "urn:ietf:params:oauth:token-type:jwt",
            "subject_token": entra_jwt,
            "options": json.dumps({"userProject": user_project}),
        },
        timeout=10,
    )
    if resp.status_code != 200:
        raise RuntimeError(
            f"STS WIF token exchange failed ({resp.status_code}): {resp.text}"
        )
    body = resp.json()
    token = body["access_token"]
    expires_in = int(body.get("expires_in", 3600))
    with _cache_lock:
        _sts_cache[subject] = (token, now + max(60, expires_in - 60))
    return token


def resolve_bigquery_bearer_token(
    *,
    state: Mapping[str, Any] | None = None,
    user_project: str | None = None,
) -> tuple[str, str, str]:
    """Return `(google_access_token, auth_mode_label, principal_display)`.

    Priority:
    1. End-user Google OAuth 2.0 access token (`ya29.*`) from GE `serverSideOauth2`.
    2. End-user Entra ID JWT exchanged via WIF STS (`WIF_PROVIDER_RESOURCE`).
    3. Fallback to Application Default Credentials (ADC) only if no end-user OAuth
       token was supplied on the request (e.g. local test / unlinked probe).
    """
    project = user_project or os.getenv("GOOGLE_CLOUD_PROJECT", "your-gcp-project-id")
    raw_token = extract_candidate_user_token(state=state)

    if raw_token:
        if is_google_oauth_access_token(raw_token):
            uinfo = resolve_google_oauth_userinfo(raw_token)
            email = str(
                uinfo.get("email")
                or uinfo.get("name")
                or "Authenticated OAuth User"
            )
            return raw_token, "End-User Google OAuth 2.0", email

        if is_entra_jwt(raw_token):
            wif_provider = os.getenv(
                "WIF_PROVIDER_RESOURCE",
                "//iam.googleapis.com/locations/global/workforcePools/m365-pool/providers/bqmcp-agent-provider",
            )
            claims = _decode_jwt_claims(raw_token) or {}
            email = str(
                claims.get("email")
                or claims.get("preferred_username")
                or claims.get("upn")
                or claims.get("name")
                or "Entra WIF User"
            )
            exchanged = exchange_entra_jwt_for_google_token(
                raw_token, wif_provider=wif_provider, user_project=project
            )
            return exchanged, "End-User Entra WIF (STS)", email

    # Check if caller_identity in state already has a user_oauth_token
    if state and isinstance(state.get("caller_identity"), Mapping):
        cid = state["caller_identity"]
        tok = cid.get("user_oauth_token")
        if isinstance(tok, str) and tok:
            email = str(cid.get("human_user") or cid.get("display_user") or "OAuth User")
            return tok, str(cid.get("auth_mode") or "End-User OAuth 2.0"), email

    # Fallback to ADC (for local unit tests or when GE hasn't injected token yet)
    creds, _ = google.auth.default(scopes=[_DEFAULT_SCOPE])
    if not creds.valid:
        creds.refresh(google.auth.transport.requests.Request())
    principal = getattr(creds, "service_account_email", None) or "ADC Service Account"
    return str(creds.token), "Service Account ADC (No OAuth Token Injected)", str(principal)


def bq_mcp_oauth_header_provider(
    user_project: str | None = None,
) -> Callable[[ReadonlyContext], dict[str, str]]:
    """ADK `McpToolset` `header_provider` that authenticates BigQuery MCP calls
    using the end-user's OAuth 2.0 identity."""

    def _provider(ctx: ReadonlyContext) -> dict[str, str]:
        project = user_project or os.getenv("GOOGLE_CLOUD_PROJECT", "your-gcp-project-id")
        state = ctx.state if hasattr(ctx, "state") else {}
        token, auth_mode, principal = resolve_bigquery_bearer_token(
            state=state, user_project=project
        )
        logger.info(
            "BigQuery MCP header_provider: mode=%s principal=%s project=%s",
            auth_mode,
            principal,
            project,
        )
        return {
            "Authorization": f"Bearer {token}",
            "X-Goog-User-Project": project,
        }

    return _provider


def execute_bq_mcp_rpc(
    tool_name: str,
    arguments: dict[str, Any],
    *,
    state: Mapping[str, Any] | None = None,
    project_id: str | None = None,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Invoke any tool (`list_dataset_ids`, `list_table_ids`, `get_table_info`,
    `execute_sql_readonly`) on `https://bigquery.googleapis.com/mcp` using the
    end-user's OAuth 2.0 token."""
    project = project_id or os.getenv("GOOGLE_CLOUD_PROJECT", "your-gcp-project-id")
    token, auth_mode, principal = resolve_bigquery_bearer_token(
        state=state, user_project=project
    )
    headers = {
        "Authorization": f"Bearer {token}",
        "X-Goog-User-Project": project,
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }
    payload = {
        "jsonrpc": "2.0",
        "id": f"bqmcp-{tool_name}-{int(time.time() * 1000)}",
        "method": "tools/call",
        "params": {
            "name": tool_name,
            "arguments": arguments,
        },
    }
    t0 = time.perf_counter()
    resp = requests.post(BQ_MCP_URL, json=payload, headers=headers, timeout=timeout)
    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 1)
    if resp.status_code != 200:
        raise RuntimeError(
            f"BigQuery MCP HTTP {resp.status_code} ({auth_mode} as {principal}): {resp.text[:500]}"
        )
    rpc_body = resp.json()
    if "error" in rpc_body:
        raise RuntimeError(
            f"BigQuery MCP RPC error ({auth_mode} as {principal}): {json.dumps(rpc_body['error'])}"
        )
    result = rpc_body.get("result", {})
    if result.get("isError"):
        err_text = "".join(
            c.get("text", "") for c in result.get("content", []) if isinstance(c, dict)
        )
        raise RuntimeError(
            f"BigQuery MCP tool error ({auth_mode} as {principal}): {err_text or json.dumps(result)}"
        )
    structured = result.get("structuredContent")
    if not structured:
        for item in result.get("content", []):
            if isinstance(item, dict) and item.get("text"):
                try:
                    structured = json.loads(item["text"])
                    break
                except Exception:
                    structured = {"text": item["text"]}
                    break
    return {
        "tool": tool_name,
        "auth_mode": auth_mode,
        "principal": principal,
        "elapsed_ms": elapsed_ms,
        "data": structured or result,
    }


def execute_bq_mcp_sql(
    query: str,
    *,
    state: Mapping[str, Any] | None = None,
    project_id: str | None = None,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Call `https://bigquery.googleapis.com/mcp` (`execute_sql_readonly`) using the
    end-user's OAuth token and return parsed rows + telemetry metadata."""
    project = project_id or os.getenv("GOOGLE_CLOUD_PROJECT", "your-gcp-project-id")
    token, auth_mode, principal = resolve_bigquery_bearer_token(
        state=state, user_project=project
    )

    headers = {
        "Authorization": f"Bearer {token}",
        "X-Goog-User-Project": project,
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }
    payload = {
        "jsonrpc": "2.0",
        "id": f"bqmcp-{int(time.time() * 1000)}",
        "method": "tools/call",
        "params": {
            "name": "execute_sql_readonly",
            "arguments": {
                "project_id": project,
                "query": query,
            },
        },
    }
    t0 = time.perf_counter()
    resp = requests.post(BQ_MCP_URL, json=payload, headers=headers, timeout=timeout)
    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 1)

    if resp.status_code != 200:
        raise RuntimeError(
            f"BigQuery MCP HTTP {resp.status_code} ({auth_mode} as {principal}): {resp.text[:500]}"
        )

    rpc_body = resp.json()
    if "error" in rpc_body:
        err = rpc_body["error"]
        raise RuntimeError(
            f"BigQuery MCP RPC error ({auth_mode} as {principal}): {json.dumps(err)}"
        )

    result = rpc_body.get("result", {})
    if result.get("isError"):
        err_text = ""
        for item in result.get("content", []):
            if isinstance(item, dict) and item.get("text"):
                err_text += item["text"]
        raise RuntimeError(
            f"BigQuery MCP tool error ({auth_mode} as {principal}): {err_text or json.dumps(result)}"
        )

    structured = result.get("structuredContent")
    if not structured:
        for item in result.get("content", []):
            if isinstance(item, dict) and item.get("text"):
                try:
                    structured = json.loads(item["text"])
                    break
                except Exception:
                    pass

    if not isinstance(structured, dict):
        structured = {}

    schema_fields = (
        structured.get("schema", {}).get("fields", [])
        if isinstance(structured.get("schema"), dict)
        else []
    )
    col_names = [str(f.get("name", f"col_{i}")) for i, f in enumerate(schema_fields)]
    raw_rows = structured.get("rows", []) or []

    parsed_rows: list[dict[str, Any]] = []
    for r in raw_rows:
        cells = r.get("f", []) if isinstance(r, dict) else []
        row_dict: dict[str, Any] = {}
        for idx, col in enumerate(col_names):
            val = cells[idx].get("v") if idx < len(cells) and isinstance(cells[idx], dict) else None
            row_dict[col] = val
        parsed_rows.append(row_dict)

    return {
        "rows": parsed_rows,
        "columns": col_names,
        "row_count": len(parsed_rows),
        "query_id": str(structured.get("queryId") or "n/a"),
        "total_bytes_processed": str(structured.get("totalBytesProcessed") or "0"),
        "total_bytes_billed": str(structured.get("totalBytesBilled") or "0"),
        "total_slot_ms": str(structured.get("totalSlotMs") or "0"),
        "elapsed_ms": elapsed_ms,
        "auth_mode": auth_mode,
        "principal": principal,
        "sql": query.strip(),
    }
