#!/usr/bin/env python3
"""Register and sync the A2UI v0.9 FinOps & Usage Pulse agent on Gemini Enterprise with End-User Google OAuth."""

import json
import os
import subprocess
import urllib.error
import urllib.request

from dotenv import load_dotenv
from fastapi.testclient import TestClient

from app.agent import STARTER_PROMPTS_LIST
from app.fast_api_app import app

load_dotenv()

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "your-gcp-project-id")
PROJECT_NUM = os.getenv("PROJECT_NUMBER", "000000000000")
GE_LOCATION = os.getenv("GE_LOCATION", "global")
GE_ENGINE_ID = os.getenv("GE_ENGINE_ID", "your-ge-engine-id")
OAUTH_AUTH_ID = os.getenv("OAUTH_AUTH_ID", "finops-google-oauth")
APP_URL = os.getenv("APP_URL", "https://your-cloud-run-service.run.app")

DISCOVERY_HOST = (
    "https://discoveryengine.googleapis.com"
    if GE_LOCATION == "global"
    else f"https://{GE_LOCATION}-discoveryengine.googleapis.com"
)
ENGINE_PARENT = (
    f"projects/{PROJECT_NUM}/locations/{GE_LOCATION}/collections/default_collection/"
    f"engines/{GE_ENGINE_ID}/assistants/default_assistant"
)
OAUTH_AUTHORIZATION = (
    f"projects/{PROJECT_NUM}/locations/{GE_LOCATION}/authorizations/{OAUTH_AUTH_ID}"
)
DISPLAY_NAME = "Gemini Enterprise FinOps & Usage Pulse (A2UI + BQ MCP)"
DESCRIPTION = (
    "Live Gemini Enterprise FinOps, token usage, and audit log analytics via "
    "BigQuery MCP using end-user Google OAuth. Supports both Inline MaterialTabs "
    "and Side-Panel Canvas A2UI v0.9 visualizations."
)


def _get_token() -> str:
    return (
        subprocess.check_output(["gcloud", "auth", "print-access-token"])
        .decode("utf-8")
        .strip()
    )


def _find_existing_agent(token: str) -> str | None:
    url = f"{DISCOVERY_HOST}/v1alpha/{ENGINE_PARENT}/agents"
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "x-goog-user-project": PROJECT_ID,
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            for agent in data.get("agents", []):
                if (
                    agent.get("displayName") == DISPLAY_NAME
                    or APP_URL in json.dumps(agent.get("a2aAgentDefinition", {}))
                ):
                    return agent.get("name")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        print(f"Warning listing agents ({exc.code}): {body}")
    return None


def main() -> None:
    with TestClient(app) as client:
        card_resp = client.get("/a2a/app/.well-known/agent-card.json")
        card_dict = card_resp.json()
        card_url = f"{APP_URL.rstrip('/')}/a2a/app"
        card_dict["url"] = card_url
        for iface in card_dict.get("supportedInterfaces", []):
            iface["url"] = card_url

    token = _get_token()
    existing_name = _find_existing_agent(token)
    payload = {
        "displayName": DISPLAY_NAME,
        "description": DESCRIPTION,
        "a2aAgentDefinition": {
            "jsonAgentCard": json.dumps(card_dict),
        },
        "authorizationConfig": {
            "agentAuthorization": OAUTH_AUTHORIZATION,
        },
        "sharingConfig": {"scope": "ALL_USERS"},
        "starterPrompts": [{"text": p} for p in STARTER_PROMPTS_LIST],
        "agentInvocationSpec": {
            "description": DESCRIPTION,
            "invocationMode": "AUTOMATIC",
        },
    }

    if existing_name:
        payload["name"] = existing_name
        endpoint = (
            f"{DISCOVERY_HOST}/v1alpha/{existing_name}"
            "?updateMask=displayName,description,a2aAgentDefinition,authorizationConfig,sharingConfig,starterPrompts,agentInvocationSpec"
        )
        method = "PATCH"
    else:
        endpoint = f"{DISCOVERY_HOST}/v1alpha/{ENGINE_PARENT}/agents"
        method = "POST"

    req = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "x-goog-user-project": PROJECT_ID,
        },
        method=method,
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        res = json.loads(resp.read().decode("utf-8"))

    agent_name = res.get("name")
    if method == "POST" and agent_name:
        patch_url = (
            f"{DISCOVERY_HOST}/v1alpha/{agent_name}"
            "?updateMask=sharingConfig,starterPrompts,agentInvocationSpec,authorizationConfig"
        )
        patch_payload = {
            "name": agent_name,
            "authorizationConfig": {"agentAuthorization": OAUTH_AUTHORIZATION},
            "sharingConfig": {"scope": "ALL_USERS"},
            "starterPrompts": [{"text": p} for p in STARTER_PROMPTS_LIST],
            "agentInvocationSpec": {
                "description": DESCRIPTION,
                "invocationMode": "AUTOMATIC",
            },
        }
        patch_req = urllib.request.Request(
            patch_url,
            data=json.dumps(patch_payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "x-goog-user-project": PROJECT_ID,
            },
            method="PATCH",
        )
        with urllib.request.urlopen(patch_req, timeout=30) as patch_resp:
            res = json.loads(patch_resp.read().decode("utf-8"))

    print(
        json.dumps(
            {
                "name": res.get("name"),
                "displayName": res.get("displayName"),
                "state": res.get("state"),
                "authorizationConfig": res.get("authorizationConfig"),
                "sharingConfig": res.get("sharingConfig"),
                "starterPrompts": res.get("starterPrompts"),
                "agentInvocationSpec": res.get("agentInvocationSpec"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
