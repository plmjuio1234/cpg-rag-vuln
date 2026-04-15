#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


ISSUER = "https://auth.openai.com"
TOKEN_ENDPOINT = f"{ISSUER}/oauth/token"
CODEX_ENDPOINT = "https://chatgpt.com/backend-api/codex/responses"
CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"


def resolve_auth_path() -> Path:
    xdg_data_home = os.getenv("XDG_DATA_HOME")
    if xdg_data_home:
        return Path(xdg_data_home) / "opencode" / "auth.json"
    return Path.home() / ".local" / "share" / "opencode" / "auth.json"


def load_openai_oauth(auth_path: Path) -> dict[str, Any]:
    if not auth_path.exists() or not auth_path.is_file():
        raise RuntimeError(f"OpenCode auth file not found: {auth_path}")
    with auth_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    openai_auth = data.get("openai")
    if not isinstance(openai_auth, dict) or openai_auth.get("type") != "oauth":
        raise RuntimeError("OpenAI OAuth not found in OpenCode auth.json")
    return data


def parse_jwt_claims(token: str) -> dict[str, Any] | None:
    parts = token.split(".")
    if len(parts) != 3:
        return None
    try:
        import base64

        payload = parts[1]
        padding = "=" * ((4 - len(payload) % 4) % 4)
        decoded = base64.urlsafe_b64decode((payload + padding).encode("utf-8"))
        parsed = json.loads(decoded.decode("utf-8"))
        if isinstance(parsed, dict):
            return parsed
        return None
    except Exception:
        return None


def extract_account_id(tokens: dict[str, Any]) -> str | None:
    for key in ("id_token", "access_token"):
        token = tokens.get(key)
        if not isinstance(token, str) or not token:
            continue
        claims = parse_jwt_claims(token)
        if not claims:
            continue
        account_id = claims.get("chatgpt_account_id")
        if isinstance(account_id, str) and account_id:
            return account_id
        auth_claim = claims.get("https://api.openai.com/auth")
        if isinstance(auth_claim, dict):
            nested_id = auth_claim.get("chatgpt_account_id")
            if isinstance(nested_id, str) and nested_id:
                return nested_id
        orgs = claims.get("organizations")
        if isinstance(orgs, list) and orgs:
            first = orgs[0]
            if isinstance(first, dict):
                org_id = first.get("id")
                if isinstance(org_id, str) and org_id:
                    return org_id
    return None


def refresh_access_token(refresh_token: str, timeout: int) -> dict[str, Any]:
    body = urllib.parse.urlencode(
        {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": CLIENT_ID,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        TOKEN_ENDPOINT,
        data=body,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def ensure_access_token(
    auth_data: dict[str, Any], auth_path: Path, timeout: int
) -> tuple[str, str | None]:
    openai_auth = auth_data["openai"]
    access = openai_auth.get("access")
    expires = int(openai_auth.get("expires", 0) or 0)
    account_id = openai_auth.get("accountId")
    now_ms = int(time.time() * 1000)

    if isinstance(access, str) and access and expires > now_ms:
        return access, account_id if isinstance(account_id, str) else None

    refresh = openai_auth.get("refresh")
    if not isinstance(refresh, str) or not refresh:
        raise RuntimeError("OpenCode OAuth refresh token not available")

    token_data = refresh_access_token(refresh, timeout=timeout)
    new_access = token_data.get("access_token")
    if not isinstance(new_access, str) or not new_access:
        raise RuntimeError("Failed to refresh OpenAI OAuth access token")

    expires_in = int(token_data.get("expires_in", 3600) or 3600)
    new_account_id = extract_account_id(token_data) or (
        account_id if isinstance(account_id, str) else None
    )

    openai_auth["access"] = new_access
    openai_auth["expires"] = now_ms + (expires_in * 1000)
    if isinstance(new_account_id, str) and new_account_id:
        openai_auth["accountId"] = new_account_id

    with auth_path.open("w", encoding="utf-8") as f:
        json.dump(auth_data, f, ensure_ascii=True, indent=2)

    return new_access, new_account_id


def parse_stream_text(raw: str) -> tuple[str, dict[str, Any] | None]:
    text_parts: list[str] = []
    completed: dict[str, Any] | None = None

    for line in raw.splitlines():
        line = line.strip()
        if not line.startswith("data: "):
            continue
        payload = line[6:]
        if not payload:
            continue
        try:
            event = json.loads(payload)
        except json.JSONDecodeError:
            continue
        event_type = str(event.get("type", ""))
        if event_type == "response.output_text.delta":
            delta = event.get("delta")
            if isinstance(delta, str):
                text_parts.append(delta)
        elif event_type == "response.output_text.done" and not text_parts:
            done_text = event.get("text")
            if isinstance(done_text, str):
                text_parts.append(done_text)
        elif event_type == "response.completed":
            completed = (
                event.get("response")
                if isinstance(event.get("response"), dict)
                else None
            )

    return "".join(text_parts).strip(), completed


def call_codex(
    *,
    access_token: str,
    account_id: str | None,
    query: str,
    model: str,
    instructions: str,
    reasoning_effort: str | None,
    timeout: int,
) -> tuple[str, dict[str, Any] | None]:
    payload: dict[str, Any] = {
        "model": model,
        "instructions": instructions,
        "store": False,
        "stream": True,
        "input": [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": query}],
            }
        ],
    }
    if isinstance(reasoning_effort, str) and reasoning_effort:
        payload["reasoning"] = {"effort": reasoning_effort}
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {access_token}",
    }
    if isinstance(account_id, str) and account_id:
        headers["ChatGPT-Account-Id"] = account_id

    req = urllib.request.Request(
        CODEX_ENDPOINT,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    return parse_stream_text(raw)


def main() -> None:
    parser = argparse.ArgumentParser(description="Call GPT via OpenCode OAuth")
    parser.add_argument("query", type=str, help="Prompt text")
    parser.add_argument("--model", type=str, default="gpt-5.3-codex")
    parser.add_argument("--effort", type=str, default="xhigh")
    parser.add_argument(
        "--instructions", type=str, default="You are a helpful assistant."
    )
    parser.add_argument("--timeout", type=int, default=90)
    args = parser.parse_args()

    auth_path = resolve_auth_path()
    try:
        auth_data = load_openai_oauth(auth_path)
        access_token, account_id = ensure_access_token(
            auth_data, auth_path, timeout=args.timeout
        )
        text, completed = call_codex(
            access_token=access_token,
            account_id=account_id,
            query=args.query,
            model=args.model,
            instructions=args.instructions,
            reasoning_effort=args.effort,
            timeout=args.timeout,
        )
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        print(f"HTTP_ERROR {e.code}")
        print(body)
        raise SystemExit(1)
    except Exception as e:  # noqa: BLE001
        print(f"ERROR: {e}")
        raise SystemExit(1)

    if text:
        print(text)
        return

    if completed:
        print(json.dumps(completed, ensure_ascii=True, indent=2))
        return

    print("(no output)")


if __name__ == "__main__":
    main()
