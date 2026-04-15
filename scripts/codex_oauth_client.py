#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import random
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


class CodexOAuthClient:
    ENDPOINT = "https://chatgpt.com/backend-api/codex/responses"
    TOKEN_ENDPOINT = "https://auth.openai.com/oauth/token"
    CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
    RETRYABLE_HTTP_CODES = {429, 500, 502, 503, 504, 529}

    def __init__(
        self,
        *,
        model: str = "gpt-5.3-codex",
        timeout: int = 90,
        max_retries: int = 2,
        reasoning_effort: str = "xhigh",
    ) -> None:
        self.model = model
        self.timeout = timeout
        self.max_retries = max(0, max_retries)
        self.reasoning_effort = reasoning_effort

    @staticmethod
    def _auth_path() -> Path:
        xdg_data_home = os.getenv("XDG_DATA_HOME")
        if xdg_data_home:
            return Path(xdg_data_home) / "opencode" / "auth.json"
        return Path.home() / ".local" / "share" / "opencode" / "auth.json"

    @classmethod
    def _load_oauth(cls) -> dict[str, Any]:
        path = cls._auth_path()
        if not path.exists() or not path.is_file():
            raise RuntimeError(f"OpenCode auth file not found: {path}")
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        openai_auth = data.get("openai")
        if not isinstance(openai_auth, dict) or openai_auth.get("type") != "oauth":
            raise RuntimeError("OpenAI OAuth not found in OpenCode auth.json")
        return data

    @staticmethod
    def _parse_jwt_claims(token: str) -> dict[str, Any] | None:
        parts = token.split(".")
        if len(parts) != 3:
            return None
        try:
            import base64

            payload = parts[1]
            padding = "=" * ((4 - len(payload) % 4) % 4)
            raw = base64.urlsafe_b64decode((payload + padding).encode("utf-8"))
            parsed = json.loads(raw.decode("utf-8"))
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            return None

    @classmethod
    def _extract_account_id(cls, token_data: dict[str, Any]) -> str | None:
        for key in ("id_token", "access_token"):
            token = token_data.get(key)
            if not isinstance(token, str) or not token:
                continue
            claims = cls._parse_jwt_claims(token)
            if not claims:
                continue
            direct = claims.get("chatgpt_account_id")
            if isinstance(direct, str) and direct:
                return direct
            auth_claim = claims.get("https://api.openai.com/auth")
            if isinstance(auth_claim, dict):
                nested = auth_claim.get("chatgpt_account_id")
                if isinstance(nested, str) and nested:
                    return nested
            orgs = claims.get("organizations")
            if isinstance(orgs, list) and orgs:
                first = orgs[0]
                if isinstance(first, dict):
                    org_id = first.get("id")
                    if isinstance(org_id, str) and org_id:
                        return org_id
        return None

    @classmethod
    def _refresh_access_token(cls, refresh_token: str, timeout: int) -> dict[str, Any]:
        body = urllib.parse.urlencode(
            {
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": cls.CLIENT_ID,
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            cls.TOKEN_ENDPOINT,
            data=body,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    @classmethod
    def _resolve_access_token(cls, timeout: int) -> tuple[str, str | None]:
        auth_data = cls._load_oauth()
        path = cls._auth_path()
        openai_auth = auth_data["openai"]

        access = openai_auth.get("access")
        expires = int(openai_auth.get("expires", 0) or 0)
        now_ms = int(time.time() * 1000)
        account_id = openai_auth.get("accountId")

        if isinstance(access, str) and access and expires > now_ms:
            return access, account_id if isinstance(account_id, str) else None

        refresh = openai_auth.get("refresh")
        if not isinstance(refresh, str) or not refresh:
            raise RuntimeError("OpenCode OAuth refresh token not available")

        token_data = cls._refresh_access_token(refresh, timeout=timeout)
        new_access = token_data.get("access_token")
        if not isinstance(new_access, str) or not new_access:
            raise RuntimeError("Failed to refresh OpenCode OAuth access token")

        expires_in = int(token_data.get("expires_in", 3600) or 3600)
        new_account_id = cls._extract_account_id(token_data) or (
            account_id if isinstance(account_id, str) else None
        )

        openai_auth["access"] = new_access
        openai_auth["expires"] = now_ms + (expires_in * 1000)
        if isinstance(new_account_id, str) and new_account_id:
            openai_auth["accountId"] = new_account_id

        with path.open("w", encoding="utf-8") as f:
            json.dump(auth_data, f, indent=2, ensure_ascii=True)

        return new_access, new_account_id

    @staticmethod
    def _extract_json(raw: str) -> dict[str, Any] | None:
        text = raw.strip()
        if "```json" in text:
            text = text.split("```json", 1)[1].split("```", 1)[0].strip()
        elif "```" in text:
            text = text.split("```", 2)[1].strip()
        else:
            s = text.find("{")
            e = text.rfind("}")
            if s != -1 and e != -1 and e > s:
                text = text[s : e + 1]
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list) and parsed:
                parsed = parsed[0]
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None

    def _call_stream(
        self, *, instructions: str, user_text: str
    ) -> tuple[str, dict[str, Any] | None]:
        access_token, account_id = self._resolve_access_token(timeout=self.timeout)

        payload: dict[str, Any] = {
            "model": self.model,
            "instructions": instructions,
            "store": False,
            "stream": True,
            "input": [
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": user_text}],
                }
            ],
            "reasoning": {"effort": self.reasoning_effort},
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {access_token}",
        }
        if isinstance(account_id, str) and account_id:
            headers["ChatGPT-Account-Id"] = account_id

        last_error: Exception | None = None
        for i in range(self.max_retries + 1):
            req = urllib.request.Request(
                self.ENDPOINT,
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    raw = resp.read().decode("utf-8", errors="replace")
                text_parts: list[str] = []
                completed: dict[str, Any] | None = None
                for line in raw.splitlines():
                    line = line.strip()
                    if not line.startswith("data: "):
                        continue
                    event_raw = line[6:]
                    if not event_raw:
                        continue
                    try:
                        event = json.loads(event_raw)
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
                        response_obj = event.get("response")
                        if isinstance(response_obj, dict):
                            completed = response_obj
                return "".join(text_parts).strip(), completed
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", errors="replace")
                if e.code in self.RETRYABLE_HTTP_CODES and i < self.max_retries:
                    time.sleep((2**i) + random.uniform(0.0, 0.25))
                    continue
                raise RuntimeError(f"Codex HTTP {e.code}: {body}") from e
            except Exception as e:  # noqa: BLE001
                last_error = e
                if i < self.max_retries:
                    time.sleep((2**i) + random.uniform(0.0, 0.25))
                    continue
                break

        if last_error is not None:
            raise RuntimeError(f"Codex request failed: {last_error}") from last_error
        raise RuntimeError("Codex request failed")

    def classify_with_rag(
        self,
        *,
        sample_id: str,
        code: str,
        anchor_prediction: int,
        anchor_confidence: float,
        retrieved_context: list[str],
        max_tokens: int = 700,
    ) -> dict[str, Any]:
        _ = max_tokens
        instructions = (
            "You are a strict vulnerability verifier. "
            "Judge only based on code that is explicitly provided. "
            "Do not assume behavior of unseen files or unresolved function bodies. "
            "Return JSON only with keys: decision, vulnerable_score, safe_score, confidence, reason, review_flag. "
            "decision must be VULNERABLE or SAFE. vulnerable_score/safe_score/confidence in [0,1]."
        )
        user_text = (
            f"sample_id: {sample_id}\n"
            f"anchor_prediction: {anchor_prediction}\n"
            f"anchor_confidence: {anchor_confidence:.6f}\n"
            f"retrieved_context:\n- "
            + "\n- ".join(retrieved_context)
            + "\ncode:\n```c\n"
            + code[:3000]
            + "\n```"
        )

        raw_text, completed = self._call_stream(
            instructions=instructions, user_text=user_text
        )
        parsed = self._extract_json(raw_text)
        if not parsed:
            return {
                "decision": "UNKNOWN",
                "confidence": 0.0,
                "vulnerable_score": 0.5,
                "safe_score": 0.5,
                "score_margin": 0.0,
                "reason": raw_text,
                "review_flag": True,
                "raw_response": raw_text,
                "parse_ok": False,
                "response_model": (
                    str(completed.get("model", ""))
                    if isinstance(completed, dict)
                    else ""
                ),
            }

        def _clip01(v: Any, default: float) -> float:
            try:
                out = float(v)
            except (TypeError, ValueError):
                return default
            return max(0.0, min(1.0, out))

        decision = str(parsed.get("decision", "UNKNOWN")).upper()
        confidence = _clip01(parsed.get("confidence", 0.0), 0.0)
        reason = str(parsed.get("reason", ""))

        raw_vul = parsed.get("vulnerable_score")
        raw_safe = parsed.get("safe_score")
        if raw_vul is None or raw_safe is None:
            if decision == "VULNERABLE":
                vulnerable_score = confidence
                safe_score = 1.0 - confidence
            elif decision == "SAFE":
                vulnerable_score = 1.0 - confidence
                safe_score = confidence
            else:
                vulnerable_score = 0.5
                safe_score = 0.5
        else:
            vulnerable_score = _clip01(raw_vul, 0.5)
            safe_score = _clip01(raw_safe, 0.5)

        if decision not in {"VULNERABLE", "SAFE"}:
            decision = "VULNERABLE" if vulnerable_score >= safe_score else "SAFE"

        confidence = max(confidence, vulnerable_score, safe_score)
        score_margin = abs(vulnerable_score - safe_score)
        review_flag = bool(parsed.get("review_flag", score_margin < 0.2))

        return {
            "decision": decision,
            "confidence": confidence,
            "vulnerable_score": vulnerable_score,
            "safe_score": safe_score,
            "score_margin": score_margin,
            "reason": reason,
            "review_flag": review_flag,
            "raw_response": raw_text,
            "parse_ok": True,
            "response_model": (
                str(completed.get("model", "")) if isinstance(completed, dict) else ""
            ),
        }
