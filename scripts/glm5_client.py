#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import random
import socket
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


class GLM5Client:
    ENDPOINT = "https://api.z.ai/api/anthropic/v1/messages"
    RETRYABLE_HTTP_CODES = {429, 500, 502, 503, 504, 529}

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = "glm-5",
        timeout: int = 60,
        max_retries: int = 2,
        endpoint: str = ENDPOINT,
        temperature: float = 0.0,
        top_p: float | None = None,
    ) -> None:
        self.api_key = api_key or self._resolve_api_key()
        if not self.api_key:
            raise ValueError("GLM API key is required (GLM_API_KEY or ZAI_API_KEY)")
        self.model = model
        self.timeout = timeout
        self.max_retries = max(0, max_retries)
        self.endpoint = endpoint
        self.temperature = float(temperature)
        self.top_p = None if top_p is None else float(top_p)

    @staticmethod
    def _read_api_key_from_env_file(path: Path) -> str | None:
        if not path.exists() or not path.is_file():
            return None
        try:
            with path.open("r", encoding="utf-8") as f:
                for raw in f:
                    line = raw.strip()
                    if not line or line.startswith("#"):
                        continue
                    if line.startswith("export "):
                        line = line[len("export ") :].strip()
                    if "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    key = k.strip()
                    if key not in {"GLM_API_KEY", "ZAI_API_KEY"}:
                        continue
                    val = v.strip()
                    if val and val[0] in {'"', "'"} and val[-1] == val[0]:
                        val = val[1:-1]
                    if val:
                        return val
        except OSError:
            return None
        return None

    @classmethod
    def _resolve_api_key(cls) -> str | None:
        env_key = os.getenv("GLM_API_KEY") or os.getenv("ZAI_API_KEY")
        if env_key:
            return env_key
        candidates = [
            Path.cwd() / ".env",
            Path(__file__).resolve().parents[1] / ".env",
        ]
        for c in candidates:
            v = cls._read_api_key_from_env_file(c)
            if v:
                os.environ.setdefault("GLM_API_KEY", v)
                return v
        return None

    def _headers(self) -> dict[str, str]:
        api_key = str(self.api_key)
        return {
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "User-Agent": "opencode/1.1.64",
            "X-Client-Name": "opencode",
        }

    @staticmethod
    def _extract_text(response_json: dict[str, Any]) -> str:
        content = response_json.get("content", [])
        if isinstance(content, str):
            return content
        parts: list[str] = []
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    txt = block.get("text")
                    if isinstance(txt, str):
                        parts.append(txt)
        return "\n".join(parts).strip()

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
            if isinstance(parsed, dict):
                return parsed
            return None
        except json.JSONDecodeError:
            return None

    def _call(
        self, *, system: str, user: str, max_tokens: int = 1024
    ) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": self.temperature,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        if self.top_p is not None:
            payload["top_p"] = self.top_p
        last_error: Exception | None = None
        for i in range(self.max_retries + 1):
            req = urllib.request.Request(
                self.endpoint,
                data=json.dumps(payload).encode("utf-8"),
                headers=self._headers(),
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    return json.loads(r.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8")
                if e.code in self.RETRYABLE_HTTP_CODES and i < self.max_retries:
                    time.sleep((2**i) + random.uniform(0.0, 0.25))
                    continue
                raise RuntimeError(f"GLM HTTP {e.code}: {body}") from e
            except (urllib.error.URLError, TimeoutError, socket.timeout) as e:
                last_error = e
                if i < self.max_retries:
                    time.sleep((2**i) + random.uniform(0.0, 0.25))
                    continue
                break
        if last_error:
            raise RuntimeError(f"GLM request failed: {last_error}") from last_error
        raise RuntimeError("GLM request failed")

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
        system = (
            "You are a strict vulnerability verifier. "
            "Return JSON only with keys: decision, vulnerable_score, safe_score, confidence, reason, review_flag. "
            "decision must be VULNERABLE or SAFE. vulnerable_score/safe_score/confidence in [0,1]. "
            "vulnerable_score and safe_score are calibrated support scores for each class."
        )
        user = (
            f"sample_id: {sample_id}\n"
            f"anchor_prediction: {anchor_prediction}\n"
            f"anchor_confidence: {anchor_confidence:.6f}\n"
            f"retrieved_context:\n- "
            + "\n- ".join(retrieved_context[:5])
            + "\ncode:\n```c\n"
            + code[:3000]
            + "\n```"
        )
        response = self._call(system=system, user=user, max_tokens=max_tokens)
        raw = self._extract_text(response)
        parsed = self._extract_json(raw)
        if not parsed:
            return {
                "decision": "UNKNOWN",
                "confidence": 0.0,
                "vulnerable_score": 0.5,
                "safe_score": 0.5,
                "score_margin": 0.0,
                "reason": raw,
                "review_flag": True,
                "raw_response": raw,
                "parse_ok": False,
            }

        def _clip01(value: Any, default: float) -> float:
            try:
                out = float(value)
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
            "raw_response": raw,
            "parse_ok": True,
        }
