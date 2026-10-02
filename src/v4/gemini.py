"""Minimal Gemini REST client (stdlib only; no SDK, no new dependency).

Key: env GEMINI_API_KEY (never logged, returned or put in URLs -- sent as the x-goog-api-key header).
Model: env GEMINI_MODEL (default below). Every failure is raised as GeminiError with a short,
non-sensitive ``code`` so callers can fall back to the V3 answer.
"""
from __future__ import annotations

import json
import os
import socket
import urllib.error
import urllib.request

DEFAULT_MODEL = "gemini-3.5-flash-lite"      # stable, free-tier, fast (checked against ai.google.dev, Oct 2026)
ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


class GeminiError(RuntimeError):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code        # missing_key | timeout | rate_limited | quota | http_<n> | network | bad_json | blocked


def api_key() -> str | None:
    k = os.environ.get("GEMINI_API_KEY", "").strip()
    return k or None


def model_name() -> str:
    return os.environ.get("GEMINI_MODEL", "").strip() or DEFAULT_MODEL


class GeminiClient:
    def __init__(self, model: str | None = None, timeout: float | None = None):
        self.model = model or model_name()
        self.timeout = timeout or float(os.environ.get("GEMINI_TIMEOUT_SECONDS", "20"))
        self.calls = 0
        self.tokens = {"prompt": 0, "output": 0}
        self.last_error_detail: str | None = None    # Google's error message (never contains the key)
        self.last_raw: str | None = None             # model text of the last unparseable reply (diagnostics)
        self.last_finish_reason: str | None = None

    @property
    def enabled(self) -> bool:
        return api_key() is not None

    def _post(self, body: dict) -> dict:
        key = api_key()
        if not key:
            raise GeminiError("missing_key")
        req = urllib.request.Request(
            ENDPOINT.format(model=self.model), data=json.dumps(body).encode("utf-8"), method="POST",
            headers={"Content-Type": "application/json", "x-goog-api-key": key})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            raw = ""
            try:
                raw = e.read().decode("utf-8", "replace")
                err = json.loads(raw).get("error", {})
                self.last_error_detail = f"{err.get('status', '')}: {str(err.get('message', ''))[:300]}"
            except (ValueError, OSError, AttributeError):
                self.last_error_detail = None
            if e.code == 400 and ("API_KEY_INVALID" in raw or "API key not valid" in raw):
                raise GeminiError("invalid_key") from None
            if e.code == 429:
                raise GeminiError("rate_limited") from None
            if e.code == 403:
                raise GeminiError("quota") from None
            raise GeminiError(f"http_{e.code}") from None
        except (socket.timeout, TimeoutError):
            raise GeminiError("timeout") from None
        except urllib.error.URLError as e:
            raise GeminiError("timeout" if isinstance(e.reason, (socket.timeout, TimeoutError)) else "network") \
                from None
        except (ValueError, OSError):
            raise GeminiError("network") from None

    def generate_json(self, system: str, prompt: str, max_tokens: int = 1200) -> dict:
        """One call; returns the parsed JSON object the model produced."""
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"responseMimeType": "application/json", "temperature": 0.1,
                                 "maxOutputTokens": max_tokens},
        }
        self.calls += 1
        data = self._post(body)
        um = data.get("usageMetadata") or {}
        self.tokens["prompt"] += int(um.get("promptTokenCount") or 0)
        self.tokens["output"] += int(um.get("candidatesTokenCount") or 0)
        try:
            cand = data["candidates"][0]
            self.last_finish_reason = cand.get("finishReason")
            text = "".join(p.get("text", "") for p in cand["content"]["parts"])
        except (KeyError, IndexError, TypeError):
            raise GeminiError("blocked") from None
        text = text.strip()
        if text.startswith("```"):
            text = text.strip("`").removeprefix("json").strip()
        try:
            out = json.loads(text)
        except ValueError:
            i, j = text.find("{"), text.rfind("}")      # tolerate prose around the object; truncation still fails
            try:
                out = json.loads(text[i:j + 1]) if 0 <= i < j else None
            except ValueError:
                out = None
            if out is None:
                self.last_raw = text[:4000]
                raise GeminiError("bad_json") from None
        if not isinstance(out, dict):
            raise GeminiError("bad_json")
        return out
