"""
Project Akshara: Kimi / Moonshot API Client (shadow/challenger use)
===================================================================

Minimal OpenAI-compatible client wrapper for Moonshot Kimi models.

Primary use in Akshara:
- shadow/challenger comparisons for assembly planning prompts
- JSON-mode operations planning (no freeform rewrite dependency)
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests


@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cached_tokens=self.cached_tokens + other.cached_tokens,
        )


@dataclass
class KimiResponse:
    text: str
    tokens: TokenUsage
    finish_reason: str
    duration_ms: int
    model: str
    success: bool
    error: Optional[str] = None
    json_data: Optional[Dict[str, Any]] = None
    raw: Optional[Dict[str, Any]] = None


class KimiClient:
    """
    OpenAI-compatible wrapper for Moonshot (Kimi) chat completions.

    Designed to mirror the subset of GeminiClient used by assembly code:
    - generate(prompt, json_mode=True/False)
    - success/error/json_data/text fields on response
    - token/cost tracking for logs
    """

    DEFAULT_BASE_URL = "https://api.moonshot.cn/v1"

    # Pricing can change; values below are placeholders for shadow logging only.
    # Use Moonshot pricing docs for authoritative current pricing.
    PRICING_HINTS = {
        "default": {"input": 0.0, "output": 0.0, "cached": 0.0},
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        default_model: str = "kimi-k2.5",
        base_url: str = DEFAULT_BASE_URL,
        timeout: int = 120,
        max_retries: int = 3,
        retry_delay: float = 2.0,
        verbose: bool = True,
    ):
        self.api_key = api_key or os.environ.get("MOONSHOT_API_KEY")
        if not self.api_key:
            raise ValueError("MOONSHOT_API_KEY not found in environment")

        self.default_model = default_model
        self.base_url = base_url.rstrip("/")
        self.timeout = int(timeout)
        self.max_retries = int(max_retries)
        self.retry_delay = float(retry_delay)
        self.verbose = verbose

        self.total_tokens = TokenUsage()
        self.total_cost = 0.0

        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
        )

    def _log(self, message: str):
        if self.verbose:
            print(f"[Kimi] {message}")

    def _calculate_cost(self, tokens: TokenUsage, model: str) -> float:
        # Placeholder until exact model pricing is wired into config.
        pricing = self.PRICING_HINTS.get(model, self.PRICING_HINTS["default"])
        return (
            (tokens.input_tokens / 1_000_000) * pricing["input"]
            + (tokens.output_tokens / 1_000_000) * pricing["output"]
            + (tokens.cached_tokens / 1_000_000) * pricing["cached"]
        )

    def _retry_with_backoff(self, func, *args, **kwargs):
        last_error = None
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** attempt)
                    m = re.search(r'retry-after=(\d+(?:\.\d+)?)', str(e), flags=re.IGNORECASE)
                    if m:
                        try:
                            delay = max(delay, float(m.group(1)))
                        except Exception:
                            pass
                    self._log(f"Retry {attempt + 1}/{self.max_retries} in {delay:.1f}s: {e}")
                    time.sleep(delay)
        raise last_error

    def _post_chat_completions(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url}/chat/completions"
        r = self.session.post(url, json=payload, timeout=self.timeout)
        if r.status_code >= 400:
            retry_after = r.headers.get("Retry-After")
            suffix = f" [retry-after={retry_after}]" if retry_after else ""
            raise RuntimeError(f"HTTP {r.status_code}{suffix}: {r.text[:1000]}")
        return r.json()

    def _get(self, path: str) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        r = self.session.get(url, timeout=self.timeout)
        if r.status_code >= 400:
            retry_after = r.headers.get("Retry-After")
            suffix = f" [retry-after={retry_after}]" if retry_after else ""
            raise RuntimeError(f"HTTP {r.status_code}{suffix}: {r.text[:1000]}")
        return r.json()

    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        r = self.session.post(url, json=payload, timeout=self.timeout)
        if r.status_code >= 400:
            retry_after = r.headers.get("Retry-After")
            suffix = f" [retry-after={retry_after}]" if retry_after else ""
            raise RuntimeError(f"HTTP {r.status_code}{suffix}: {r.text[:1000]}")
        return r.json()

    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        thinking: Any = None,  # kept for API compatibility; currently ignored
        system_prompt: Optional[str] = None,
        json_mode: bool = False,
        json_schema: Optional[Dict[str, Any]] = None,  # currently not enforced
    ) -> KimiResponse:
        model_id = model or self.default_model
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload: Dict[str, Any] = {
            "model": model_id,
            "messages": messages,
            "stream": False,
        }
        if json_mode:
            # OpenAI-compatible JSON mode.
            if json_schema:
                payload["response_format"] = {
                    "type": "json_schema",
                    "json_schema": json_schema,
                }
            else:
                payload["response_format"] = {"type": "json_object"}

        start = time.time()
        try:
            data = self._retry_with_backoff(self._post_chat_completions, payload)
            duration_ms = int((time.time() - start) * 1000)

            choices = data.get("choices") or []
            choice0 = choices[0] if choices else {}
            message = choice0.get("message") or {}
            content = message.get("content") or ""
            finish_reason = str(choice0.get("finish_reason") or "STOP")

            usage = data.get("usage") or {}
            tokens = TokenUsage(
                input_tokens=int(usage.get("prompt_tokens") or 0),
                output_tokens=int(usage.get("completion_tokens") or 0),
                cached_tokens=int(
                    (usage.get("cached_tokens") or 0)
                    if isinstance(usage, dict)
                    else 0
                ),
            )
            self.total_tokens = self.total_tokens + tokens
            self.total_cost += self._calculate_cost(tokens, model_id)

            parsed_json = None
            if json_mode and content:
                try:
                    parsed_json = json.loads(content)
                except Exception:
                    parsed_json = None

            return KimiResponse(
                text=content,
                tokens=tokens,
                finish_reason=finish_reason,
                duration_ms=duration_ms,
                model=model_id,
                success=True,
                json_data=parsed_json,
                raw=data,
            )
        except Exception as e:
            duration_ms = int((time.time() - start) * 1000)
            return KimiResponse(
                text="",
                tokens=TokenUsage(),
                finish_reason="ERROR",
                duration_ms=duration_ms,
                model=model_id,
                success=False,
                error=str(e),
            )

    def get_usage_summary(self) -> Dict[str, Any]:
        total = self.total_tokens.input_tokens + self.total_tokens.output_tokens + self.total_tokens.cached_tokens
        return {
            "provider": "moonshot",
            "model": self.default_model,
            "total_cost_usd": round(self.total_cost, 6),
            "total_tokens": {
                "input": self.total_tokens.input_tokens,
                "output": self.total_tokens.output_tokens,
                "cached": self.total_tokens.cached_tokens,
                "total": total,
            },
        }

    # --- Capability helpers for orchestration / observability ---

    def list_models(self) -> Dict[str, Any]:
        return self._retry_with_backoff(self._get, "/models")

    def get_balance(self) -> Dict[str, Any]:
        return self._retry_with_backoff(self._get, "/users/me/balance")

    def estimate_token_count(self, messages: Any, model: Optional[str] = None) -> Dict[str, Any]:
        payload = {
            "model": model or self.default_model,
            "messages": messages,
        }
        # Endpoint name observed in docs guide examples.
        return self._retry_with_backoff(self._post, "/tokenizers/estimate-token-count", payload)
