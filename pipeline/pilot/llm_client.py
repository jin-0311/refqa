#!/usr/bin/env python3
"""Minimal async client wrapper for an OpenAI-compatible (vLLM) chat endpoint.

This is the small helper that `extract_qa_glm_v03.py` imports
(`LLMClient`, `make_vllm_config`). It is intentionally self-contained and
dependency-light: it talks to any OpenAI-compatible `/v1` endpoint (such as
a local vLLM server) via the `openai` async client, with

  - round-robin load balancing across one or more `base_urls`,
  - bounded retries with temperature ramping,
  - JSON parsing + optional Pydantic validation of the model output,
  - an optional SQLite response cache keyed by SHA-256 of (model, system, user),
  - token-usage / failure-kind statistics.

No endpoint addresses or credentials are hard-coded: the base URL is passed
in (see `--base-url`, default `http://localhost:8000/v1`) and the API key is
read from the `OPENAI_API_KEY` environment variable, defaulting to the dummy
value `"EMPTY"` that vLLM accepts.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import time
from collections import Counter
from dataclasses import dataclass, field
from itertools import cycle
from typing import Any, Optional

try:
    from openai import AsyncOpenAI
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "llm_client requires the 'openai' package: pip install openai"
    ) from exc


# ---------------------------------------------------------------------------
# Config

@dataclass
class VLLMConfig:
    model: str
    base_url: str
    base_urls: list[str] = field(default_factory=list)
    api_key: str = ""

    def __post_init__(self):
        if not self.base_urls:
            self.base_urls = [self.base_url]
        if not self.api_key:
            self.api_key = os.environ.get("OPENAI_API_KEY", "EMPTY")


def make_vllm_config(model: str, base_url: str,
                     base_urls: Optional[list[str]] = None) -> VLLMConfig:
    """Build a config for one or more OpenAI-compatible endpoints."""
    return VLLMConfig(model=model, base_url=base_url,
                      base_urls=list(base_urls) if base_urls else [base_url])


# ---------------------------------------------------------------------------
# Result

@dataclass
class ChatResult:
    ok: bool
    data: Any = None            # parsed dict on success; error message on failure
    raw_text: str = ""
    usage_in: int = 0
    usage_out: int = 0
    latency_s: float = 0.0
    attempts: int = 0
    error_kind: str = ""        # json_parse | schema_invalid | api_error | empty


# ---------------------------------------------------------------------------
# JSON extraction helper

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def _extract_json(text: str) -> dict:
    """Parse a JSON object from a model response, tolerating code fences and
    leading/trailing prose."""
    s = text.strip()
    s = _FENCE.sub("", s).strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    # fall back to the first balanced {...} span
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(s[start:end + 1])
    raise json.JSONDecodeError("no JSON object found", s, 0)


# ---------------------------------------------------------------------------
# Client

class LLMClient:
    def __init__(self, cfg: VLLMConfig, *, concurrency: int = 16,
                 cache_db: Optional[str] = None, validate_with: Any = None,
                 max_retry: int = 3, temperatures: Optional[list[float]] = None,
                 max_tokens: int = 2048):
        self.cfg = cfg
        self.concurrency = concurrency
        self.validate_with = validate_with
        self.max_retry = max_retry
        self.max_tokens = max_tokens
        # one temperature per attempt; ramps up so retries diversify
        self.temperatures = temperatures or [0.0, 0.3, 0.6, 0.9]
        self._clients = [AsyncOpenAI(base_url=u, api_key=cfg.api_key)
                         for u in cfg.base_urls]
        self._rr = cycle(range(len(self._clients)))
        self._use_response_format = True
        self._in = 0
        self._out = 0
        self._fail_kinds: Counter = Counter()
        self._cache = None
        if cache_db:
            self._cache = sqlite3.connect(cache_db, check_same_thread=False)
            self._cache.execute(
                "CREATE TABLE IF NOT EXISTS cache "
                "(key TEXT PRIMARY KEY, raw TEXT, data TEXT)")
            self._cache.commit()

    @property
    def stats(self) -> dict:
        return {"input_tokens": self._in, "output_tokens": self._out,
                "fail_kinds": self._fail_kinds}

    # -- cache helpers --
    def _key(self, system: str, user: str) -> str:
        h = hashlib.sha256()
        h.update(self.cfg.model.encode()); h.update(b"\x00")
        h.update(system.encode()); h.update(b"\x00")
        h.update(user.encode())
        return h.hexdigest()

    def _cache_get(self, key: str):
        if not self._cache:
            return None
        row = self._cache.execute(
            "SELECT raw, data FROM cache WHERE key=?", (key,)).fetchone()
        if row is None:
            return None
        raw, data = row
        return ChatResult(ok=True, data=json.loads(data), raw_text=raw,
                          attempts=0)

    def _cache_put(self, key: str, raw: str, data: dict):
        if not self._cache:
            return
        self._cache.execute(
            "INSERT OR REPLACE INTO cache (key, raw, data) VALUES (?,?,?)",
            (key, raw, json.dumps(data, ensure_ascii=False)))
        self._cache.commit()

    # -- main entry point --
    async def chat_json(self, system: str, user: str) -> ChatResult:
        key = self._key(system, user)
        cached = self._cache_get(key)
        if cached is not None:
            return cached

        t0 = time.time()
        last_kind = "api_error"
        last_msg = ""
        last_raw = ""
        client_idx = next(self._rr)
        for attempt in range(1, self.max_retry + 2):
            temp = self.temperatures[min(attempt - 1, len(self.temperatures) - 1)]
            client = self._clients[(client_idx + attempt - 1) % len(self._clients)]
            try:
                kwargs = dict(
                    model=self.cfg.model,
                    messages=[{"role": "system", "content": system},
                              {"role": "user", "content": user}],
                    temperature=temp,
                    max_tokens=self.max_tokens,
                )
                if self._use_response_format:
                    kwargs["response_format"] = {"type": "json_object"}
                resp = await client.chat.completions.create(**kwargs)
            except Exception as e:  # noqa: BLE001
                msg = str(e)
                # endpoint that rejects response_format: drop it and retry
                if self._use_response_format and "response_format" in msg:
                    self._use_response_format = False
                last_kind, last_msg = "api_error", msg
                continue

            content = (resp.choices[0].message.content or "") if resp.choices else ""
            if resp.usage:
                self._in += getattr(resp.usage, "prompt_tokens", 0) or 0
                self._out += getattr(resp.usage, "completion_tokens", 0) or 0
            last_raw = content
            if not content.strip():
                last_kind, last_msg = "empty", "empty completion"
                continue
            try:
                parsed = _extract_json(content)
            except json.JSONDecodeError as e:
                last_kind, last_msg = "json_parse", str(e)
                continue
            if self.validate_with is not None:
                try:
                    self.validate_with.model_validate(parsed)
                except Exception as e:  # noqa: BLE001  (pydantic ValidationError)
                    last_kind, last_msg = "schema_invalid", str(e)
                    continue
            # success
            self._cache_put(key, content, parsed)
            return ChatResult(
                ok=True, data=parsed, raw_text=content,
                usage_in=self._last_in(resp), usage_out=self._last_out(resp),
                latency_s=time.time() - t0, attempts=attempt)

        self._fail_kinds[last_kind] += 1
        return ChatResult(ok=False, data=last_msg, raw_text=last_raw,
                          latency_s=time.time() - t0,
                          attempts=self.max_retry + 1, error_kind=last_kind)

    @staticmethod
    def _last_in(resp) -> int:
        return getattr(getattr(resp, "usage", None), "prompt_tokens", 0) or 0

    @staticmethod
    def _last_out(resp) -> int:
        return getattr(getattr(resp, "usage", None), "completion_tokens", 0) or 0
