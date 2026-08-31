"""OpenRouter LLM client for synthetic data generation.

Handles API calls with retries, reasoning-mode control, and cost tracking.
"""
import json
import os
import time
import requests

API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
if not API_KEY:
    raise RuntimeError("Set OPENROUTER_API_KEY environment variable before running.")
URL = "https://openrouter.ai/api/v1/chat/completions"


class LLMClient:
    def __init__(self, model, temperature=0.8, max_tokens=4000, reasoning_off=True):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.reasoning_off = reasoning_off
        self.total_cost = 0.0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.calls = 0
        self.failures = 0

    def _payload(self, messages):
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        # Disable reasoning for models that support it (cheaper, cleaner JSON)
        if self.reasoning_off:
            payload["reasoning"] = {"effort": "none"}
        return payload

    def complete(self, messages, retries=4, timeout=120):
        """Send a chat completion request with retries. Returns content string or None."""
        for attempt in range(retries):
            try:
                resp = requests.post(
                    URL,
                    headers={
                        "Authorization": f"Bearer {API_KEY}",
                        "Content-Type": "application/json",
                    },
                    json=self._payload(messages),
                    timeout=timeout,
                )
                data = resp.json()
                if "choices" in data and data["choices"]:
                    msg = data["choices"][0]["message"]
                    content = msg.get("content")
                    usage = data.get("usage", {})
                    self.total_cost += usage.get("cost", 0.0)
                    self.total_prompt_tokens += usage.get("prompt_tokens", 0)
                    self.total_completion_tokens += usage.get("completion_tokens", 0)
                    self.calls += 1
                    if content:
                        return content
                    # content is None -> reasoning-only output or empty
                    self.failures += 1
                    return None
                else:
                    err = data.get("error", {})
                    msg_text = err.get("message", str(data))[:200]
                    code = err.get("code")
                    # Rate limit / upstream errors -> retry with backoff
                    if code in (429, 503) or "rate" in msg_text.lower() or "temporarily" in msg_text.lower():
                        wait = 8 * (attempt + 1)
                        time.sleep(wait)
                        continue
                    self.failures += 1
                    return None
            except Exception as e:
                self.failures += 1
                time.sleep(6 * (attempt + 1))
        self.failures += 1
        return None

    def stats(self):
        return {
            "model": self.model,
            "cost": self.total_cost,
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "calls": self.calls,
            "failures": self.failures,
        }
