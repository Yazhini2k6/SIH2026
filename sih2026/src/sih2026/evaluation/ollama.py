"""Ollama Client for local LLM skill-gap analysis inference."""

import json
import urllib.request
import urllib.error
from typing import Any, Optional


class OllamaClient:
    """Client for communicating with local Ollama server over HTTP REST API."""

    def __init__(
        self,
        host: str = "",
        model: str = "",
        timeout: int = 180
    ):
        self.host = (host.strip() if host else "http://localhost:11434").rstrip("/")
        self.model = model.strip() if model else "llama3.1"
        self.timeout = timeout

    def is_server_online(self) -> bool:
        """Checks if local Ollama server is running and reachable."""
        try:
            req = urllib.request.Request(f"{self.host}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=3) as resp:
                return resp.status == 200
        except Exception:
            return False

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        format_json: bool = True
    ) -> Any:
        """Sends generation prompt to Ollama model.

        Returns:
            dict (parsed JSON) or raw response string.
        """
        if not self.is_server_online():
            raise ConnectionError(
                f"Ollama server is not responding at '{self.host}'. "
                "Ensure Ollama is running (`ollama serve`)."
            )

        endpoint = f"{self.host}/api/generate"
        payload = {
            "model": self.model,
            "prompt": f"{system_prompt}\n\n{prompt}" if system_prompt else prompt,
            "stream": False
        }
        if format_json:
            payload["format"] = "json"

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            endpoint,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST"
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                result_json = json.loads(resp.read().decode("utf-8"))
                response_text = result_json.get("response", "").strip()

                if format_json:
                    try:
                        return json.loads(response_text)
                    except json.JSONDecodeError:
                        return {"raw_response": response_text}
                return response_text
        except urllib.error.URLError as err:
            raise RuntimeError(f"Ollama API request failed: {err}")
