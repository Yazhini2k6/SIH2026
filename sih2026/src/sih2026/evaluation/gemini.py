"""Google Gemini Client for cloud LLM skill-gap analysis inference."""

import json
import os
import urllib.request
import urllib.error
from typing import Any, Optional


class GeminiClient:
    """Client for communicating with Google Gemini REST API."""

    def __init__(
        self,
        api_key: str = "",
        model: str = "gemini-2.5-flash"
    ):
        self.api_key = api_key.strip() or os.environ.get("GEMINI_API_KEY", "")
        self.model = model.strip() if model else "gemini-2.5-flash"

    def is_configured(self) -> bool:
        """Checks if Gemini API key is available."""
        return bool(self.api_key)

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        format_json: bool = True
    ) -> Any:
        """Sends generation prompt to Gemini model.

        Returns:
            dict (parsed JSON) or error dictionary.
        """
        if not self.is_configured():
            return {
                "status": "Gemini Unconfigured",
                "message": "GEMINI_API_KEY environment variable or argument is missing."
            }

        endpoint = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
            f"?key={self.api_key}"
        )

        contents = []
        if system_prompt:
            contents.append({"role": "user", "parts": [{"text": f"SYSTEM INSTRUCTIONS:\n{system_prompt}"}]})
        contents.append({"role": "user", "parts": [{"text": prompt}]})

        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": 0.2,
                "responseMimeType": "application/json" if format_json else "text/plain"
            }
        }

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            endpoint,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST"
        )

        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                result_json = json.loads(resp.read().decode("utf-8"))
                try:
                    text_content = result_json["candidates"][0]["content"]["parts"][0]["text"]
                    if format_json:
                        try:
                            return json.loads(text_content)
                        except json.JSONDecodeError:
                            return {"raw_response": text_content}
                    return text_content
                except (KeyError, IndexError) as err:
                    return {"status": "Gemini Parse Error", "raw": result_json}
        except urllib.error.URLError as err:
            return {"status": "Gemini API Error", "message": str(err)}
