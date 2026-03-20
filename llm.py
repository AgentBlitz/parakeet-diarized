"""LLM client for Ollama-hosted meeting intelligence extraction.

Wraps Ollama's OpenAI-compatible API to send diarized transcripts to
IBM Granite 3.3 8B Instruct and parse structured meeting intelligence.
"""

import json
import logging
import re
import time
from typing import Optional

import httpx

from config import get_config
from models import MeetingIntelligence
from prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE

logger = logging.getLogger(__name__)


class LLMClient:
    """Async client for Ollama-hosted LLM (OpenAI-compatible API)."""

    def __init__(self):
        config = get_config()
        self.base_url = config.llm_base_url
        self.model = config.llm_model
        self.timeout = config.llm_timeout
        self.max_tokens = config.llm_max_tokens
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.timeout, connect=10.0)
        )

    async def analyze_transcript(
        self, transcript: str, model: Optional[str] = None
    ) -> MeetingIntelligence:
        """Send diarized transcript to LLM and return structured meeting intelligence.

        Args:
            transcript: Full diarized transcript text with speaker labels.
            model: Optional model override (defaults to config).

        Returns:
            MeetingIntelligence with extracted fields.

        Raises:
            httpx.HTTPStatusError: If the LLM API returns an error.
            json.JSONDecodeError: If the LLM response cannot be parsed as JSON.
        """
        t_start = time.perf_counter()
        use_model = model or self.model

        # Truncate very long transcripts to stay within context window
        max_chars = 400_000  # ~100K tokens, well within Granite's 128K context
        if len(transcript) > max_chars:
            logger.warning(
                f"Transcript truncated from {len(transcript)} to {max_chars} chars"
            )
            transcript = transcript[:max_chars]

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": USER_PROMPT_TEMPLATE.format(
                    transcript_text=transcript
                ),
            },
        ]

        logger.info(
            f"Sending transcript ({len(transcript)} chars) to LLM "
            f"model={use_model}"
        )

        response = await self._client.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": use_model,
                "messages": messages,
                "temperature": 0,
                "top_p": 1.0,
                "max_tokens": self.max_tokens,
            },
        )
        response.raise_for_status()

        data = response.json()
        content = data["choices"][0]["message"]["content"]
        t_elapsed = time.perf_counter() - t_start

        # Parse the structured JSON from the LLM response
        try:
            parsed = self._extract_json(content)
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"First JSON parse failed ({e}), retrying with repair prompt")
            parsed = await self._retry_json_parse(content, use_model)

        result = MeetingIntelligence(**parsed)
        result.model = use_model
        result.generation_time_seconds = round(t_elapsed, 2)

        logger.info(
            f"LLM analysis completed in {t_elapsed:.2f}s — "
            f"{len(result.action_items)} actions, "
            f"{len(result.decisions)} decisions, "
            f"{len(result.unresolved_questions)} questions"
        )
        return result

    def _extract_json(self, content: str) -> dict:
        """Extract JSON from Granite response, handling <response> tags.

        Tries in order:
        1. JSON inside <response>...</response> tags
        2. JSON inside markdown code fences
        3. Raw content as JSON
        """
        # Try <response> tags first
        match = re.search(r"<response>(.*?)</response>", content, re.DOTALL)
        if match:
            content = match.group(1).strip()

        # Strip markdown code fences if present
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)

        return json.loads(content)

    async def _retry_json_parse(self, raw_content: str, model: str) -> dict:
        """Retry JSON extraction by asking the LLM to fix its output."""
        messages = [
            {
                "role": "system",
                "content": "You are a JSON repair assistant. Fix the following text so it is valid JSON. Output ONLY the corrected JSON, nothing else.",
            },
            {
                "role": "user",
                "content": f"Fix this JSON:\n{raw_content}",
            },
        ]

        response = await self._client.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": model,
                "messages": messages,
                "temperature": 0,
                "max_tokens": self.max_tokens,
            },
        )
        response.raise_for_status()

        data = response.json()
        content = data["choices"][0]["message"]["content"]
        return self._extract_json(content)

    async def is_available(self) -> bool:
        """Check if the Ollama service is reachable."""
        try:
            resp = await self._client.get(
                f"{self.base_url}/api/tags", timeout=5.0
            )
            return resp.status_code == 200
        except Exception:
            return False

    async def ensure_model_pulled(self):
        """Pull the model if not already available in Ollama.

        This may take several minutes on first run (~5GB download).
        """
        logger.info(f"Ensuring model {self.model} is available in Ollama...")
        try:
            resp = await self._client.post(
                f"{self.base_url}/api/pull",
                json={"name": self.model, "stream": False},
                timeout=httpx.Timeout(600.0, connect=10.0),
            )
            resp.raise_for_status()
            logger.info(f"Model {self.model} ready in Ollama")
        except httpx.TimeoutException:
            logger.error(
                f"Timeout pulling model {self.model} — "
                "model may still be downloading in background"
            )
            raise
        except Exception as e:
            logger.error(f"Failed to pull model {self.model}: {e}")
            raise

    async def close(self):
        """Close the HTTP client."""
        await self._client.aclose()
