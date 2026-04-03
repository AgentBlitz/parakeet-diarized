"""
Entity Extraction Module

Extracts named entities (people, projects, organizations, brands) from
diarized transcripts using a local LLM via an OpenAI-compatible API
(Ollama, vLLM, or any compatible server).

Used by the MeetingFlow pipeline to identify entities before obfuscation.
"""

import json
import logging
import re
import time
from typing import Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

EXTRACTION_PROMPT = """\
You are an entity extractor. Given a meeting transcript, extract all named entities.

Return ONLY a JSON object with these keys:
- "people": list of person names mentioned (full names where possible)
- "projects": list of project or product names
- "organizations": list of company, team, or department names
- "brands": list of brand names

Rules:
- Only include specific, proper names — not generic terms like "the team" or "the project"
- Deduplicate: if "John" and "John Smith" both appear, keep only "John Smith"
- Do not include speaker labels like "Speaker 1"
- Return empty lists for categories with no entities
- Return ONLY the JSON object, no other text

Transcript:
{transcript}"""


async def extract_entities(
    transcript: str,
    llm_base_url: str,
    llm_model: str,
    timeout: float = 120.0,
) -> Dict[str, List[str]]:
    """
    Extract entities from a transcript using a local LLM.

    Args:
        transcript: The diarized transcript text
        llm_base_url: Base URL of the OpenAI-compatible API (e.g. http://localhost:8003/v1)
        llm_model: Model name (e.g. "Qwen3.5-9B-Q4_K_M")
        timeout: Request timeout in seconds

    Returns:
        Dict with keys: people, projects, organizations, brands
        Each value is a list of entity name strings.

    Raises:
        Exception on LLM errors (caller should handle gracefully)
    """
    t_start = time.perf_counter()

    prompt = EXTRACTION_PROMPT.format(transcript=transcript[:15000])  # Cap input size

    payload = {
        "model": llm_model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1,
        "stream": False,
    }

    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(
            f"{llm_base_url}/chat/completions",
            json=payload,
        )
        response.raise_for_status()
        data = response.json()

    content = data["choices"][0]["message"]["content"]

    # Parse JSON from response — handle thinking tags, markdown code blocks,
    # and extra text before/after JSON (common with reasoning models like Qwen3.5)
    content = content.strip()

    # Strip <think>...</think> blocks
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

    if content.startswith("```"):
        # Strip ```json ... ``` wrapper
        lines = content.split("\n")
        content = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        content = content.strip()

    # Find the first valid JSON object in the response
    entities = None
    for match in re.finditer(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", content):
        try:
            candidate = json.loads(match.group())
            if any(k in candidate for k in ("people", "projects", "organizations", "brands")):
                entities = candidate
                break
        except json.JSONDecodeError:
            continue

    if entities is None:
        raise ValueError(f"No valid entity JSON found in LLM response: {content[:200]}")

    # Normalize: ensure all expected keys exist with list values
    result = {
        "people": _dedupe(entities.get("people", [])),
        "projects": _dedupe(entities.get("projects", [])),
        "organizations": _dedupe(entities.get("organizations", [])),
        "brands": _dedupe(entities.get("brands", [])),
    }

    total = sum(len(v) for v in result.values())
    t_elapsed = time.perf_counter() - t_start
    logger.info(f"Entity extraction: {total} entities in {t_elapsed:.2f}s")

    return result


def _dedupe(items: List) -> List[str]:
    """Deduplicate and clean a list of entity names."""
    seen = set()
    result = []
    for item in items:
        if not isinstance(item, str) or not item.strip():
            continue
        normalized = item.strip()
        if normalized.lower() not in seen:
            seen.add(normalized.lower())
            result.append(normalized)
    return result
