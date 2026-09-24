"""Lenient JSON loading for LLM responses.

Some OpenAI-compatible proxies accept `response_format=json_schema` but do not
enforce it, so models return a variety of shapes. Callers normalize the shape
and validate with pydantic; this module only gets the first JSON value out.
"""

from __future__ import annotations

import json
import re
from typing import Any

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$")


def loads_first(content: str | None) -> Any:
    """Parse the first JSON value in `content`, ignoring code fences and trailing text."""
    text = _FENCE.sub("", (content or "").strip())
    value, _ = json.JSONDecoder().raw_decode(text)
    return value
