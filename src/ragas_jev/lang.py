"""Tiny language guess used to pick per-language calibration maps."""

from __future__ import annotations


def detect_lang(text: str) -> str:
    """'ko' when Hangul makes up at least 20% of the letters, otherwise 'en'."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return "en"
    hangul = sum("가" <= c <= "힣" or "ㄱ" <= c <= "ㆎ" for c in letters)
    return "ko" if hangul / len(letters) >= 0.2 else "en"
