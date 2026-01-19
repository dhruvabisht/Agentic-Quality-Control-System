"""
Sentinel-AI LLM Module

Unified LLM provider interface supporting:
- Google Gemini 1.5 Pro (primary)
- OpenAI GPT-4o (fallback)
"""

from .provider import LLMProvider, get_llm

__all__ = ["LLMProvider", "get_llm"]
