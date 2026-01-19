"""
LLM Provider for Sentinel-AI

Unified interface for Large Language Model integration.
Supports Gemini 1.5 Pro (primary) and GPT-4o (fallback).
Includes automatic provider switching, rate limiting, and error handling.
"""

import os
from typing import Optional, Dict, Any
from enum import Enum

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel


class LLMProviderType(str, Enum):
    """Supported LLM providers"""
    GEMINI = "gemini"
    OPENAI = "openai"


class LLMConfig(BaseModel):
    """Configuration for LLM provider"""
    provider: LLMProviderType = LLMProviderType.GEMINI
    temperature: float = 0.1
    max_tokens: int = 4096
    timeout: int = 60


class LLMProvider:
    """
    Unified LLM provider with automatic fallback.
    
    Usage:
        provider = LLMProvider()
        response = provider.generate("Analyze this content...")
    """
    
    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self._primary_llm: Optional[BaseChatModel] = None
        self._fallback_llm: Optional[BaseChatModel] = None
        self._initialize_providers()
    
    def _initialize_providers(self) -> None:
        """Initialize LLM providers based on available API keys."""
        google_api_key = os.getenv("GOOGLE_API_KEY")
        openai_api_key = os.getenv("OPENAI_API_KEY")
        default_provider = os.getenv("DEFAULT_LLM_PROVIDER", "gemini").lower()
        
        # Try to initialize Gemini
        if google_api_key:
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
                gemini = ChatGoogleGenerativeAI(
                    model="gemini-1.5-pro",
                    google_api_key=google_api_key,
                    temperature=self.config.temperature,
                    max_output_tokens=self.config.max_tokens,
                    timeout=self.config.timeout
                )
                if default_provider == "gemini":
                    self._primary_llm = gemini
                else:
                    self._fallback_llm = gemini
            except Exception as e:
                print(f"⚠️ Failed to initialize Gemini: {e}")
        
        # Try to initialize OpenAI
        if openai_api_key:
            try:
                from langchain_openai import ChatOpenAI
                openai = ChatOpenAI(
                    model="gpt-4o",
                    api_key=openai_api_key,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    timeout=self.config.timeout
                )
                if default_provider == "openai" or self._primary_llm is None:
                    self._primary_llm = openai
                elif self._fallback_llm is None:
                    self._fallback_llm = openai
            except Exception as e:
                print(f"⚠️ Failed to initialize OpenAI: {e}")
        
        # Validate at least one provider is available
        if self._primary_llm is None and self._fallback_llm is None:
            raise ValueError(
                "No LLM provider available! Please set GOOGLE_API_KEY or OPENAI_API_KEY "
                "environment variable."
            )
        
        # If only fallback is set, make it primary
        if self._primary_llm is None:
            self._primary_llm = self._fallback_llm
            self._fallback_llm = None
        
        print(f"✅ LLM Provider initialized: Primary={self._get_provider_name(self._primary_llm)}")
        if self._fallback_llm:
            print(f"   Fallback={self._get_provider_name(self._fallback_llm)}")
    
    def _get_provider_name(self, llm: BaseChatModel) -> str:
        """Get the name of the LLM provider."""
        class_name = llm.__class__.__name__
        if "Google" in class_name or "Gemini" in class_name:
            return "Gemini 1.5 Pro"
        elif "OpenAI" in class_name:
            return "GPT-4o"
        return class_name
    
    def generate(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None,
        use_fallback: bool = True
    ) -> str:
        """
        Generate a response from the LLM.
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt for context
            use_fallback: Whether to try fallback provider on failure
            
        Returns:
            Generated text response
        """
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))
        
        try:
            response = self._primary_llm.invoke(messages)
            return response.content
        except Exception as e:
            print(f"⚠️ Primary LLM failed: {e}")
            
            if use_fallback and self._fallback_llm:
                print("🔄 Trying fallback LLM...")
                try:
                    response = self._fallback_llm.invoke(messages)
                    return response.content
                except Exception as fallback_error:
                    print(f"❌ Fallback LLM also failed: {fallback_error}")
                    raise
            raise
    
    def generate_structured(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        output_schema: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate a structured JSON response from the LLM.
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            output_schema: Expected output schema (for validation)
            
        Returns:
            Parsed JSON response as dictionary
        """
        import json
        
        json_prompt = f"""
{prompt}

IMPORTANT: Respond ONLY with valid JSON. No markdown, no explanation.
"""
        
        response = self.generate(json_prompt, system_prompt)
        
        # Clean up response (remove markdown code blocks if present)
        response = response.strip()
        if response.startswith("```json"):
            response = response[7:]
        if response.startswith("```"):
            response = response[3:]
        if response.endswith("```"):
            response = response[:-3]
        response = response.strip()
        
        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            print(f"⚠️ Failed to parse JSON response: {e}")
            print(f"Raw response: {response[:500]}...")
            raise ValueError(f"Invalid JSON response from LLM: {e}")
    
    @property
    def llm(self) -> BaseChatModel:
        """Get the underlying LLM instance for LangGraph/LangChain use."""
        return self._primary_llm
    
    def health_check(self) -> bool:
        """Check if the LLM provider is working."""
        try:
            response = self.generate("Say 'OK' if you're working.")
            return "ok" in response.lower()
        except Exception:
            return False


# Singleton instance for convenience
_default_provider: Optional[LLMProvider] = None


def get_llm(config: Optional[LLMConfig] = None) -> LLMProvider:
    """
    Get the default LLM provider instance.
    Creates a new one if not already initialized.
    """
    global _default_provider
    if _default_provider is None or config is not None:
        _default_provider = LLMProvider(config)
    return _default_provider
