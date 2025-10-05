import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence

from openai import OpenAI


def _content_to_text(content: Any) -> str:
    """Normalize message content fragments into a single text string."""

    if isinstance(content, str):
        return content
    if isinstance(content, list):
        pieces: List[str] = []
        for item in content:
            if isinstance(item, dict):
                text_value = item.get("text")
                if text_value:
                    pieces.append(str(text_value))
            elif item:
                pieces.append(str(item))
        return "\n".join(piece for piece in pieces if piece)
    if content is None:
        return ""
    return str(content)


def _completion_to_text(completion: Any) -> str:
    """Extract concatenated text from a chat completion response."""

    fragments: List[str] = []
    for choice in getattr(completion, "choices", []) or []:
        message = getattr(choice, "message", None)
        if message is None:
            continue
        fragments.append(_content_to_text(getattr(message, "content", "")))
    text = "\n".join(fragment.strip() for fragment in fragments if fragment and fragment.strip()).strip()
    if not text:
        raise RuntimeError("No text output returned by the LLM")
    return text


def _safe_structured_parse(completion: Any, response_model: Any):
    for choice in getattr(completion, "choices", []) or []:
        message = getattr(choice, "message", None)
        if message is None:
            continue
        parsed = getattr(message, "parsed", None)
        if parsed is not None:
            if isinstance(parsed, response_model):
                return parsed
            return response_model.model_validate(parsed)
    # Fall back to validating from raw JSON content when no parsed payload is present.
    raw_text = _completion_to_text(completion)
    return response_model.model_validate_json(raw_text)


class BaseLLM(ABC):
    """Abstract base interface for LLM wrappers."""

    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    def parse_structured(self, messages: Sequence[Dict[str, str]], response_model: Any) -> Any:
        """Return a structured response parsed into the provided Pydantic model."""

    @abstractmethod
    def complete_text(self, messages: Sequence[Dict[str, str]]) -> str:
        """Return plain-text completion output for the conversation."""


class GeminiLLM(BaseLLM):
    """LLM wrapper that routes Gemini models through the OpenAI compatibility API."""

    def __init__(self, model: str, api_key: str, base_url: str):
        super().__init__(model)
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    def parse_structured(self, messages: Sequence[Dict[str, str]], response_model: Any) -> Any:
        completion = self._client.beta.chat.completions.parse(
            model=self.model_name,
            messages=list(messages),
            response_format=response_model,
        )
        return _safe_structured_parse(completion, response_model)

    def complete_text(self, messages: Sequence[Dict[str, str]]) -> str:
        completion = self._client.chat.completions.create(
            model=self.model_name,
            messages=list(messages),
        )
        return _completion_to_text(completion)


class OpenAILLM(BaseLLM):
    """LLM wrapper for native OpenAI chat models."""

    def __init__(self, model: str, api_key: str):
        super().__init__(model)
        self._client = OpenAI(api_key=api_key)

    def parse_structured(self, messages: Sequence[Dict[str, str]], response_model: Any) -> Any:
        completion = self._client.responses.parse(
            model=self.model_name,
            input=list(messages),
            text_format=response_model,
        )
        return _safe_structured_parse(completion, response_model)

    def complete_text(self, messages: Sequence[Dict[str, str]]) -> str:
        completion = self._client.chat.completions.create(
            model=self.model_name,
            messages=list(messages),
        )
        return _completion_to_text(completion)


def get_llm(model_name: Optional[str] = None) -> BaseLLM:
    """Instantiate the appropriate LLM wrapper based on the requested model."""
    if model_name is None:
        model_name = "gemini-2.5-flash"  # default model

    if model_name.lower() in ["gemini-2.5-flash"]:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise EnvironmentError(f"GOOGLE_API_KEY  must be set to use {model_name} models")

        return GeminiLLM(
            model=model_name, api_key=api_key, base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )
    if model_name.lower() in ["gpt-4o-mini"]:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(f"OPENAI_API_KEY must be set to use {model_name} models")
        return OpenAILLM(model=model_name, api_key=api_key)

    raise NotImplementedError(f"Model '{model_name}' is not supported by the LLM registry")
