from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ProviderConfig:
    """Student TODO: define the provider configuration shared by the agents.

    Required providers for this lab:
    - openai
    - custom (OpenAI-compatible base URL)
    - gemini
    - anthropic
    - ollama
    - openrouter
    """

    provider: str
    model_name: str
    temperature: float
    api_key: str | None = None
    base_url: str | None = None
    timeout_seconds: float = 30.0
    max_retries: int = 1


def normalize_provider(value: str) -> str:
    """Student TODO: map aliases like `anthorpic` -> `anthropic`."""
    normalized = value.strip().lower().replace("-", "_")
    aliases = {
        "anthorpic": "anthropic",
        "google": "gemini",
        "google_genai": "gemini",
        "open_router": "openrouter",
        "openai_compatible": "custom",
    }
    normalized = aliases.get(normalized, normalized)
    supported = {"openai", "custom", "gemini", "anthropic", "ollama", "openrouter"}
    if normalized not in supported:
        choices = ", ".join(sorted(supported))
        raise ValueError(f"Unsupported provider {value!r}. Choose one of: {choices}")
    return normalized


def build_chat_model(config: ProviderConfig):
    """Student TODO: instantiate the real chat model for the selected provider.

    Pseudocode:
    - `openai` -> `ChatOpenAI`
    - `custom` -> `ChatOpenAI` with `base_url`
    - `gemini` -> `ChatGoogleGenerativeAI`
    - `anthropic` -> `ChatAnthropic`
    - `ollama` -> `ChatOllama`
    - `openrouter` -> `ChatOpenRouter`
    """

    provider = normalize_provider(config.provider)
    common = {"model": config.model_name, "temperature": config.temperature}

    if provider in {"openai", "custom"}:
        from langchain_openai import ChatOpenAI

        common["timeout"] = config.timeout_seconds
        common["max_retries"] = config.max_retries
        if config.api_key:
            common["api_key"] = config.api_key
        if provider == "custom":
            if not config.base_url:
                raise ValueError("CUSTOM_BASE_URL is required for the custom provider")
            common["base_url"] = config.base_url
        return ChatOpenAI(**common)

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        common["request_timeout"] = config.timeout_seconds
        common["retries"] = config.max_retries
        if config.api_key:
            common["google_api_key"] = config.api_key
        return ChatGoogleGenerativeAI(**common)

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        common["timeout"] = config.timeout_seconds
        common["max_retries"] = config.max_retries
        if config.api_key:
            common["api_key"] = config.api_key
        return ChatAnthropic(**common)

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        if config.base_url:
            common["base_url"] = config.base_url
        return ChatOllama(**common)

    from langchain_openrouter import ChatOpenRouter

    common["timeout"] = config.timeout_seconds
    common["max_retries"] = config.max_retries
    if config.api_key:
        common["api_key"] = config.api_key
    return ChatOpenRouter(**common)
