"""LLM helper utilities for LangChain/Ollama."""

from __future__ import annotations

import json
import os
from typing import Callable, Optional, Type

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, ValidationError


def _load_chat_model(model: Optional[str] = None, temperature: float = 0.0) -> BaseChatModel:
    """Instantiate a chat model, preferring Gemini 3 Pro when configured."""
    target_model = model or os.environ.get("LLM_MODEL")
    if not target_model:
        if os.environ.get("GOOGLE_API_KEY"):
            # Default to current Google preview name
            target_model = "gemini:gemini-3-pro-preview"
        else:
            raise RuntimeError(
                "No LLM_MODEL provided and GOOGLE_API_KEY is missing. "
                "Set LLM_MODEL (e.g., gemini:gemini-3-pro-preview) and GOOGLE_API_KEY, "
                "or point LLM_MODEL to an available Ollama/OpenAI model."
            )

    if target_model.startswith("ollama:"):
        name = target_model.split("ollama:", 1)[1]
        try:
            from langchain_community.chat_models import ChatOllama
        except ImportError as exc:
            raise RuntimeError(
                "ChatOllama not available. Install langchain-community or set LLM_MODEL to an OpenAI/Gemini model."
            ) from exc
        return ChatOllama(model=name, temperature=temperature)

    if target_model.startswith("gemini:"):
        # Expected format gemini:<model-name>, e.g., gemini:gemini-1.5-pro
        name = target_model.split("gemini:", 1)[1]
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as exc:
            raise RuntimeError(
                "Gemini requested but langchain-google-genai is missing. Install it or change LLM_MODEL."
            ) from exc
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY is required for Gemini models.")
        return ChatGoogleGenerativeAI(model=name, temperature=temperature, api_key=api_key)

    # Fallback to OpenAI-compatible models if provided
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise RuntimeError(
            "OpenAI chat model requested but langchain-openai is missing. "
            "Install langchain-openai or use ollama:<model>."
        ) from exc

    return ChatOpenAI(model=target_model, temperature=temperature)


def call_llm(
    prompt,
    pydantic_model: Type[BaseModel],
    model: Optional[str] = None,
    temperature: float = 0.0,
    default_factory: Optional[Callable[[], BaseModel]] = None,
    debug: bool = False,
    agent_name: Optional[str] = None,
):
    """Call an LLM and parse JSON into a Pydantic model."""
    chat_model = _load_chat_model(model=model, temperature=temperature)
    parser = StrOutputParser()

    # Accept ChatPromptTemplate or raw string
    chain = prompt | chat_model | parser if isinstance(prompt, ChatPromptTemplate) else chat_model | parser
    raw = chain.invoke(prompt if not isinstance(prompt, ChatPromptTemplate) else {})

    if debug:
        who = agent_name or "llm"
        print(f"\n[LLM raw][{who}]\n{raw}\n")

    try:
        # Attempt direct JSON parsing
        payload = json.loads(raw)
        return pydantic_model.model_validate(payload)
    except (json.JSONDecodeError, ValidationError):
        if default_factory:
            return default_factory()
        raise
