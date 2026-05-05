"""SPINE Models - LLM model definitions and data models."""

# OpenRouter model identifiers (use OpenAI-compatible format)
OPENROUTER_MODELS = {
    "openai-gpt-4": "openai/gpt-4",
    "openai-gpt-4-turbo": "openai/gpt-4-turbo",
    "openai-gpt-3.5-turbo": "openai/gpt-3.5-turbo",
    "anthropic-claude-3-opus": "anthropic/claude-3-opus",
    "anthropic-claude-3-sonnet": "anthropic/claude-3-sonnet",
    "anthropic-claude-3-haiku": "anthropic/claude-3-haiku",
    "google-gemini-pro": "google/gemini-pro",
    "google-gemini-flash": "google/gemini-flash",
    "meta-llama-3-70b": "meta/llama-3-70b-instruct",
    "meta-llama-3-8b": "meta/llama-3-8b-instruct",
    "mistral-large": "mistralai/mistral-large",
    "mistral-medium": "mistralai/mistral-medium",
}

# Common defaults
DEFAULT_OPENROUTER_MODEL = OPENROUTER_MODELS["openai-gpt-4"]
DEFAULT_LOCAL_MODEL = "local-model"
DEFAULT_OPENAI_MODEL = "gpt-4"

# Import data models
from .enums import PhaseName, StateStatus, SubPhaseStatus
from .types import Task, SubPhase, Phase, PhaseResult, SubPhaseResult, SpineState
from .dag import SwarmDAGExecutor

__all__ = [
    # LLM models
    "OPENROUTER_MODELS",
    "DEFAULT_OPENROUTER_MODEL",
    "DEFAULT_LOCAL_MODEL",
    "DEFAULT_OPENAI_MODEL",
    # Data models - enums
    "PhaseName",
    "StateStatus",
    "SubPhaseStatus",
    # Data models - types
    "Task",
    "SubPhase",
    "Phase",
    "PhaseResult",
    "SubPhaseResult",
    "SpineState",
    # Data models - dag
    "SwarmDAGExecutor",
]