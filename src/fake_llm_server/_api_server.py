"""FastAPI server for the Fake LLM."""

from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from typing import Any

import jinja2
from fastapi import FastAPI, HTTPException, Request
from llama_cpp import Llama, llama_chat_format
from llama_cpp.llama_types import (
    ChatCompletion,
    ChatCompletionChunk,
)
from pydantic import BaseModel, ConfigDict

from ._models import ModelSpec

# Monkeypatch Jinja2ChatFormatter to survive bad templates (like SmolLM3's)
_original_jinja2_formatter_init = llama_chat_format.Jinja2ChatFormatter.__init__


def _safe_jinja2_formatter_init(
    self: Any,  # noqa: ANN401
    template: str,
    eos_token: str,
    bos_token: str,
    *args: Any,  # noqa: ANN401
    **kwargs: Any,  # noqa: ANN401
) -> None:
    """Initializes Jinja2ChatFormatter with error handling for malformed templates.

    Args:
        self: The formatter instance.
        template: The chat template string.
        eos_token: The end-of-sequence token.
        bos_token: The beginning-of-sequence token.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.
    """
    try:
        _original_jinja2_formatter_init(
            self,
            template,
            eos_token,
            bos_token,
            *args,
            **kwargs,
        )
    except (jinja2.exceptions.TemplateSyntaxError, TypeError):
        # Initialize with a dummy template to avoid attribute errors later if used
        self.template = template
        self.eos_token = eos_token
        self.bos_token = bos_token


llama_chat_format.Jinja2ChatFormatter.__init__ = _safe_jinja2_formatter_init  # type: ignore[method-assign]


type ServingConfiguration = dict[str, Llama]


def parse_server_args(
    model_names: tuple[str, ...] = ("gemma-3-270m",),
    aliases: dict[str, str] = {},  # noqa: B006
) -> ServingConfiguration:
    """Parses server arguments and prepares the serving configuration.

    Args:
        model_names: Tuple of model names or repo IDs to use.
        aliases: Dictionary mapping aliases to model names/repo IDs.

    Returns:
        A ServingConfiguration mapping model names/aliases to Llama instances.

    Raises:
        TypeError: If model_names is not a tuple.
        ValueError: If an alias target is not in model_names.
    """
    if not isinstance(model_names, tuple):
        msg = "model_names must be a tuple"
        raise TypeError(msg)

    # Validate aliases
    for target in aliases.values():
        if target not in model_names:
            msg = f"Alias target '{target}' not in model_names"
            raise ValueError(msg)

    downloaded_specs = {}
    loaded_models = {}
    llms: ServingConfiguration = {}

    # Download and load all models in model_names
    for name in model_names:
        spec = ModelSpec.from_name(name).download()
        downloaded_specs[name] = spec

        if spec.model_path not in loaded_models:
            loaded_models[spec.model_path] = spec.load()

        llms[name] = loaded_models[spec.model_path]

    # Map aliases
    for alias, target in aliases.items():
        # Target is guaranteed to be in model_names and thus in llms
        llms[alias] = llms[target]

    return llms


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manages the lifespan of the FastAPI application.

    Args:
        app: The FastAPI application instance.
    """
    llms: ServingConfiguration = getattr(app.state, "llms", {})
    if not llms:
        msg = "llms configuration missing"
        raise RuntimeError(msg)

    yield
    if hasattr(app.state, "llms"):
        del app.state.llms


class _ChatMessage(BaseModel):
    """Represents a single message in a chat completion request."""

    role: str
    content: str


class _ChatCompletionRequest(BaseModel):
    """Represents a request to create a chat completion."""

    model: str
    messages: list[_ChatMessage]
    stream: bool = False
    max_tokens: int | None = None
    temperature: float = 0.0
    top_p: float = 0.95

    # Allow extra fields without failure
    model_config = ConfigDict(extra="ignore")


def create_server_app(llms: ServingConfiguration) -> FastAPI:
    """Creates a FastAPI application for serving LLM models.

    Args:
        llms: A dictionary mapping model aliases to Llama instances.

    Returns:
        A FastAPI application.
    """
    app = FastAPI(lifespan=_lifespan)
    app.state.llms = llms

    @app.post("/v1/chat/completions", response_model=None)
    def create_chat_completion(
        request: _ChatCompletionRequest,
    ) -> ChatCompletion | Iterator[ChatCompletionChunk]:
        """Endpoint for creating chat completions.

        Args:
            request: The chat completion request.

        Returns:
            The generated chat completion response.

        Raises:
            HTTPException: If the model is not loaded or an error occurs during
                inference.
        """
        llms: dict[str, Llama] = getattr(app.state, "llms", {})
        llm = llms.get(request.model)

        if not llm:
            raise HTTPException(
                status_code=404, detail=f"Model '{request.model}' not found"
            )

        messages = [msg.model_dump() for msg in request.messages]

        try:
            return llm.create_chat_completion(
                messages=messages,  # type: ignore[arg-type]
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                stream=request.stream,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.get("/v1/models")
    def list_models(request: Request) -> dict[str, Any]:
        """Endpoint for listing available models.

        Args:
            request: The FastAPI request instance.

        Returns:
            A dictionary containing the list of available models.
        """
        llms: dict[str, Llama] = getattr(request.app.state, "llms", {})
        return {
            "object": "list",
            "data": [
                {
                    "id": model_id,
                    "object": "model",
                    "created": 1677610602,
                    "owned_by": "fake-llm-server",
                }
                for model_id in llms
            ],
        }

    return app
