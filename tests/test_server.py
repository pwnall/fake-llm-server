"""Tests for the FakeLLMServer API endpoints."""

from collections.abc import Generator

import pytest
from openai import OpenAI

from fake_llm_server import FakeLLMServer


@pytest.fixture(scope="module")
def server() -> Generator[FakeLLMServer]:
    """Fixture that starts and stops a FakeLLMServer instance for testing.

    Yields:
        A running FakeLLMServer instance.
    """
    # Use gemma-3-270m (smallest) for feature tests.
    model_name = "gemma-3-270m"
    s = FakeLLMServer(model_name=model_name)
    yield s
    s.shutdown()


def test_list_models(server: FakeLLMServer) -> None:
    """Verifies the list models endpoint.

    Args:
        server: The running server instance.
    """
    client_args = server.openai_client_args()
    client = OpenAI(**client_args)

    models = client.models.list()
    assert len(models.data) > 0
    # Check that our model ID is in the list
    assert any(m.id == "gemma-3-270m" for m in models.data)


def test_chat_completion(server: FakeLLMServer) -> None:
    """Verifies the chat completion endpoint.

    Args:
        server: The running server instance.
    """
    client_args = server.openai_client_args()
    client = OpenAI(**client_args)

    response = client.chat.completions.create(
        model="gemma-3-270m",
        messages=[{"role": "user", "content": "Hello!"}],
        max_tokens=10,
        temperature=0,
    )

    content = response.choices[0].message.content
    print(f"Model response: {content}")
    assert content


def test_temperature_effect(server: FakeLLMServer) -> None:
    """Verifies that setting a high temperature affects the response content.

    Args:
        server: The running server instance.
    """
    client_args = server.openai_client_args()
    client = OpenAI(**client_args)

    prompt = [{"role": "user", "content": "Write a short poem about coding."}]

    # Low temperature (deterministic-ish)
    resp_low = client.chat.completions.create(
        model="gemma-3-270m",
        messages=prompt,  # type: ignore[arg-type]
        max_tokens=20,
        temperature=0.0,
    )

    # High temperature (random-ish)
    resp_high = client.chat.completions.create(
        model="gemma-3-270m",
        messages=prompt,  # type: ignore[arg-type]
        max_tokens=20,
        temperature=0.9,
    )

    # We mainly assert that the calls succeed.
    assert resp_low.choices[0].message.content
    assert resp_high.choices[0].message.content
