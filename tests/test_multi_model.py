"""Tests for multi-model support and aliases."""

import gc

from openai import OpenAI

from fake_llm_server import FakeLLMServer


def test_multi_model_support() -> None:
    """Verifies that the server can handle multiple models."""
    # Using the two smallest models to minimize memory usage during tests
    model_names = ["gemma-3-270m", "gemma-3-1b"]
    server = FakeLLMServer(model_names=model_names)

    try:
        client = OpenAI(**server.openai_client_args())

        # Check list models
        models = client.models.list()
        model_ids = [m.id for m in models.data]
        assert "gemma-3-270m" in model_ids
        assert "gemma-3-1b" in model_ids

        # Test completion for first model
        resp1 = client.chat.completions.create(
            model="gemma-3-270m",
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=5,
        )
        assert resp1.choices[0].message.content

        # Test completion for second model
        resp2 = client.chat.completions.create(
            model="gemma-3-1b",
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=5,
        )
        assert resp2.choices[0].message.content

    finally:
        server.shutdown()
        gc.collect()


def test_aliases() -> None:
    """Verifies that aliases work correctly."""
    model_names = ["gemma-3-270m"]
    aliases = {"my-gemma": "gemma-3-270m"}
    server = FakeLLMServer(model_names=model_names, aliases=aliases)

    try:
        client = OpenAI(**server.openai_client_args())

        # Check list models
        models = client.models.list()
        model_ids = [m.id for m in models.data]
        assert "gemma-3-270m" in model_ids
        assert "my-gemma" in model_ids

        # Test completion using alias
        resp = client.chat.completions.create(
            model="my-gemma",
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=5,
        )
        assert resp.choices[0].message.content

    finally:
        server.shutdown()
        gc.collect()
