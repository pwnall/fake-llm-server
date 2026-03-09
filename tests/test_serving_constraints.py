"""Tests for verifying FakeLLMServer configuration constraints."""

import pytest

from fake_llm_server import open_fake_llm_server


def test_model_names_must_be_tuple() -> None:
    """Verifies that model_names must be a tuple."""
    with (
        pytest.raises(TypeError, match="model_names must be a tuple"),
        open_fake_llm_server(model_names=["gemma-3-270m"]),  # type: ignore[arg-type]
    ):
        pass


def test_aliases_values_must_be_in_model_names() -> None:
    """Verifies that alias values must be present in model_names."""
    with (
        pytest.raises(
            ValueError, match="Alias target 'unknown-model' not in model_names"
        ),
        open_fake_llm_server(
            model_names=("gemma-3-270m",),
            aliases={"my-alias": "unknown-model"},
        ),
    ):
        pass


def test_transitive_aliases_not_supported() -> None:
    """Verifies that transitive aliases are not supported."""
    # This is implicitly covered by test_aliases_values_must_be_in_model_names
    # because the target alias won't be in model_names.
    with (
        pytest.raises(ValueError, match="Alias target 'alias-1' not in model_names"),
        open_fake_llm_server(
            model_names=("gemma-3-270m",),
            aliases={
                "alias-1": "gemma-3-270m",
                "alias-2": "alias-1",
            },
        ),
    ):
        pass


def test_valid_configuration() -> None:
    """Verifies that a valid configuration is accepted."""
    with open_fake_llm_server(
        model_names=("gemma-3-270m",),
        aliases={"my-gemma": "gemma-3-270m"},
    ):
        pass
