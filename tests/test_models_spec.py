"""Tests for the _RemoteModelSpec class."""

from unittest.mock import MagicMock, patch

import pytest

from fake_llm_server._models import SUPPORTED_MODEL_ALIASES, ModelSpec


def test_from_name_supported_model() -> None:
    """Test from_name with a known supported model name."""
    # Pick a key from SUPPORTED_MODEL_ALIASES
    model_name = "qwen-2.5-coder-1.5b"
    spec = ModelSpec.from_name(model_name)
    assert spec == SUPPORTED_MODEL_ALIASES[model_name]
    assert spec.repo_id == "Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF"


@patch("fake_llm_server._models.list_repo_files")
def test_from_name_repo_id_success(mock_list_files: MagicMock) -> None:
    """Test from_name with a valid repo ID."""
    repo_id = "someuser/somerepo"
    mock_list_files.return_value = [
        "config.json",
        "model-q4_k_m.gguf",
        "README.md",
    ]

    spec = ModelSpec.from_name(repo_id)
    assert spec.repo_id == repo_id
    assert spec.filename == "model-q4_k_m.gguf"


def test_from_name_invalid_name() -> None:
    """Test from_name with an invalid name (no slash, not in supported)."""
    with pytest.raises(ValueError, match="Model 'invalid-model' not supported"):
        ModelSpec.from_name("invalid-model")


@patch("fake_llm_server._models.list_repo_files")
def test_from_name_repo_no_gguf(mock_list_files: MagicMock) -> None:
    """Test from_name with a repo having no gguf files."""
    repo_id = "someuser/emptyrepo"
    mock_list_files.return_value = ["README.md", "config.json"]

    with pytest.raises(ValueError, match=f"No .gguf files found in {repo_id}"):
        ModelSpec.from_name(repo_id)


@patch("fake_llm_server._models.list_repo_files")
def test_from_repo_id_method(mock_list_files: MagicMock) -> None:
    """Test the new from_repo_id method directly."""
    repo_id = "user/repo"
    mock_list_files.return_value = ["file.gguf"]

    spec = ModelSpec.from_repo_id(repo_id)
    assert spec.repo_id == repo_id
    assert spec.filename == "file.gguf"
