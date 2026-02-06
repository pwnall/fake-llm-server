"""
Tests for model name translation logic.
"""

from unittest.mock import MagicMock, patch

import pytest

from fake_llm_server.models import SUPPORTED_MODELS, RemoteModelSpec


def test_translate_known_alias() -> None:
    """
    Verifies translation of a known model alias.
    """
    spec = RemoteModelSpec.from_name("gemma-3-270m")
    assert spec == SUPPORTED_MODELS["gemma-3-270m"]
    assert spec.repo_id == "unsloth/gemma-3-270m-it-GGUF"
    assert spec.filename == "gemma-3-270m-it-Q4_K_M.gguf"


def test_translate_unknown_alias_raises() -> None:
    """
    Verifies that an unknown alias raises a ValueError.
    """
    with pytest.raises(ValueError, match="Model 'unknown-model' not supported"):
        RemoteModelSpec.from_name("unknown-model")


@patch("fake_llm_server.models.list_repo_files")
def test_translate_repo_id_with_q4_k_m(mock_list_files: MagicMock) -> None:
    """
    Verifies translation of a repo ID when a Q4_K_M file is available.

    Args:
        mock_list_files: Mocked list_repo_files function.
    """
    repo_id = "someuser/some-repo"
    mock_list_files.return_value = [
        "model.f16.gguf",
        "model.q4_k_m.gguf",
        "README.md",
    ]

    spec = RemoteModelSpec.from_name(repo_id)
    assert spec.repo_id == repo_id
    assert spec.filename == "model.q4_k_m.gguf"


@patch("fake_llm_server.models.list_repo_files")
def test_translate_repo_id_without_preference(mock_list_files: MagicMock) -> None:
    """
    Verifies translation of a repo ID when no preferred file is available.

    Args:
        mock_list_files: Mocked list_repo_files function.
    """
    repo_id = "someuser/other-repo"
    mock_list_files.return_value = [
        "model.q8_0.gguf",
        "README.md",
    ]

    spec = RemoteModelSpec.from_name(repo_id)
    assert spec.repo_id == repo_id
    assert spec.filename == "model.q8_0.gguf"


@patch("fake_llm_server.models.list_repo_files")
def test_translate_repo_id_no_gguf(mock_list_files: MagicMock) -> None:
    """
    Verifies that a repo ID with no GGUF files raises a ValueError.

    Args:
        mock_list_files: Mocked list_repo_files function.
    """
    repo_id = "someuser/empty-repo"
    mock_list_files.return_value = ["README.md", "config.json"]

    with pytest.raises(ValueError, match="No .gguf files found"):
        RemoteModelSpec.from_name(repo_id)


@patch("fake_llm_server.models.list_repo_files")
def test_translate_repo_id_list_fails(mock_list_files: MagicMock) -> None:
    """
    Verifies that an error during file listing raises a ValueError.

    Args:
        mock_list_files: Mocked list_repo_files function.
    """
    repo_id = "someuser/bad-repo"
    mock_list_files.side_effect = Exception("Network error")

    with pytest.raises(ValueError, match="Could not list files"):
        RemoteModelSpec.from_name(repo_id)