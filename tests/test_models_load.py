"""Tests for loading models with Llama.cpp."""

from unittest.mock import MagicMock, patch

from fake_llm_server._models import DownloadedModel


@patch("fake_llm_server._models.Llama")
@patch("fake_llm_server._models.psutil")
@patch("fake_llm_server._models.os")
def test_load_sets_n_threads(
    mock_os: MagicMock, mock_psutil: MagicMock, mock_llama: MagicMock
) -> None:
    """Verifies that load() sets n_threads based on psutil.cpu_count()."""
    mock_psutil.cpu_count.return_value = 4
    mock_os.process_cpu_count.return_value = 2  # Should not be used
    model = DownloadedModel(model_name="test-model", model_path="/path/to/model.gguf")

    model.load()

    mock_llama.assert_called_once()
    kwargs = mock_llama.call_args.kwargs
    assert kwargs["n_threads"] == 4


@patch("fake_llm_server._models.Llama")
@patch("fake_llm_server._models.psutil")
@patch("fake_llm_server._models.os")
def test_load_fallback_os_process_cpu_count(
    mock_os: MagicMock, mock_psutil: MagicMock, mock_llama: MagicMock
) -> None:
    """Verifies that load() fallbacks to os.process_cpu_count()."""
    mock_psutil.cpu_count.return_value = None
    mock_os.process_cpu_count.return_value = 6
    model = DownloadedModel(model_name="test-model", model_path="/path/to/model.gguf")

    model.load()

    mock_llama.assert_called_once()
    kwargs = mock_llama.call_args.kwargs
    assert kwargs["n_threads"] == 6


@patch("fake_llm_server._models.Llama")
@patch("fake_llm_server._models.psutil")
@patch("fake_llm_server._models.os")
def test_load_fallback_minimum(
    mock_os: MagicMock, mock_psutil: MagicMock, mock_llama: MagicMock
) -> None:
    """Verifies that load() fallbacks to 1 thread if everything fails."""
    mock_psutil.cpu_count.return_value = None
    mock_os.process_cpu_count.return_value = None
    model = DownloadedModel(model_name="test-model", model_path="/path/to/model.gguf")

    model.load()

    mock_llama.assert_called_once()
    kwargs = mock_llama.call_args.kwargs
    assert kwargs["n_threads"] == 1


@patch("fake_llm_server._models.Llama")
@patch("fake_llm_server._models.psutil")
def test_load_allows_override_n_threads(
    mock_psutil: MagicMock, mock_llama: MagicMock
) -> None:
    """Verifies that load() allows overriding n_threads."""
    mock_psutil.cpu_count.return_value = 4
    model = DownloadedModel(model_name="test-model", model_path="/path/to/model.gguf")

    model.load(n_threads=8)

    mock_llama.assert_called_once()
    kwargs = mock_llama.call_args.kwargs
    assert kwargs["n_threads"] == 8
