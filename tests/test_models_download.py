"""Tests for model downloading and verifying file sizes."""

from pathlib import Path

import pytest

from fake_llm_server._models import SUPPORTED_MODEL_ALIASES, ModelSpec

# Approximate expected minimum sizes in bytes
# 3B ~ 2GB, 1.5B ~ 1GB, 1B ~ 600MB, 270M ~ 150MB
MIN_SIZES = {
    "qwen-2.5-coder-3b": 1.5 * 1024 * 1024 * 1024,
    "qwen-2.5-coder-1.5b": 0.8 * 1024 * 1024 * 1024,
    "llama-3.2-3b-instruct": 1.5 * 1024 * 1024 * 1024,
    "smollm3": 1.5 * 1024 * 1024 * 1024,  # It says 3B in README but key is smollm3
    "gemma-3-1b": 0.5 * 1024 * 1024 * 1024,
    "gemma-3-270m": 100 * 1024 * 1024,
}


@pytest.mark.parametrize("model_name", list(SUPPORTED_MODEL_ALIASES.keys()))
def test_model_download_and_size(model_name: str) -> None:
    """Verifies that each model can be downloaded.

    Checks if the file size is within expected bounds.

    Args:
        model_name: The name of the model to test.
    """
    try:
        path_str = ModelSpec.from_name(model_name).download().model_path
        path = Path(path_str)
        assert path.exists()

        size = path.stat().st_size
        min_size = MIN_SIZES.get(model_name, 100 * 1024 * 1024)  # Default 100MB

        assert size > min_size, (
            f"Model {model_name} file size {size} is smaller than expected {min_size}"
        )
    except (RuntimeError, ValueError) as e:
        pytest.fail(f"Download failed for {model_name}: {e}")
