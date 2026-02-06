"""
Tests for model downloading and verifying file sizes.
"""

import os

import pytest

from fake_llm_server.models import SUPPORTED_MODELS, RemoteModelSpec

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


@pytest.mark.parametrize("model_name", list(SUPPORTED_MODELS.keys()))
def test_model_download_and_size(model_name: str) -> None:
    """
    Verifies that each model can be downloaded and the file size is within
    expected bounds.

    Args:
        model_name: The name of the model to test.
    """
    print(f"Testing download for {model_name}...")
    try:
        path = RemoteModelSpec.from_name(model_name).download().model_path
        assert os.path.exists(path)

        size = os.path.getsize(path)
        min_size = MIN_SIZES.get(model_name, 100 * 1024 * 1024)  # Default 100MB

        assert size > min_size, (
            f"Model {model_name} file size {size} is smaller than expected {min_size}"
        )
        print(f"Model {model_name} OK: {size} bytes")
    except Exception as e:
        pytest.fail(f"Download failed for {model_name}: {e}")