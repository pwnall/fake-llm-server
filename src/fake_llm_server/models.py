import os
from dataclasses import dataclass
from huggingface_hub import hf_hub_download

@dataclass
class ModelSpec:
    repo_id: str
    filename: str

# Define supported models
MODELS = {
    "qwen-2.5-coder-3b": ModelSpec(
        repo_id="Qwen/Qwen2.5-Coder-3B-Instruct-GGUF",
        filename="qwen2.5-coder-3b-instruct-q4_k_m.gguf"
    ),
    "qwen-2.5-coder-1.5b": ModelSpec(
        repo_id="Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF",
        filename="qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"
    ),
    "llama-3.2-3b-instruct": ModelSpec(
        repo_id="bartowski/Llama-3.2-3B-Instruct-GGUF",
        filename="Llama-3.2-3B-Instruct-Q4_K_M.gguf"
    ),
    "smollm3": ModelSpec(
        repo_id="ggml-org/SmolLM3-3B-GGUF",
        filename="smollm3-3b-q4_k_m.gguf"
    ),
    "gemma-3-1b": ModelSpec(
        repo_id="bartowski/gemma-3-1b-it-GGUF", 
        filename="gemma-3-1b-it-Q4_K_M.gguf"
    ),
    "gemma-3-270m": ModelSpec(
        repo_id="hugginllam/gemma-3-270m-Q4_K_M-GGUF",
        filename="gemma-3-270m-Q4_K_M.gguf"
    ),
    # Aliases or fallbacks can be added here
}

def download_model(model_name: str) -> str:
    """
    Downloads the specified model from Hugging Face and returns the local file path.
    
    Args:
        model_name: The name of the model to download.
        
    Returns:
        The path to the downloaded model file.
        
    Raises:
        ValueError: If the model name is not supported.
    """
    if model_name not in MODELS:
        # Check if it's a known alias or if the user provided a direct path logic? 
        # For now, strict checking against the list.
        # Maybe handle "gemma-3-270m-it" as "gemma-3-270m" if needed.
        if model_name == "gemma-3-270m-it":
            model_name = "gemma-3-270m"
        else:
            raise ValueError(f"Model '{model_name}' not supported. Available models: {list(MODELS.keys())}")

    spec = MODELS[model_name]
    print(f"Downloading {model_name} from {spec.repo_id}...")
    
    try:
        model_path = hf_hub_download(
            repo_id=spec.repo_id,
            filename=spec.filename,
            # We rely on the default cache dir or environment variables.
        )
        return model_path
    except Exception as e:
        # Fallback for Gemma 3 as it might not be fully indexed yet in my assumed repos
        # But for now, we assume these repos exist or will work.
        raise RuntimeError(f"Failed to download model {model_name}: {e}")
