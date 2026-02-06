"""Models and specifications for LLM models."""

from dataclasses import dataclass

from huggingface_hub import hf_hub_download, list_repo_files


@dataclass(frozen=True)
class LocalModelSpec:
    """Identifies a model on the local filesystem."""

    model_name: str
    model_path: str


@dataclass(frozen=True)
class RemoteModelSpec:
    """Identifies a model that can be downloaded from Hugging Face Hub."""

    model_name: str
    repo_id: str
    filename: str

    @classmethod
    def from_repo_id(cls, repo_id: str) -> "RemoteModelSpec":
        """Creates a RemoteModelSpec from a Hugging Face repo ID.

        Args:
            repo_id: The Hugging Face repo ID (e.g., "username/repo_name").

        Returns:
            A RemoteModelSpec for the model.

        Raises:
            ValueError: If the repo does not exist or has no suitable .gguf files.
        """
        try:
            files = list_repo_files(repo_id=repo_id)
        except Exception as e:
            msg = f"Could not list files for repo {repo_id}: {e}"
            raise ValueError(
                msg,
            ) from e

        gguf_files = [f for f in files if f.endswith(".gguf")]
        if not gguf_files:
            msg = f"No .gguf files found in {repo_id}"
            raise ValueError(msg)

        # Prefer q4_k_m
        preferred = [f for f in gguf_files if "q4_k_m" in f.lower()]
        filename = preferred[0] if preferred else gguf_files[0]

        return cls(model_name=repo_id, repo_id=repo_id, filename=filename)

    @classmethod
    def from_name(cls, model_name: str) -> "RemoteModelSpec":
        """Parses a model identifier, which can be a name or repo ID.

        Args:
            model_name: The model's short name or Hugging Face repo ID.

        Returns:
            A RemoteModelSpec for the model.

        Raises:
            ValueError: If the model is not supported or cannot be found.
        """
        if model_name in SUPPORTED_MODELS:
            return SUPPORTED_MODELS[model_name]

        if "/" in model_name:
            return cls.from_repo_id(model_name)

        msg = (
            f"Model '{model_name}' not supported. "
            f"Available models: {list(SUPPORTED_MODELS.keys())}"
        )
        raise ValueError(msg)

    def download(self) -> LocalModelSpec:
        """Downloads the model from Hugging Face and returns the local model spec.

        Returns:
            A LocalModelSpec containing the path to the downloaded model file.
        """
        try:
            model_path = hf_hub_download(
                repo_id=self.repo_id,
                filename=self.filename,
                # We rely on the default cache dir or environment variables.
            )
            return LocalModelSpec(model_name=self.model_name, model_path=model_path)
        except Exception as e:
            msg = f"Failed to download model from {self.repo_id}: {e}"
            raise RuntimeError(
                msg,
            ) from e


# Define supported models
SUPPORTED_MODELS: dict[str, RemoteModelSpec] = {
    "qwen-2.5-coder-3b": RemoteModelSpec(
        model_name="qwen-2.5-coder-3b",
        repo_id="Qwen/Qwen2.5-Coder-3B-Instruct-GGUF",
        filename="qwen2.5-coder-3b-instruct-q4_k_m.gguf",
    ),
    "qwen-2.5-coder-1.5b": RemoteModelSpec(
        model_name="qwen-2.5-coder-1.5b",
        repo_id="Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF",
        filename="qwen2.5-coder-1.5b-instruct-q4_k_m.gguf",
    ),
    "llama-3.2-3b-instruct": RemoteModelSpec(
        model_name="llama-3.2-3b-instruct",
        repo_id="bartowski/Llama-3.2-3B-Instruct-GGUF",
        filename="Llama-3.2-3B-Instruct-Q4_K_M.gguf",
    ),
    "smollm3": RemoteModelSpec(
        model_name="smollm3",
        repo_id="ggml-org/SmolLM3-3B-GGUF",
        filename="SmolLM3-Q4_K_M.gguf",
    ),
    "gemma-3-1b": RemoteModelSpec(
        model_name="gemma-3-1b",
        repo_id="unsloth/gemma-3-1b-it-GGUF",
        filename="gemma-3-1b-it-Q4_K_M.gguf",
    ),
    "gemma-3-270m": RemoteModelSpec(
        model_name="gemma-3-270m",
        repo_id="unsloth/gemma-3-270m-it-GGUF",
        filename="gemma-3-270m-it-Q4_K_M.gguf",
    ),
    # Aliases or fallbacks can be added here
}
