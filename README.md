# Fake LLM Server

This is a lightweight implementation of LLM APIs that can be used as a
[fake](https://abseil.io/resources/swe-book/html/ch13.html) in tests.

The "lightweight" aspect is accomplished by using a small language model (SLM)
instead of a large one (LLM).

## Usage

Theoretical usage in a non-testing context:

```python
from fake_llm_server import open_fake_llm_server
from openai import OpenAI

with open_fake_llm_server(
        model_names=("gemma-3-270m", "qwen-2.5-coder-3b"),
        aliases={"GPT-5.2-Codex": "qwen-2.5-coder-3b"}) as fake_llm_server:

    llm_client = OpenAI(**fake_llm_server.openai_client_args())

    # Use `llm_client`.
```

Using in tests:

```python
import pytest
from fake_llm_server import open_fake_llm_server
from openai import OpenAI

@pytest.fixture
def llm_client():
    """Provides an OpenAI client connected to a Fake LLM server."""

    with open_fake_llm_server(
            model_names=("gemma-3-270m", "qwen-2.5-coder-3b"),
            aliases={"GPT-5.2-Codex": "qwen-2.5-coder-3b"}) as fake_llm_server:

        client = OpenAI(**fake_llm_server.openai_client_args())
        yield client

def test_logic(llm_client):
    # Use `llm_client`.
```

Each `model_names` element must be one of the short names for the supported
models below, or a HuggingFace repository ID. Short names never contain `/`,
whereas repository IDs always contain a `/`.

Each `aliases` dictionary value must be one of the entries in `model_names`.

Model aliases must point to canonical model names -- transitive aliases are not
supported. Aliases are recognized by all APIs.

### Supported models

* `qwen-2.5-coder-3b`
* `qwen-2.5-coder-1.5b` - recommended for code generation
* `llama-3.2-3b-instruct`
* `smollm3` - SmolLM3 3B - recommended for reasoning
* `gemma-3-1b` - Gemma 3 1B Instruction Tuned
* `gemma-3-270m` - Gemma 3 270M Instruction Tuned - recommended for small tests

## Supported LLM APIs

The API server currently implements the following subset of the
[OpenAI Rest API](https://platform.openai.com/docs/api-reference/).

* [Models](https://developers.openai.com/api/reference/resources/models/)
    * [List models](https://platform.openai.com/docs/api-reference/models/list)
* [Chat completions](https://platform.openai.com/docs/api-reference/chat/)
    * [Create chat completion](https://platform.openai.com/docs/api-reference/completions/create)
* [Responses](https://platform.openai.com/docs/api-reference/responses)
    * [Create a model response](https://platform.openai.com/docs/api-reference/responses/create)

The API server uses the testing-friendly defaults below:

* Temperature is set to 0 (zero) by default. This makes tests easier to debug.
* Requests are stateless by default. This is similar to being subject to
  [Zero Data Retention](https://platform.openai.com/docs/guides/your-data#zero-data-retention).

## Development

The following documents are useful for development.

* [project specification](docs/spec.md)
* [code map](docs/code_map.md)
