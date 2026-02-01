# Fake LLM Server

This is a lightweight implementation of LLM APIs that can be used as a
[fake](https://abseil.io/resources/swe-book/html/ch13.html) in tests.

The "lightweight" aspect is accomplished by using a small language model (SLM)
instead of a large one (LLM).

## Requirements

The system aims to fit in
[GitHub-hosted runners for private repositories](https://docs.github.com/en/actions/reference/runners/github-hosted-runners#standard-github-hosted-runners-for--private-repositories)
and scale to take advantage of
[GitHub-hosted runners for public repositories](https://docs.github.com/en/actions/reference/runners/github-hosted-runners#standard-github-hosted-runners-for-public-repositories).

This principle currently translates to the limitations below.

1. Inference works without a GPU.
2. The model fits in 4 GB of RAM.
3. All supporting files (files, Python libraries) fit in the 10 GB cache limit.
4. CPU-based inference produces a reasonable latency for tests.

## Architecture

The library exposes the high-level `FakeLLMServer` class, which is responsible
for starting and running an API server. Each test is expected to create a new
instance of this class.

The API server is implemented using [FastAPI](https://fastapi.tiangolo.com/).

Each `FakeLLMServer` instance launches the FastAPI server on top of
[uvicorn](https://www.uvicorn.org/) in a separate thread.

The library uses [llama.cpp](https://github.com/ggml-org/llama.cpp) via
[llama-cpp-python](https://github.com/abetlen/llama-cpp-python) for inference.
The library is focused on models in
[the GGUF format](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)

The library downloads and caches models from Hugging Face via
[huggingface_hub](https://github.com/huggingface/huggingface_hub). The library
prefers the Q4_K_M quantization format, which is a good trade-off between CPU
inference time and quality.

The API server facilitates uses the testing-friendly defaults below:

* Temperature is set to 0 (zero) by default. This makes tests easier to debug.
* Requests are stateless by default. This is similar to being subject to
  [Zero Data Retention](https://platform.openai.com/docs/guides/your-data#zero-data-retention).

The API server currently implements the following subset of the
[OpenAI Rest API](https://platform.openai.com/docs/api-reference/).
* [Chat completions](https://platform.openai.com/docs/api-reference/chat/)
    * [Create chat completion](https://platform.openai.com/docs/api-reference/completions/create)
* [Responses](https://platform.openai.com/docs/api-reference/responses)
    * [Create a model response](https://platform.openai.com/docs/api-reference/responses/create)

The API server implementation is based on the
[OpenAPI](https://spec.openapis.org/oas/latest.html)
[specification for the OpenAI API](https://app.stainless.com/api/spec/documented/openai/openapi.documented.yml).

## Supported models

* Qwen 2.5 Coder 3B
* Qwen 2.5 Coder 1.5B (recommended for code generation)
* Llama 3.2 Instruct 3B
* SmolLM3 3B (recommended for reasoning)
* Gemma 3 1B Instruction Tuned
* Gemma 3 270M Instruction Tuned (recommended for smaller tests)

## Testing

The model download functionality is tested by unit tests covering each model.
The tests assert that the downloaded `.gguf` file dimension exceeds a lower
bound derived from the model size. For example, the Gemma 3 270M model file
should have at least 135 MB, in order to store the quantized weihgts.

The fake server is tested using the
[openai-python](https://github.com/openai/openai-python) client library.
Tests instantiate an `OpenAI` client with custom `base_url` and `api_key`
options.

Server tests verify model support by issuing one query against each supported
model. Separate server tests cover each feature, using the smallest model that
works.

Tests use the default temperature 0 (zero), with the exception of a test
covering support for different temperatures. The temperature support test
sets a higher temperature, generates a few completions, and asserts that the
completions are not identical.

## Usage

Create a fake API server.

```python
fake_llm_server = FakeLLMServer(model_name="gemma-3-270m")
```

Obtain client arguments for OpenAI API implementation.

```python
fake_llm_server.openai_client_args()
```
