# Fake LLM server specification

This document describes the repository's desired end state.

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

## System overview

### Serving

The library exposes the high-level `FakeLLMServer` class, which is responsible
for starting and running an API server. Each test is expected to create a new
instance of this class.

The API server is implemented using [FastAPI](https://fastapi.tiangolo.com/).

Each `FakeLLMServer` instance launches the FastAPI server on top of
[uvicorn](https://www.uvicorn.org/) in a separate thread.

### Inference

The library uses [llama.cpp](https://github.com/ggml-org/llama.cpp) via
[llama-cpp-python](https://github.com/abetlen/llama-cpp-python) for inference.
The library is focused on models in
[the GGUF format](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md).

The library explicitly relies on llama.cpp's mmap (memory mapping) feature to
minimize I/O while `FakeLLMServer` is instantiated as much as one time per test.

The library uses [psutil](https://psutil.readthedocs.io/) to compute the number
of physical cores on the CPU, and sets the llama.cpp number of threads to
match. If psutil fails, `os.process_cpu_count()` is used as fallback.

### Model management

The library downloads and caches models from Hugging Face via
[huggingface_hub](https://github.com/huggingface/huggingface_hub). The library
prefers the Q4_K_M quantization format, which is a good trade-off between CPU
inference time and quality.

### API

The API server currently implements the following subset of the
[OpenAI Rest API](https://platform.openai.com/docs/api-reference/).

* [Models](https://developers.openai.com/api/reference/resources/models/)
    * [List models](https://platform.openai.com/docs/api-reference/models/list)
* [Chat completions](https://platform.openai.com/docs/api-reference/chat/)
    * [Create chat completion](https://platform.openai.com/docs/api-reference/completions/create)
* [Responses](https://platform.openai.com/docs/api-reference/responses)
    * [Create a model response](https://platform.openai.com/docs/api-reference/responses/create)

The API server implementation is based on the
[OpenAPI](https://spec.openapis.org/oas/latest.html)
[specification for the OpenAI API](https://app.stainless.com/api/spec/documented/openai/openapi.documented.yml).

The API server uses the testing-friendly defaults below:

* Temperature is set to 0 (zero) by default. This makes tests easier to debug.
* Requests are stateless by default. This is similar to being subject to
  [Zero Data Retention](https://platform.openai.com/docs/guides/your-data#zero-data-retention).

## Testing strategy

### Model management

The model download functionality is tested by unit tests.

Each supported model is covered by a model download test. The download test
asserts tha the downloaded `.gguf` file dimension exceeds a lower bound derived
from the model size. For example, the Gemma 3 270M model file should have at
least 135 MB, in order to store the quantized weihgts.

### API serving

The API serving functionality is covered by integration tests that use the
[openai-python](https://github.com/openai/openai-python) client library and
the top-level `FakeLLMServer` class.

Each supported API method is covered by a test that expresses typical usage,
and one test for each interesting parameter. For example, the chat completion
API has one test that covers the temperature parameter, and all other tests use
the default temperature zero.

Each integration test instantiates an `OpenAI` client with arguments obtained
from `FakeLLMServer`. Each integration test uses the smallest model whose
capabilities meet the test's needs.

### Inference

Model inference is tested using integration tests, using the same method as API
serving.

Each supported model is covered by an inference test. Inference tests use
the chat completion API, and issue a query whose difficulty level matches the
model capabilities.

### Bug fixes

Whenver possible, bug fixes will be accompanied by tests that ensure the bugs
don't return.
