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

The library exposes the high-level `open_fake_llm_server` function, which
returns a `FakeLLMServer` context manager. The factory manager and context
manager are responsible for starting and running an API server, and for shutting
it down and releasing the resources. Each test is expected to call the factory
method and create a new `FakeLLMServer`.

The API server is implemented using [FastAPI](https://fastapi.tiangolo.com/).

Each `FakeLLMServer` instance created by `open_fake_llm_server` runs a FastAPI
application on top of [uvicorn](https://www.uvicorn.org/) in a separate thread.

### Inference

The library uses [llama.cpp](https://github.com/ggml-org/llama.cpp) via
[llama-cpp-python](https://github.com/abetlen/llama-cpp-python) for inference.
The library is focused on models in
[the GGUF format](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md).

The library explicitly relies on llama.cpp's mmap (memory mapping) feature to
minimize I/O while `open_fake_llm_server` is called as much as one time per
test.

The library uses [psutil](https://psutil.readthedocs.io/) to compute the number
of physical cores on the CPU, and sets the llama.cpp number of threads to
match. If psutil fails, `os.process_cpu_count()` is used as fallback.

### Model management

The library downloads and caches models from Hugging Face via
[huggingface_hub](https://github.com/huggingface/huggingface_hub). The library
prefers the Q4_K_M quantization format, which is a good trade-off between CPU
inference time and quality.

### LLM API implementation

The API server implementation is based on the
[OpenAPI](https://spec.openapis.org/oas/latest.html)
[specification for the OpenAI API](https://app.stainless.com/api/spec/documented/openai/openapi.documented.yml).

## Testing strategy

### Model management

The model download functionality is tested by unit tests.

Each supported model is covered by a model download test. The download test
asserts tha the downloaded `.gguf` file dimension exceeds a lower bound derived
from the model size. For example, the Gemma 3 270M model file should have at
least 135 MB, in order to store the quantized weihgts.

### LLM API implementation

The LLM API impelementation is covered by integration tests that use the
[openai-python](https://github.com/openai/openai-python) client library and the
top-level `open_fake_llm_server` function.

Each supported LLM API method is covered by a test that expresses typical usage,
and one test for each interesting parameter. For example, the OpenAI chat
completion API method has one test that covers the temperature parameter, and
all other tests use the default temperature zero.

Each integration test instantiates an `OpenAI` client with arguments obtained
from `FakeLLMServer` (yielded by `open_fake_llm_server`). Each integration test
uses the smallest model whose capabilities meet the test's needs.

These integration tests provide coverage for the serving infrastructure in
`open_fake_llm_server` and `FakeLLMServer`.

### Inference

Model inference is tested using integration tests, using the same method as API
serving.

Each supported model is covered by an inference test. Inference tests use
the chat completion API, and issue a query whose difficulty level matches the
model capabilities.

### Bug fixes

Whenver possible, bug fixes will be accompanied by tests that ensure the bugs
don't return.
