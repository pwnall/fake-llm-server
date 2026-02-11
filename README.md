# Fake LLM Server

This is a lightweight implementation of LLM APIs that can be used as a
[fake](https://abseil.io/resources/swe-book/html/ch13.html) in tests.

The "lightweight" aspect is accomplished by using a small language model (SLM)
instead of a large one (LLM).

## Usage

Create a fake API server.

```python
fake_llm_server = FakeLLMServer(model_name="gemma-3-270m")
```

`model_name` can be one of the aliases for the supported models below, or a
HuggingFace repository ID. Aliases never contain `/`, whereas repository IDs
always contain a `/`.

### Supported models

* `qwen-2.5-coder-3b`
* `qwen-2.5-coder-1.5b` -recommended for code generation
* `llama-3.2-3b-instruct`
* `smollm3` - SmolLM3 3B - recommended for reasoning
* `gemma-3-1b` - Gemma 3 1B Instruction Tuned
* `gemma-3-270m` - Gemma 3 270M Instruction Tuned - recommended for small tests

Obtain client arguments for OpenAI API implementation.

```python
fake_llm_server.openai_client_args()
```

## Development

The following documents are useful for development.

*  [project specification](docs/spec.md)
