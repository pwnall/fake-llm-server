# Fake LLM server code map

The library's code lives in `src/fake_llm_server`, which is a Python module with
a `py.typed` marker.

The library's public API is defined by `main.py`, which imports symbols from
the other non-public modules.

The library's tests live in `tests/`. Test file names follow the convention
`test_{TESTED_MODULE}_{FEATURE}.py`.

## Model management

The model management code lives in `_models`.

`ModelSpec` represents a model that can be downloaded from Hugging Face.

* The `from_model_name` factory method accepts both supported model names and
  Hugging Face repository names.
* The `download` method integrates with `huggingface_hub` and creates a
  `DownloadedModel` that represents the downloaded model.

`DownloadedModel` represents a model downloaded to a local file.

* The `load` method integrates with `llama-cpp-python` and creates a
  `Llama` instance pointing to the local model.

Both `ModelSpec` and `DownloadedModel` retain the original name passed to
`from_model_name`. This name is used in the serving infrastructure.

`SUPPORTED_MODEL_ALIASES` is a dictionary that maps strings to `ModelSpec`
instances.

## API server

The API server code lives in `_api_server`.

`ServingConfiguration` is a type alias for a frozen dictionary mapping strings
to `Llama` (from `llama-cpp-python`) instances.

`parse_server_args` is a function that takes in the same arguments as
`FakeLLMServer` and returns a `ServingConfiguration`. The `FakeLLMServer`
arguments are documented in the README.

`create_server_app` is a factory function that receives a `ServingConfiguration`
and produces a `FastAPI` application.

## Serving

The serving code lives in `_serving`.

The main module export is the `FakeLLMServer` class, whose usage is documented
in the README. Instances of this class run on the main thread.

`ServingThread` encapsulates the data and logic for serving the FastAPI
application via uvicorn on a separate thread. All the logic (including the
constructor) runs exclusively on the serving thread. All the data is accessed
exclusively on the serving thread.

`StartInfo` stores information created by the uvicorn server start. The data is
accessed on both the main thread and the serving thread, so data accesses are
protected by a `threading.Lock` mutex. The data includes
* the port that the uvicorn listens to
* a `threading.Event` event that is set after the uvicorn server starts
* a reference to the uvicorn server instance

The `FakeLLMServer` constructor calls `parse_server_args`, creates the
`StartInfo` instance, and spawns the uvicorn serving thread. The serving
thread's target is a lambda that calls the `ServingThread` constructor,
passing the `ServingConfiguration` produced by `parse_server_args` and the
`StartInfo`.

