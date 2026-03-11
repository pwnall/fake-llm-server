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

The main module exports are the `open_fake_llm_server` context manager factory
function, whose usage is documented in the README, and the `FakeLLMServer`
class, which is the type of the context object returned by the context manager.

Some of the `FakeLLMServer` interface is specified by the README, such as the
`openai_client_args()` method. In addition, the class has a constructor that
starts the LLM API server on a new serving thread and blocks until the HTTP
server is listening, and a `_shutdown()` method that stops the API server and
blocks until the serving thread is joined.

Both `open_fake_llm_server` and `FakeLLMServer` reside on the main thread.

The `serving_thread_main` function runs on the serving thread, and has all the
logic for serving the FastAPI application via uvicorn.

`ServerState` manages the uvicorn server state data needed to coordinate the
main thread and the serving thread. Data accesses, which can happen on both
threads, are protected by a `threading.Lock` mutex. The server state data
includes
* a `threading.Event` event that is set after the uvicorn server starts
* a reference to the uvicorn server instance

The `open_fake_llm_server` function is a factory function for a context manager
object. The function uses the contextlib.contextmanager decorator to simplify
the implementation. The function calls `parse_server_args`, then it creates,
yields and shuts down a `FakeLLMServer`.

The `FakeLLMServer` constructor creates the `ServerState` instance, binds a
socket to a free port, and spawns the uvicorn serving thread. The serving
thread's target is a lambda that calls `serving_thread_main`, passing the
`ServingConfiguration`, the `ServerState`, and the bound socket.

