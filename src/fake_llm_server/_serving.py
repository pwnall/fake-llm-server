"""Serving infrastructure for the Fake LLM Server."""

import gc
import socket
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import uvicorn

from ._api_server import ServingConfiguration, create_server_app, parse_server_args


class ServerState:
    """Stores information created by the uvicorn server start."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._server: uvicorn.Server = uvicorn.Server(config=uvicorn.Config(app=""))
        self.started_event = threading.Event()

    @property
    def server(self) -> uvicorn.Server:
        """Returns the uvicorn server instance."""
        with self._lock:
            return self._server

    @server.setter
    def server(self, value: uvicorn.Server) -> None:
        """Sets the uvicorn server instance."""
        with self._lock:
            self._server = value


class _NotifyServer(uvicorn.Server):
    """Uvicorn server that notifies when it has started."""

    def __init__(
        self, uvicorn_configuration: uvicorn.Config, server_state: ServerState
    ) -> None:
        super().__init__(config=uvicorn_configuration)
        self._server_state = server_state

    async def startup(self, sockets: list[Any] | None = None) -> None:
        """Called when the server is starting up."""
        await super().startup(sockets=sockets)
        self._server_state.started_event.set()


def serving_thread_main(
    serving_configuration: ServingConfiguration,
    server_state: ServerState,
    server_socket: socket.socket,
) -> None:
    """Encapsulates the data and logic for serving the FastAPI application.

    Args:
        serving_configuration: The server configuration.
        server_state: The start information object.
        server_socket: The socket to run the server on.
    """
    # Create the FastAPI app
    app = create_server_app(llms=serving_configuration)

    # Configure Uvicorn
    uvicorn_configuration = uvicorn.Config(
        app=app,
        log_level="info",
    )
    server = _NotifyServer(
        uvicorn_configuration=uvicorn_configuration,
        server_state=server_state,
    )
    server_state.server = server

    # Run the server (this blocks)
    server.run(sockets=[server_socket])


class FakeLLMServer:
    """A lightweight, fake implementation of an LLM API server for testing."""

    def __init__(self, serving_configuration: ServingConfiguration) -> None:
        """Initializes the FakeLLMServer and starts the server.

        Args:
            serving_configuration: The server configuration.
        """
        self._server_state = ServerState()
        self._server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_socket.bind(("127.0.0.1", 0))
        self._port: int = self._server_socket.getsockname()[1]

        self._thread = threading.Thread(
            target=lambda: serving_thread_main(
                serving_configuration=serving_configuration,
                server_state=self._server_state,
                server_socket=self._server_socket,
            ),
            daemon=True,
        )
        self._thread.start()

        self._wait_for_start()

    def _wait_for_start(self) -> None:
        """Waits for the server to become ready.

        Raises:
            RuntimeError: If the server thread dies.
        """
        while not self._server_state.started_event.wait(0.1):
            if not self._thread.is_alive():
                msg = "Server thread died unexpectedly"
                raise RuntimeError(msg)

    def openai_client_args(self) -> dict[str, Any]:
        """Returns arguments suitable for initializing an OpenAI API client.

        Returns:
            A dictionary containing base_url and api_key.
        """
        if self._port == 0:
            msg = "Port not set"
            raise RuntimeError(msg)
        return {
            "base_url": f"http://127.0.0.1:{self._port}/v1",
            "api_key": "fake-key",
        }

    def _shutdown(self) -> None:
        """Shuts down the server and joins the background thread."""
        if self._server_state.server:
            self._server_state.server.should_exit = True

        if self._thread.is_alive():
            self._thread.join(timeout=5)


@contextmanager
def open_fake_llm_server(
    model_names: tuple[str, ...] = ("gemma-3-270m",),
    aliases: dict[str, str] = {},  # noqa: B006
) -> Iterator[FakeLLMServer]:
    """Context manager to start and stop a FakeLLMServer.

    Args:
        model_names: Tuple of model names or repo IDs to use.
        aliases: Dictionary mapping aliases to model names/repo IDs.

    Yields:
        The running FakeLLMServer instance.
    """
    llms = parse_server_args(model_names=model_names, aliases=aliases)
    server = None
    try:
        server = FakeLLMServer(serving_configuration=llms)
        yield server
    finally:
        if server is not None:
            server._shutdown()  # noqa: SLF001
        gc.collect()
