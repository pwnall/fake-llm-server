"""Serving infrastructure for the Fake LLM Server."""

import gc
import socket
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import uvicorn

from ._api_server import ServingConfiguration, create_server_app, parse_server_args


class StartInformation:
    """Stores information created by the uvicorn server start."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._port: int = 0
        self._server: uvicorn.Server = uvicorn.Server(config=uvicorn.Config(app=""))
        self.started_event = threading.Event()

    @property
    def port(self) -> int:
        """Returns the port that the uvicorn listens to."""
        with self._lock:
            if self._port == 0:
                msg = "Port not set"
                raise RuntimeError(msg)
            return self._port

    @port.setter
    def port(self, value: int) -> None:
        """Sets the port that the uvicorn listens to."""
        with self._lock:
            self._port = value

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
        self, uvicorn_configuration: uvicorn.Config, start_information: StartInformation
    ) -> None:
        super().__init__(config=uvicorn_configuration)
        self._start_information = start_information

    async def startup(self, sockets: list[Any] | None = None) -> None:
        """Called when the server is starting up."""
        await super().startup(sockets=sockets)
        self._start_information.started_event.set()


class ServingThread:
    """Encapsulates the data and logic for serving the FastAPI application."""

    def __init__(
        self,
        serving_configuration: ServingConfiguration,
        start_information: StartInformation,
        server_socket: socket.socket,
    ) -> None:
        """Initializes the ServingThread and runs the server.

        Args:
            serving_configuration: The server configuration.
            start_information: The start information object.
            server_socket: The socket to run the server on.
        """
        self.serving_configuration = serving_configuration
        self.start_information = start_information

        # Create the FastAPI app
        self.app = create_server_app(llms=serving_configuration)

        # Configure Uvicorn
        self.uvicorn_configuration = uvicorn.Config(
            app=self.app,
            log_level="info",
        )
        self.server = _NotifyServer(
            uvicorn_configuration=self.uvicorn_configuration,
            start_information=start_information,
        )
        start_information.server = self.server

        # Run the server (this blocks)
        self.server.run(sockets=[server_socket])


class FakeLLMServer:
    """A lightweight, fake implementation of an LLM API server for testing."""

    def __init__(self, serving_configuration: ServingConfiguration) -> None:
        """Initializes the FakeLLMServer.

        Args:
            serving_configuration: The server configuration.
        """
        self.serving_configuration = serving_configuration
        self.start_information = StartInformation()
        self.thread: threading.Thread = threading.Thread()
        self._server_socket: socket.socket = socket.socket(
            socket.AF_INET, socket.SOCK_STREAM
        )

    def start(self) -> None:
        """Starts the server."""
        self._server_socket.bind(("127.0.0.1", 0))
        self.start_information.port = self._server_socket.getsockname()[1]

        self.thread = threading.Thread(
            target=lambda: ServingThread(
                serving_configuration=self.serving_configuration,
                start_information=self.start_information,
                server_socket=self._server_socket,
            ),
            daemon=True,
        )
        self.thread.start()

        self._wait_for_start()

    def _wait_for_start(self, timeout: int = 300) -> None:  # 5 minutes max
        """Waits for the server to become ready.

        Args:
            timeout: Maximum time to wait in seconds.

        Raises:
            RuntimeError: If the server thread dies or startup times out.
        """
        start_time = time.time()
        while not self.start_information.started_event.is_set():
            if not self.thread.is_alive():
                msg = "Server thread died unexpectedly"
                raise RuntimeError(msg)

            if time.time() - start_time > timeout:
                self.shutdown()
                msg = "Server timed out waiting for startup"
                raise RuntimeError(msg)

            time.sleep(0.1)

    def openai_client_args(self) -> dict[str, Any]:
        """Returns arguments suitable for initializing an OpenAI API client.

        Returns:
            A dictionary containing base_url and api_key.
        """
        return {
            "base_url": f"http://127.0.0.1:{self.start_information.port}/v1",
            "api_key": "fake-key",
        }

    def shutdown(self) -> None:
        """Shuts down the server and joins the background thread."""
        if self.start_information.server:
            self.start_information.server.should_exit = True

        if self.thread.is_alive():
            self.thread.join(timeout=5)


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
    server = FakeLLMServer(serving_configuration=llms)
    try:
        server.start()
        yield server
    finally:
        server.shutdown()
        gc.collect()
