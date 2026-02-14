"""Serving infrastructure for the Fake LLM Server."""

import socket
import threading
import time
from typing import Any

import uvicorn

from ._api_server import ServingConfiguration, create_server_app, parse_server_args


class StartInfo:
    """Stores information created by the uvicorn server start."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._port: int | None = None
        self._server: uvicorn.Server | None = None
        self.started_event = threading.Event()

    @property
    def port(self) -> int:
        """Returns the port that the uvicorn listens to."""
        with self._lock:
            if self._port is None:
                msg = "Port not set"
                raise RuntimeError(msg)
            return self._port

    @port.setter
    def port(self, value: int) -> None:
        """Sets the port that the uvicorn listens to."""
        with self._lock:
            self._port = value

    @property
    def server(self) -> uvicorn.Server | None:
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

    def __init__(self, config: uvicorn.Config, start_info: StartInfo) -> None:
        super().__init__(config)
        self._start_info = start_info

    async def startup(self, sockets: list[Any] | None = None) -> None:
        """Called when the server is starting up."""
        await super().startup(sockets)
        self._start_info.started_event.set()


class ServingThread:
    """Encapsulates the data and logic for serving the FastAPI application."""

    def __init__(self, config: ServingConfiguration, start_info: StartInfo) -> None:
        """Initializes the ServingThread and runs the server.

        Args:
            config: The server configuration.
            start_info: The start information object.
        """
        self.config = config
        self.start_info = start_info

        # Create the FastAPI app
        self.app = create_server_app(config)

        # Configure Uvicorn
        self.uvicorn_config = uvicorn.Config(
            app=self.app,
            host="127.0.0.1",
            port=start_info.port,
            log_level="info",
        )
        self.server = _NotifyServer(self.uvicorn_config, start_info)
        start_info.server = self.server

        # Run the server (this blocks)
        self.server.run()


class FakeLLMServer:
    """A lightweight, fake implementation of an LLM API server for testing."""

    def __init__(
        self,
        model_names: tuple[str, ...] = ("gemma-3-270m",),
        aliases: dict[str, str] = {},  # noqa: B006
    ) -> None:
        """Initializes the FakeLLMServer.

        Args:
            model_names: Tuple of model names or repo IDs to use.
            aliases: Dictionary mapping aliases to model names/repo IDs.
        """
        llms = parse_server_args(model_names, aliases)

        self.start_info = StartInfo()
        self.start_info.port = self._get_free_port()

        self.thread: threading.Thread | None = threading.Thread(
            target=lambda: ServingThread(llms, self.start_info),
            daemon=True,
        )
        self.thread.start()

        self._wait_for_start()

    def _get_free_port(self) -> int:
        """Finds a free port to bind the server to.

        Returns:
            An available port number.
        """
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return int(s.getsockname()[1])

    def _wait_for_start(self, timeout: int = 300) -> None:  # 5 minutes max
        """Waits for the server to become ready.

        Args:
            timeout: Maximum time to wait in seconds.

        Raises:
            RuntimeError: If the server thread dies or startup times out.
        """
        start_time = time.time()
        while not self.start_info.started_event.is_set():
            if self.thread is None or not self.thread.is_alive():
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
            "base_url": f"http://127.0.0.1:{self.start_info.port}/v1",
            "api_key": "fake-key",
        }

    def shutdown(self) -> None:
        """Shuts down the server and joins the background thread."""
        if hasattr(self, "start_info") and self.start_info and self.start_info.server:
            self.start_info.server.should_exit = True

        if hasattr(self, "thread") and self.thread and self.thread.is_alive():
            self.thread.join(timeout=5)
        self.thread = None

    def __del__(self) -> None:
        """Ensures the server is shut down when the object is destroyed."""
        self.shutdown()
