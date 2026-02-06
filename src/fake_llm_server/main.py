"""Main entry point for the Fake LLM Server."""

import socket
import threading
import time
from dataclasses import dataclass
from typing import Any

import httpx
import uvicorn

from .models import LocalModelSpec, RemoteModelSpec
from .server import app


@dataclass
class ServerConfig:
    """Configuration data passed from the main thread to the serving thread."""

    model_spec: LocalModelSpec
    port: int


class ServingThread:
    """Encapsulates the data accessed on the uvicorn serving thread."""

    def __init__(self, config: ServerConfig) -> None:
        """Initializes the ServingThread.

        Args:
            config: The server configuration.
        """
        self.config = config

        # Configure app state
        app.state.model_path = config.model_spec.model_path
        app.state.model_id = config.model_spec.model_name
        # Store reference to self for shutdown from main thread
        app.state.serving_thread = self

        # Configure Uvicorn
        self.uvicorn_config = uvicorn.Config(
            app=app,
            host="127.0.0.1",
            port=config.port,
            log_level="info",
        )
        self.server = uvicorn.Server(self.uvicorn_config)

    def run_server(self) -> None:
        """Runs the uvicorn server."""
        self.server.run()

    def stop(self) -> None:
        """Signals the server to stop."""

        # Setting a boolean is atomic in CPython.
        self.server.should_exit = True

    @staticmethod
    def run(config: ServerConfig) -> None:
        """Entry point for the serving thread.

        Args:
            config: The server configuration.
        """
        serving_thread = ServingThread(config)
        serving_thread.run_server()


class FakeLLMServer:
    """A lightweight, fake implementation of an LLM API server for testing."""

    def __init__(self, model_name: str = "gemma-3-270m") -> None:
        """Initializes the FakeLLMServer.

        Args:
            model_name: The name or repo ID of the model to use.
        """
        port = self._get_free_port()
        model_spec = RemoteModelSpec.from_name(model_name).download()
        self.config = ServerConfig(
            model_spec=model_spec,
            port=port,
        )
        self.thread: threading.Thread | None = None
        self._start_server()

    def _get_free_port(self) -> int:
        """Finds a free port to bind the server to.

        Returns:
            An available port number.
        """
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return int(s.getsockname()[1])

    def _start_server(self) -> None:
        """Starts the FastAPI server in a separate thread."""
        self.thread = threading.Thread(
            target=lambda: ServingThread.run(self.config), daemon=True,
        )
        self.thread.start()

        self._wait_for_server()

    def _wait_for_server(self, timeout: int = 300) -> None:  # 5 minutes max
        """Waits for the server to become ready by polling its health endpoint.

        Args:
            timeout: Maximum time to wait in seconds.

        Raises:
            RuntimeError: If the server thread dies or startup times out.
        """
        start_time = time.time()
        url = f"http://127.0.0.1:{self.config.port}/v1/models"

        while time.time() - start_time < timeout:
            if self.thread is None or not self.thread.is_alive():
                msg = "Server thread died unexpectedly"
                raise RuntimeError(msg)

            try:
                response = httpx.get(url, timeout=1)
                if response.status_code == httpx.codes.OK:
                    return
            except httpx.RequestError:
                pass

            time.sleep(1)

        self.shutdown()
        msg = "Server timed out waiting for startup"
        raise RuntimeError(msg)

    def openai_client_args(self) -> dict[str, Any]:
        """Returns arguments suitable for initializing an OpenAI API client.

        Returns:
            A dictionary containing base_url and api_key.
        """
        return {
            "base_url": f"http://127.0.0.1:{self.config.port}/v1",
            "api_key": "fake-key",
        }

    def shutdown(self) -> None:
        """Shuts down the server and joins the background thread."""
        # Retrieve the ServingThread instance from app state
        serving_thread = getattr(app.state, "serving_thread", None)
        if serving_thread:
            serving_thread.stop()
            # Clear the reference
            del app.state.serving_thread

        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5)
        self.thread = None

    def __del__(self) -> None:
        """Ensures the server is shut down when the object is destroyed."""
        self.shutdown()
