"""Entry point for running the Fake LLM Server as a standalone process."""

import logging
import time

from fake_llm_server import FakeLLMServer

# Configure logging to look like print for this simple CLI
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    """Starts the FakeLLMServer and keeps it running until interrupted."""
    logger.info("Starting FakeLLMServer with default model (gemma-3-270m)...")
    # Using gemma-3-270m as it is the recommended small model
    server = FakeLLMServer(model_name="gemma-3-270m")
    logger.info("Server running.")
    logger.info("Client configuration: %s", server.openai_client_args())
    logger.info("Press Ctrl+C to stop.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("\nStopping server...")
        server.shutdown()
        logger.info("Server stopped.")


if __name__ == "__main__":
    main()
