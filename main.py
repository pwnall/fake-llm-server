"""
Entry point for running the Fake LLM Server as a standalone process.
"""

import time

from fake_llm_server import FakeLLMServer


def main() -> None:
    """
    Starts the FakeLLMServer and keeps it running until interrupted.
    """
    print("Starting FakeLLMServer with default model (gemma-3-270m)...")
    # Using gemma-3-270m as it is the recommended small model
    server = FakeLLMServer(model_name="gemma-3-270m")
    print("Server running.")
    print(f"Client configuration: {server.openai_client_args()}")
    print("Press Ctrl+C to stop.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping server...")
        server.shutdown()
        print("Server stopped.")


if __name__ == "__main__":
    main()