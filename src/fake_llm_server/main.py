import os
import time
import socket
import subprocess
import sys
import httpx
from typing import Dict, Any
from .models import download_model

class FakeLLMServer:
    def __init__(self, model: str = "gemma-3-270m"):
        self.model = model
        self.port = self._get_free_port()
        print(f"Initializing FakeLLMServer with model {model} on port {self.port}...")
        self.model_path = download_model(model)
        self.process = None
        self._start_server()

    def _get_free_port(self) -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            return s.getsockname()[1]

    def _start_server(self):
        env = os.environ.copy()
        env["FAKE_LLM_MODEL_PATH"] = self.model_path
        env["FAKE_LLM_MODEL_ID"] = self.model
        env["PYTHONUNBUFFERED"] = "1"
        
        # Ensure the current directory (workspace root) is in PYTHONPATH 
        # so src.fake_llm_server is importable
        cwd = os.getcwd()
        python_path = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{cwd}:{python_path}"
        
        cmd = [
            sys.executable, "-m", "uvicorn", 
            "src.fake_llm_server.server:app", 
            "--host", "127.0.0.1", 
            "--port", str(self.port)
        ]
        
        self.process = subprocess.Popen(
            cmd,
            env=env,
            stdout=sys.stdout, # redirect to main process stdout for visibility
            stderr=sys.stderr  # redirect to main process stderr
        )
        
        self._wait_for_server()

    def _wait_for_server(self, timeout: int = 300): # 5 minutes max (downloading can take time but we download before start)
        # Loading the model into RAM can take a few seconds
        start_time = time.time()
        url = f"http://127.0.0.1:{self.port}/v1/models"
        
        print("Waiting for server to become ready...")
        while time.time() - start_time < timeout:
            if self.process.poll() is not None:
                raise RuntimeError(f"Server process exited unexpectedly with code {self.process.returncode}")
            
            try:
                response = httpx.get(url, timeout=1)
                if response.status_code == 200:
                    print("Server is ready.")
                    return
            except httpx.RequestError:
                pass
            
            time.sleep(1)
            
        self.shutdown()
        raise RuntimeError("Server timed out waiting for startup")

    def openai_client_args(self) -> Dict[str, Any]:
        return {
            "base_url": f"http://127.0.0.1:{self.port}/v1",
            "api_key": "fake-key"
        }

    def shutdown(self):
        if self.process:
            print("Shutting down FakeLLMServer...")
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None

    def __del__(self):
        self.shutdown()
