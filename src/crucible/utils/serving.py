"""vLLM server management utilities.

Handles starting, stopping, and health-checking vLLM servers for
all models in the pipeline.
"""

import logging
import subprocess
import time
from dataclasses import dataclass

import requests
import yaml

logger = logging.getLogger(__name__)


@dataclass
class VLLMServer:
    """Manages a single vLLM server process."""
    name: str
    model: str
    port: int
    tensor_parallel_size: int = 1
    quantization: str | None = None
    max_model_len: int = 8192
    gpu_memory_utilization: float = 0.85
    dtype: str = "float16"
    process: subprocess.Popen | None = None

    def start(self):
        """Start the vLLM server as a background process."""
        cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.model,
            "--port", str(self.port),
            "--tensor-parallel-size", str(self.tensor_parallel_size),
            "--max-model-len", str(self.max_model_len),
            "--gpu-memory-utilization", str(self.gpu_memory_utilization),
            "--dtype", self.dtype,
            "--trust-remote-code",
        ]

        if self.quantization and self.quantization != "none":
            cmd.extend(["--quantization", self.quantization])

        logger.info(f"Starting vLLM server '{self.name}' on port {self.port}")
        logger.debug(f"Command: {' '.join(cmd)}")

        self.process = subprocess.Popen(
            cmd,
            stdout=open(f"logs/vllm_{self.name}.log", "w"),
            stderr=subprocess.STDOUT,
        )

        # Wait for server to be ready
        self._wait_for_ready()

    def _wait_for_ready(self, timeout: int = 600, interval: int = 10):
        """Poll the health endpoint until the server is ready."""
        url = f"http://localhost:{self.port}/health"
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                resp = requests.get(url, timeout=5)
                if resp.status_code == 200:
                    logger.info(f"Server '{self.name}' is ready on port {self.port}")
                    return
            except requests.ConnectionError:
                pass

            if self.process and self.process.poll() is not None:
                raise RuntimeError(
                    f"Server '{self.name}' exited with code {self.process.returncode}. "
                    f"Check logs/vllm_{self.name}.log"
                )

            logger.debug(f"Waiting for '{self.name}' to be ready...")
            time.sleep(interval)

        raise TimeoutError(
            f"Server '{self.name}' did not become ready within {timeout}s"
        )

    def stop(self):
        """Stop the vLLM server."""
        if self.process:
            logger.info(f"Stopping server '{self.name}'")
            self.process.terminate()
            self.process.wait(timeout=30)
            self.process = None

    def is_running(self) -> bool:
        """Check if the server process is still running."""
        if self.process is None:
            return False
        return self.process.poll() is None


def start_all_servers(config_path: str = "configs/models.yaml") -> dict[str, VLLMServer]:
    """Start vLLM servers for all models defined in the config.

    Args:
        config_path: Path to the models config file.

    Returns:
        Dict mapping server name to VLLMServer instance.
    """
    import os
    os.makedirs("logs", exist_ok=True)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    servers = {}

    # Start attacker
    atk = config["attacker"]
    servers["attacker"] = VLLMServer(
        name="attacker",
        model=atk["hf_repo"],
        port=atk["vllm_port"],
        tensor_parallel_size=atk.get("tensor_parallel_size", 1),
        quantization=atk.get("quantization"),
        max_model_len=atk.get("max_model_len", 8192),
        gpu_memory_utilization=atk.get("gpu_memory_utilization", 0.85),
        dtype=atk.get("dtype", "float16"),
    )

    # Start target
    tgt = config["target"]
    servers["target"] = VLLMServer(
        name="target",
        model=tgt["hf_repo"],
        port=tgt["vllm_port"],
        tensor_parallel_size=tgt.get("tensor_parallel_size", 2),
        quantization=tgt.get("quantization"),
        max_model_len=tgt.get("max_model_len", 8192),
        gpu_memory_utilization=tgt.get("gpu_memory_utilization", 0.85),
        dtype=tgt.get("dtype", "float16"),
    )

    # Start evaluators
    for eval_name, eval_cfg in config.get("evaluators", {}).items():
        servers[eval_name] = VLLMServer(
            name=eval_name,
            model=eval_cfg["hf_repo"],
            port=eval_cfg["vllm_port"],
            tensor_parallel_size=eval_cfg.get("tensor_parallel_size", 1),
            quantization=eval_cfg.get("quantization"),
            max_model_len=eval_cfg.get("max_model_len", 4096),
            gpu_memory_utilization=eval_cfg.get("gpu_memory_utilization", 0.85),
            dtype=eval_cfg.get("dtype", "float16"),
        )

    # Start all servers
    for server in servers.values():
        server.start()

    return servers


def stop_all_servers(servers: dict[str, VLLMServer]):
    """Stop all running vLLM servers."""
    for server in servers.values():
        server.stop()
    logger.info("All servers stopped.")
