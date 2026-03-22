import json
import time

import httpx

from local_inferbench.adapters.base import BaseAdapter, GenerationResult


class OllamaAdapter(BaseAdapter):
    """Adapter for the Ollama REST API.

    Talks to a local Ollama server (default http://localhost:11434) using
    the /api/generate endpoint with streaming to measure time-to-first-token.
    """

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        timeout: float = 300.0,
    ) -> None:
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._client = httpx.Client(base_url=self._base_url, timeout=self._timeout)

    def name(self) -> str:
        return "ollama"

    def model_id(self) -> str:
        return self._model

    def load(self) -> None:
        """Warm up the model by sending a keep_alive request."""
        self._client.post(
            "/api/generate",
            json={"model": self._model, "prompt": "", "keep_alive": "10m"},
        )

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
        **kwargs: object,
    ) -> GenerationResult:
        """Generate text via streaming, measuring TTFT from first chunk."""
        payload: dict[str, object] = {
            "model": self._model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
            },
        }

        text_parts: list[str] = []
        time_to_first_token = 0.0
        first_token_received = False
        prompt_tokens = 0
        completion_tokens = 0
        finish_reason = "stop"

        start = time.perf_counter()

        with self._client.stream("POST", "/api/generate", json=payload) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue

                chunk = json.loads(line)

                if not first_token_received and chunk.get("response"):
                    time_to_first_token = time.perf_counter() - start
                    first_token_received = True

                if chunk.get("response"):
                    text_parts.append(chunk["response"])

                if chunk.get("done"):
                    prompt_tokens = chunk.get("prompt_eval_count", 0)
                    completion_tokens = chunk.get("eval_count", 0)
                    if chunk.get("done_reason"):
                        finish_reason = chunk["done_reason"]
                    break

        total_time = time.perf_counter() - start

        return GenerationResult(
            text="".join(text_parts),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            time_to_first_token=time_to_first_token,
            total_time=total_time,
            finish_reason=finish_reason,
        )

    def unload(self) -> None:
        """Release model from VRAM by setting keep_alive to 0."""
        self._client.post(
            "/api/generate",
            json={"model": self._model, "prompt": "", "keep_alive": 0},
        )

    def metadata(self) -> dict[str, object]:
        """Fetch model info from Ollama's /api/show endpoint."""
        try:
            resp = self._client.post("/api/show", json={"name": self._model})
            resp.raise_for_status()
            data = resp.json()
            details = data.get("details", {})
            return {
                "parameter_size": details.get("parameter_size", "unknown"),
                "quantization_level": details.get("quantization_level", "unknown"),
                "family": details.get("family", "unknown"),
                "format": details.get("format", "unknown"),
            }
        except Exception:
            return {}

    def __del__(self) -> None:
        try:
            self._client.close()
        except Exception:
            pass
