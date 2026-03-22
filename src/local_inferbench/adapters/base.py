from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class GenerationResult:
    """Result from a single generation call."""

    text: str
    prompt_tokens: int
    completion_tokens: int
    time_to_first_token: float
    total_time: float
    finish_reason: str


class BaseAdapter(ABC):
    """All inference backend adapters implement this interface."""

    @abstractmethod
    def name(self) -> str:
        """Human-readable backend name, e.g. 'ollama', 'llamacpp'."""
        ...

    @abstractmethod
    def model_id(self) -> str:
        """Model identifier as the backend understands it."""
        ...

    @abstractmethod
    def load(self) -> None:
        """Pre-warm the model (pull into memory, etc). May be a no-op."""
        ...

    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
        **kwargs: object,
    ) -> GenerationResult:
        """Run a single generation and return structured result."""
        ...

    @abstractmethod
    def unload(self) -> None:
        """Release model resources. May be a no-op."""
        ...

    def metadata(self) -> dict[str, object]:
        """Optional: return model metadata (parameter count, quant, context length)."""
        return {}
