from typing import Protocol, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LLMClient(Protocol):
    def generate_structured(self, prompt: str, response_model: type[T]) -> T: ...

