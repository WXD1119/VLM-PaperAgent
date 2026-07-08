import json
import re
import urllib.request
from typing import Protocol, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LLMClient(Protocol):
    def generate_structured(self, prompt: str, response_model: type[T]) -> T: ...


class OpenAICompatibleClient:
    """Structured-output client for OpenAI-compatible APIs such as DeepSeek and GLM."""

    def __init__(self, model: str, api_key: str, base_url: str, temperature: float = 0) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install LLM dependencies: pip install -e '.[llm]'") from exc
        self.model = model
        self.temperature = temperature
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def generate_structured(self, prompt: str, response_model: type[T]) -> T:
        schema = json.dumps(response_model.model_json_schema(), ensure_ascii=False)
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": "Return one JSON object only. Required JSON Schema: " + schema,
                },
                {"role": "user", "content": prompt},
            ],
        )
        content = response.choices[0].message.content
        if not content:
            raise ValueError("LLM returned empty content")
        return response_model.model_validate_json(content)


class RemoteStructuredClient:
    """Call the isolated local judge service without adding an HTTP dependency."""

    def __init__(self, base_url: str, timeout: float = 300) -> None:
        self.url = base_url.rstrip("/") + "/generate"
        self.timeout = timeout

    def generate_structured(self, prompt: str, response_model: type[T]) -> T:
        payload = json.dumps(
            {"prompt": prompt, "schema": response_model.model_json_schema()},
            ensure_ascii=False,
        ).encode("utf-8")
        request = urllib.request.Request(
            self.url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            raise RuntimeError(f"judge service request failed: {exc}") from exc
        if "content" not in body:
            raise ValueError("judge service response is missing content")
        return response_model.model_validate(body["content"])


class TransformersStructuredClient:
    """Run a local Hugging Face causal/VLM checkpoint without an API service."""

    def __init__(
        self,
        model_path: str,
        device_map: str = "auto",
        device: str | None = None,
        max_new_tokens: int = 1024,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as exc:
            raise RuntimeError("Install local VLM dependencies: pip install -e '.[local-vlm]'") from exc

        self.torch = torch
        self.max_new_tokens = max_new_tokens
        self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        placement = {"": device} if device else device_map
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            dtype="auto",
            device_map=placement,
            local_files_only=True,
        )

    def generate_structured(self, prompt: str, response_model: type[T]) -> T:
        schema = json.dumps(response_model.model_json_schema(), ensure_ascii=False)
        messages = [
            {
                "role": "system",
                "content": "Return one JSON object only. Required JSON Schema: " + schema,
            },
            {"role": "user", "content": prompt},
        ]
        rendered = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(text=[rendered], return_tensors="pt", padding=True)
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
        with self.torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        prompt_length = inputs["input_ids"].shape[1]
        content = self.processor.batch_decode(
            generated[:, prompt_length:], skip_special_tokens=True
        )[0]
        return response_model.model_validate_json(self._extract_json(content))

    @staticmethod
    def _extract_json(content: str) -> str:
        stripped = content.strip()
        fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", stripped, re.DOTALL)
        if fenced:
            return fenced.group(1)
        start, end = stripped.find("{"), stripped.rfind("}")
        if start < 0 or end < start:
            raise ValueError("local model did not return a JSON object")
        return stripped[start : end + 1]


class GlmStructuredClient:
    """Local GLM-V client intended for an isolated, sequential judge process."""

    def __init__(
        self,
        model_path: str,
        max_memory_gib: int = 12,
        max_new_tokens: int = 2048,
    ) -> None:
        try:
            import torch
            from transformers import AutoProcessor, Glm4vForConditionalGeneration
        except ImportError as exc:
            raise RuntimeError("Install judge dependencies: pip install -e '.[judge]'") from exc
        self.torch = torch
        self.max_new_tokens = max_new_tokens
        self.processor = AutoProcessor.from_pretrained(
            model_path, local_files_only=True, use_fast=True
        )
        max_memory = {
            index: f"{max_memory_gib}GiB" for index in range(torch.cuda.device_count())
        }
        max_memory["cpu"] = "32GiB"
        self.model = Glm4vForConditionalGeneration.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map="auto",
            max_memory=max_memory,
            low_cpu_mem_usage=True,
            local_files_only=True,
        )

    def generate_structured(self, prompt: str, response_model: type[T]) -> T:
        schema = json.dumps(response_model.model_json_schema(), ensure_ascii=False)
        messages = [{
            "role": "user",
            "content": [{
                "type": "text",
                "text": (
                    "Return exactly one JSON object matching this schema in the answer tag. "
                    f"Schema: {schema}\n\nTask:\n{prompt}"
                ),
            }],
        }]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
        with self.torch.inference_mode():
            generated = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False
            )
        new_tokens = generated[:, inputs["input_ids"].shape[1] :]
        content = self.processor.batch_decode(new_tokens, skip_special_tokens=False)[0]
        return response_model.model_validate(self._extract_json(content))

    @staticmethod
    def _extract_json(content: str) -> dict:
        answer = re.search(r"<answer>\s*(.*?)\s*</answer>", content, re.DOTALL)
        candidate = answer.group(1) if answer else content
        value = json.loads(TransformersStructuredClient._extract_json(candidate))
        if not isinstance(value, dict):
            raise ValueError("GLM judge must return a JSON object")
        return value
