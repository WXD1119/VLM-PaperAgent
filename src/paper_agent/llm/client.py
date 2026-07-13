import json
import re
import urllib.error
import urllib.request
from typing import Protocol, TypeVar
from urllib.parse import urlparse

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
        self.url = self._normalize_generate_url(base_url)
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
        except urllib.error.HTTPError as exc:
            hint = ""
            if exc.code == 404:
                hint = (
                    " Check that the GLM judge service is running from "
                    "scripts/serve_glm_judge.py and that --judge-url points to "
                    "either http://host:port or http://host:port/generate."
                )
            raise RuntimeError(
                f"judge service request failed at {self.url}: HTTP Error {exc.code}: "
                f"{exc.reason}.{hint}"
            ) from exc
        except Exception as exc:
            raise RuntimeError(f"judge service request failed at {self.url}: {exc}") from exc
        if "content" not in body:
            raise ValueError("judge service response is missing content")
        return response_model.model_validate(body["content"])

    @staticmethod
    def _normalize_generate_url(base_url: str) -> str:
        """Accept either a service root URL or the concrete /generate endpoint."""
        stripped = base_url.strip().rstrip("/")
        if not stripped:
            raise ValueError("judge service URL must not be empty")
        parsed = urlparse(stripped)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError(
                "judge service URL must include scheme and host, e.g. http://127.0.0.1:8765"
            )
        if parsed.path.rstrip("/") == "/generate":
            return stripped
        if parsed.path in ("", "/"):
            return stripped + "/generate"
        return stripped


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
        return _extract_first_json_object(stripped)


def _extract_first_json_object(content: str) -> str:
    """Return the first balanced JSON object embedded in model output.

    Local thinking models sometimes emit prose after the JSON object, or repeat
    a corrected JSON object. A greedy ``rfind("}")`` slice turns that into
    invalid JSON with "Extra data". This scanner keeps string escaping rules and
    stops at the first balanced top-level object.
    """
    start = content.find("{")
    if start < 0:
        raise ValueError("local model did not return a JSON object")

    depth = 0
    in_string = False
    escaped = False
    for index, char in enumerate(content[start:], start=start):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return content[start : index + 1]

    raise ValueError("local model returned an incomplete JSON object")


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
        instruction = self._format_instruction(response_model)
        messages = [{
            "role": "user",
            "content": [{
                "type": "text",
                "text": (
                    "Return exactly one JSON object in the answer tag. "
                    "Do not return a JSON Schema. Do not explain outside JSON. "
                    f"{instruction}\n\nTask:\n{prompt}"
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
    def _format_instruction(response_model: type[BaseModel]) -> str:
        model_name = response_model.__name__
        if model_name == "ClaimSupportAssessment":
            return (
                "Fill this JSON template with the actual judgment:\n"
                '{"claim_index": 1, "verdict": "supported", '
                '"evidence_ids": ["E1"], '
                '"reasoning_summary": "Short reason grounded in the cited evidence."}\n'
                'Allowed verdict values: "supported", "partially_supported", "unsupported".'
            )
        if model_name == "SemanticCitationReport":
            return (
                "Fill this JSON template with one assessment for each claim:\n"
                '{"assessments": ['
                '{"claim_index": 1, "verdict": "supported", '
                '"evidence_ids": ["E1"], '
                '"reasoning_summary": "Short reason grounded in the cited evidence."}'
                "]}\n"
                'Allowed verdict values: "supported", "partially_supported", "unsupported".'
            )
        schema = json.dumps(response_model.model_json_schema(), ensure_ascii=False)
        return "Required JSON Schema: " + schema

    @staticmethod
    def _extract_json(content: str) -> dict:
        answer = re.search(r"<answer>\s*(.*?)\s*</answer>", content, re.DOTALL)
        candidate = answer.group(1) if answer else content
        value = json.loads(_extract_first_json_object(candidate.strip()))
        if not isinstance(value, dict):
            raise ValueError("GLM judge must return a JSON object")
        if {"$defs", "properties", "title", "type"}.issubset(value):
            raise ValueError("GLM judge returned a JSON Schema instead of a JSON instance")
        return value
