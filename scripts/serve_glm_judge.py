import argparse
import json
import re
import threading

import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoProcessor, Glm4vForConditionalGeneration


class GenerateRequest(BaseModel):
    prompt: str
    schema: dict


class GenerateResponse(BaseModel):
    content: dict


def extract_json(text: str) -> dict:
    answer = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL)
    candidate = answer.group(1) if answer else text
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", candidate, re.DOTALL)
    if fenced:
        candidate = fenced.group(1)
    else:
        candidate = extract_first_json_object(candidate)
    value = json.loads(candidate)
    if not isinstance(value, dict):
        raise ValueError("GLM judge must return a JSON object")
    if {"$defs", "properties", "title", "type"}.issubset(value):
        raise ValueError("GLM judge returned a JSON Schema instead of a JSON instance")
    return value


def extract_first_json_object(text: str) -> str:
    start = text.find("{")
    if start < 0:
        raise ValueError("GLM judge did not return a JSON object")
    depth = 0
    in_string = False
    escaped = False
    for index, char in enumerate(text[start:], start=start):
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
                return text[start : index + 1]
    raise ValueError("GLM judge returned an incomplete JSON object")


def format_instruction(schema: dict) -> str:
    title = schema.get("title")
    if title == "ClaimSupportAssessment":
        return (
            "Fill this JSON template with the actual judgment:\n"
            '{"claim_index": 1, "verdict": "supported", '
            '"evidence_ids": ["E1"], '
            '"reasoning_summary": "Short reason grounded in the cited evidence."}\n'
            'Allowed verdict values: "supported", "partially_supported", "unsupported".'
        )
    if title == "SemanticCitationReport":
        return (
            "Fill this JSON template with one assessment for each claim:\n"
            '{"assessments": ['
            '{"claim_index": 1, "verdict": "supported", '
            '"evidence_ids": ["E1"], '
            '"reasoning_summary": "Short reason grounded in the cited evidence."}'
            "]}\n"
            'Allowed verdict values: "supported", "partially_supported", "unsupported".'
        )
    return "Required JSON Schema: " + json.dumps(schema, ensure_ascii=False)


def create_app(model_path: str, max_memory_gib: int, max_new_tokens: int) -> FastAPI:
    processor = AutoProcessor.from_pretrained(
        model_path, local_files_only=True, use_fast=True
    )
    max_memory = {
        index: f"{max_memory_gib}GiB" for index in range(torch.cuda.device_count())
    }
    max_memory["cpu"] = "32GiB"
    model = Glm4vForConditionalGeneration.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="auto",
        max_memory=max_memory,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    lock = threading.Lock()
    app = FastAPI(title="Local GLM Citation Judge")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "model": model_path, "device_map": model.hf_device_map}

    @app.post("/generate", response_model=GenerateResponse)
    def generate(request: GenerateRequest) -> GenerateResponse:
        instruction = format_instruction(request.schema)
        messages = [{
            "role": "user",
            "content": [{
                "type": "text",
                "text": (
                    "Return exactly one JSON object in the answer tag. "
                    "Do not return a JSON Schema. Do not explain outside JSON. "
                    f"{instruction}\n\nTask:\n{request.prompt}"
                ),
            }],
        }]
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {key: value.to(model.device) for key, value in inputs.items()}
        with lock, torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        generated = output[:, inputs["input_ids"].shape[1] :]
        text = processor.batch_decode(generated, skip_special_tokens=False)[0]
        return GenerateResponse(content=extract_json(text))

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve GLM as an isolated citation judge")
    parser.add_argument("--model", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--max-memory-gib", type=int, default=12)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    args = parser.parse_args()
    uvicorn.run(
        create_app(args.model, args.max_memory_gib, args.max_new_tokens),
        host=args.host,
        port=args.port,
        workers=1,
    )


if __name__ == "__main__":
    main()
