from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Literal

from pydantic import BaseModel

from kt3.engine import ACCESS_DENIED, MAX_INPUT, Session

HERE = Path(__file__).resolve().parent
PROMPT_FILE = HERE / "prompts" / "llm_baseline_v1.txt"
MODEL = "gpt-6.2-sol"
PROMPT_VERSION = "llm_baseline_v1"
MAX_OUTPUT_TOKENS = 800
REQUEST_TIMEOUT_SECONDS = 30.0
PRICING_SNAPSHOT_DATE = "2026-11-23"
PRICING_USD_PER_MILLION = {"input": 2.0, "cached_input": 0.1, "output": 10.0}


class BaselineReply(BaseModel):
    status: Literal["answer", "clarify", "refuse"]
    text: str
    reason: str


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def as_dict(value) -> dict:
    if value is None:
        return {}
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, dict):
        return value
    return {}


def estimate_cost_usd(usage: dict) -> float:
    input_tokens = int(usage.get("input_tokens") or 0)
    output_tokens = int(usage.get("output_tokens") or 0)
    input_details = usage.get("input_tokens_details") or {}
    cached_tokens = min(input_tokens, int(input_details.get("cached_tokens") or 0))
    regular_tokens = input_tokens - cached_tokens
    cost = (
        regular_tokens * PRICING_USD_PER_MILLION["input"]
        + cached_tokens * PRICING_USD_PER_MILLION["cached_input"]
        + output_tokens * PRICING_USD_PER_MILLION["output"]
    ) / 1_000_000
    return round(cost, 8)


def simple_result(status: str, text: str, reason: str) -> dict:
    return {"status": status, "text": text, "reason": reason, "answer": None}


@dataclass
class LLMBaselineRunner:
    client: object
    sdk_version: str
    prompt: str
    model: str = MODEL

    @classmethod
    def from_environment(cls) -> "LLMBaselineRunner":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set.")
        try:
            import openai
            from openai import OpenAI
        except ImportError as error:
            raise ValueError("Install dependencies from requirements.txt before the LLM run.") from error
        client = OpenAI(api_key=api_key, timeout=REQUEST_TIMEOUT_SECONDS, max_retries=0)
        return cls(
            client=client,
            sdk_version=openai.__version__,
            prompt=PROMPT_FILE.read_text(encoding="utf-8"),
        )

    def metadata(self) -> dict:
        return {
            "provider": "OpenAI",
            "api": "Responses API",
            "model_requested": self.model,
            "prompt_file": PROMPT_FILE.relative_to(HERE.parent).as_posix(),
            "prompt_version": PROMPT_VERSION,
            "prompt_sha256": sha256_text(self.prompt),
            "reasoning_effort": "low",
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "temperature": None,
            "seed": None,
            "request_timeout_seconds": REQUEST_TIMEOUT_SECONDS,
            "max_retries": 0,
            "sdk": "openai",
            "sdk_version": self.sdk_version,
            "pricing_snapshot_date": PRICING_SNAPSHOT_DATE,
            "pricing_usd_per_million_tokens": PRICING_USD_PER_MILLION,
        }

    def run(self, session: Session, question: str, as_of: datetime, *, restricted: bool = False) -> dict:
        started = perf_counter()
        state_read = len(session.history)
        corrected = False
        model_calls = 0
        steps = []
        model_metadata = self.metadata()

        if not 0 < len(question.strip()) <= MAX_INPUT:
            result = simple_result(
                "clarify", "Введите вопрос длиной от 1 до 2000 символов.", "invalid_question"
            )
        elif restricted or ACCESS_DENIED.search(question):
            result = simple_result(
                "refuse",
                "Запрос требует доступа или действий вне разрешённого чтения.",
                "access_denied",
            )
        else:
            session.expire(as_of)
            previous_questions = list(session.history)
            normalized, corrected = session.turn(question, as_of)
            payload = {
                "reference_time": as_of.isoformat(),
                "previous_questions": previous_questions,
                "question": normalized,
            }
            call_started = perf_counter()
            event = {
                "step_index": 0,
                "step": "llm",
                "input": {
                    "model_requested": self.model,
                    "prompt_version": PROMPT_VERSION,
                    "prompt_sha256": model_metadata["prompt_sha256"],
                    "reference_time": payload["reference_time"],
                    "history_messages": len(previous_questions),
                },
            }
            try:
                model_calls = 1
                response = self.client.responses.parse(
                    model=self.model,
                    instructions=self.prompt,
                    input=json.dumps(payload, ensure_ascii=False),
                    reasoning={"effort": "low"},
                    max_output_tokens=MAX_OUTPUT_TOKENS,
                    text_format=BaselineReply,
                    store=False,
                )
                reply = response.output_parsed
                if reply is None:
                    raise ValueError("The model returned no structured output.")
                usage = as_dict(response.usage)
                actual_model = getattr(response, "model", None)
                event["result"] = {
                    "response_id": getattr(response, "id", None),
                    "model_actual": actual_model,
                    "status": reply.status,
                    "usage": usage,
                }
                model_metadata["model_actual"] = actual_model
                result = {
                    "status": reply.status,
                    "text": reply.text,
                    "reason": reply.reason,
                    "answer": {"text": reply.text} if reply.status == "answer" else None,
                    "usage": usage,
                    "model_cost_usd": estimate_cost_usd(usage),
                }
            except Exception as error:
                event["error"] = {
                    "code": "model_error",
                    "type": type(error).__name__,
                    "message": "Вызов модели завершился ошибкой или вернул невалидный формат.",
                }
                result = {
                    **simple_result(
                        "report_model_error",
                        "Вызов модели завершился ошибкой. Повторный вызов автоматически не выполнялся.",
                        "model_error",
                    ),
                    "usage": {},
                    "model_cost_usd": 0.0,
                }
            finally:
                event["duration_ms"] = round((perf_counter() - call_started) * 1000, 3)
                steps.append(event)

        return {
            **result,
            "steps": steps,
            "facts": [],
            "duration_ms": round((perf_counter() - started) * 1000, 3),
            "state": {
                "before": state_read,
                "after": len(session.history),
                "user_corrected": corrected,
            },
            "step_budget": 1,
            "model_calls": model_calls,
            "model_metadata": model_metadata,
            "model_cost_usd": result.get("model_cost_usd", 0.0),
        }
