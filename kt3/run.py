from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sqlite3
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from time import perf_counter, process_time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))

from score_steps import corpus_document_keys, score_trace
from kt3.engine import MAX_STEPS, Session, run_turn
from kt3.tools import CORPUS, Corpus

HERE = Path(__file__).resolve().parent
VISIBLE_FILE = ROOT / "eval" / "visible.jsonl"
REFERENCE_TIME = "2026-10-01T12:00:00+03:00"
MODES = ("baseline", "system")
LLM_MODE = "llm_baseline"


def input_digest(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(b"\0")
        # Use canonical line endings for the same snapshot on Windows and Linux.
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
        digest.update(b"\0")
    return digest.hexdigest()


def quantiles(milliseconds: list[float]) -> dict:
    ordered = sorted(milliseconds)
    return {f"p{p}_ms": ordered[math.ceil(len(ordered) * p / 100) - 1] for p in (50, 95)}


def measure(corpus: Corpus, mode: str, session: Session, question: str, reference: str, rate: float,
            *, llm_runner=None, **options) -> dict:
    cpu_start = process_time()
    if mode == LLM_MODE:
        if llm_runner is None:
            raise ValueError("LLM runner is required for llm_baseline mode.")
        result = llm_runner.run(session, question, datetime.fromisoformat(reference),
                                restricted=options.get("restricted", False))
    else:
        result = run_turn(corpus, mode, session, question, datetime.fromisoformat(reference), **options)
    cpu_seconds = process_time() - cpu_start
    return {
        **result, "cpu_seconds": cpu_seconds,
        "model_cost_usd": result.get("model_cost_usd", 0.0),
        "estimated_cpu_cost_usd": cpu_seconds * rate / 3600,
    }


def translate_for_eval(record: dict, result: dict) -> dict:
    # Evaluation runs after the request. Tool steps are matched in order within their type.
    next_index = {}
    steps = []
    for step in result["steps"]:
        if "error" in step:
            continue
        kind = step["step"]
        for index in range(next_index.get(kind, 0), len(record["expected_steps"])):
            expected = record["expected_steps"][index]
            if expected["step"] != kind:
                continue
            mapped = {"step_index": index, "step": kind, "result": step["result"]}
            if "scope" in expected:
                # The prototype has no multi-window retrieval route. Do not assign
                # an unscoped result to an arbitrary month.
                continue
            steps.append(mapped)
            next_index[kind] = index + 1
            break
    return {"id": record["id"], "steps": steps}


def subset(expected, actual) -> bool:
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(
            key in actual and subset(value, actual[key]) for key, value in expected.items()
        )
    return expected == actual


def check_turn(result: dict, expected: dict) -> list[str]:
    checks = {
        "status": result["status"],
        "reason": result["reason"],
        "order": [f"{step['step']}:{step.get('error', {}).get('code', 'ok')}" for step in result["steps"]],
        "answer": result["answer"],
        "user_corrected": result["state"]["user_corrected"],
    }
    sql = next((step for step in result["steps"] if step["step"] == "sql"), None)
    calculator_step = next((step for step in result["steps"] if step["step"] == "calculate"), None)
    if sql:
        checks["sql_start"] = sql["input"]["windows"][0]["start"]
    if sql and calculator_step and "result" in sql:
        checks["calculator_from_sql"] = calculator_step["input"]["inputs"] == sql["result"]["rows"]
    checks["missing_answer_fields"] = expected.get("missing_answer_fields", []) if all(
        not isinstance(result["answer"], dict) or key not in result["answer"]
        for key in expected.get("missing_answer_fields", [])
    ) else None
    return [key for key, value in expected.items() if key not in checks or not subset(value, checks[key])]


def scenarios(corpus: Corpus, mode: str, cases: list[dict], rate: float) -> list[dict]:
    results = []
    for case in cases:
        session = Session()
        turns = []
        for turn in case["turns"]:
            output = measure(
                corpus, mode, session, turn["question"], turn.get("reference_time", REFERENCE_TIME), rate,
                step_limit=turn.get("step_limit", MAX_STEPS), fault=turn.get("fault"),
            )
            failures = check_turn(output, turn["expect"])
            turns.append({"input": turn, "output": output, "failed_checks": failures})
        results.append({"id": case["id"], "passed": all(not turn["failed_checks"] for turn in turns), "turns": turns})
    return results


def summarize(entries: list[dict]) -> dict:
    step_scores = [step for item in entries for step in item["step_eval"]]
    totals = {}
    for kind in ("sql", "retrieve_documents", "calculate"):
        steps = [step for step in step_scores if step["step"] == kind]
        if kind == "retrieve_documents":
            matched, expected = sum(step["hits"] for step in steps), sum(step["required"] for step in steps)
        else:
            matched, expected = sum(step["correct"] for step in steps), len(steps)
        totals[kind] = {"matched": matched, "expected": expected}
    counts = Counter(item["output"]["status"] for item in entries)
    return {
        "runs": len(entries), "behavior_matches": sum(item["behavior_matches"] for item in entries),
        "outcomes": dict(counts), "tool_metrics": totals,
        "latency": quantiles([item["output"]["duration_ms"] for item in entries]),
        "model_cost_usd": sum(item["output"]["model_cost_usd"] for item in entries),
        "cpu_seconds": sum(item["output"]["cpu_seconds"] for item in entries),
        "estimated_cpu_cost_usd": sum(item["output"]["estimated_cpu_cost_usd"] for item in entries),
    }


def evaluate(corpus: Corpus, mode: str, records: list[dict], repeats: int, rate: float, *, llm_runner=None) -> dict:
    document_keys = corpus_document_keys()
    entries = []
    for repeat in range(1, repeats + 1):
        for record in records:
            # Only user input and time cross the runtime boundary.
            output = measure(
                corpus, mode, Session(), record["question"], record["reference_time"], rate,
                llm_runner=llm_runner,
            )
            step_eval = score_trace(record, translate_for_eval(record, output), document_keys)
            entries.append({
                "id": record["id"], "repeat": repeat,
                "input": {"question": record["question"], "reference_time": record["reference_time"]},
                "expected_behavior": record["expected_behavior"],
                "behavior_matches": output["status"] == record["expected_behavior"],
                "output": output, "step_eval": step_eval,
            })
    return {
        "summary": summarize(entries), "by_repeat": [
            summarize([entry for entry in entries if entry["repeat"] == repeat]) for repeat in range(1, repeats + 1)
        ],
        "traces": entries,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run baseline and system on the frozen visible eval and KT3 scenarios.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--cpu-hour-usd", type=float, default=0.05, help="Assumed CPU-hour cost for an estimate.")
    parser.add_argument("--include-llm", action="store_true", help="Also run the paid OpenAI LLM baseline.")
    parser.add_argument("--llm-limit", type=int, help="Run the LLM baseline on the first N visible examples.")
    parser.add_argument("--llm-repeats", type=int, help="Repeat count for the LLM baseline. Defaults to --repeats.")
    args = parser.parse_args()
    if not 1 <= args.repeats <= 10 or not math.isfinite(args.cpu_hour_usd) or args.cpu_hour_usd < 0:
        parser.error("Expected 1-10 repeats and a finite nonnegative CPU-hour rate.")
    if args.llm_repeats is not None and not 1 <= args.llm_repeats <= 10:
        parser.error("Expected 1-10 LLM repeats.")
    if args.llm_limit is not None and not 1 <= args.llm_limit <= 45:
        parser.error("Expected an LLM limit from 1 to 45.")
    records = [json.loads(line) for line in VISIBLE_FILE.read_text(encoding="utf-8").splitlines()]
    if len(records) != 45 or {record["split"] for record in records} != {"visible"}:
        parser.error("Expected 45 visible records.")
    case_file = HERE / "scenarios.json"
    cases = json.loads(case_file.read_text(encoding="utf-8-sig"))
    llm_runner = None
    if args.include_llm:
        try:
            from kt3.llm_baseline import LLMBaselineRunner
            llm_runner = LLMBaselineRunner.from_environment()
        except ValueError as error:
            parser.error(str(error))
    clocks = (perf_counter(), process_time())
    corpus = Corpus()
    report = {
        "metadata": {
            "engine": "deterministic_rules_and_sql", "model": None, "prompts": [],
            "temperature": None, "seed": None, "per_case_step_limit": MAX_STEPS,
            "repeats": args.repeats, "eval_version": records[0]["eval_version"],
            "eval_sha256": input_digest([VISIBLE_FILE]),
            "scenarios_sha256": input_digest([case_file]),
            "corpus_sha256": input_digest(list((CORPUS / "tables").glob("*.csv")) + list((CORPUS / "documents").rglob("*.md"))),
            "code_sha256": input_digest(list(HERE.glob("*.py")) + [ROOT / "eval" / "score_steps.py"]),
            "python": platform.python_version(), "sqlite": sqlite3.sqlite_version,
            "architecture": platform.machine(), "os": platform.system(),
            "cpu_hour_usd_estimate": args.cpu_hour_usd,
        },
        "setup": {"wall_ms": round((perf_counter() - clocks[0]) * 1000, 3), "cpu_seconds": process_time() - clocks[1]},
        "modes": {},
    }
    for mode in MODES:
        data = evaluate(corpus, mode, records, args.repeats, args.cpu_hour_usd)
        data["scenarios"] = scenarios(corpus, mode, cases, args.cpu_hour_usd)
        data["summary"]["scenarios_passed"] = sum(case["passed"] for case in data["scenarios"])
        report["modes"][mode] = data
    if llm_runner is not None:
        llm_records = records[:args.llm_limit] if args.llm_limit is not None else records
        llm_repeats = args.llm_repeats if args.llm_repeats is not None else args.repeats
        data = evaluate(
            corpus, LLM_MODE, llm_records, llm_repeats, args.cpu_hour_usd,
            llm_runner=llm_runner,
        )
        data["summary"]["eval_records"] = len(llm_records)
        data["summary"]["scenarios_passed"] = None
        data["scenarios"] = []
        report["modes"][LLM_MODE] = data
        report["metadata"]["llm_baseline"] = {
            **llm_runner.metadata(),
            "eval_records": len(llm_records),
            "repeats": llm_repeats,
        }
    report["run_wall_ms"] = round((perf_counter() - clocks[0]) * 1000, 3)
    report["run_cpu_seconds"] = process_time() - clocks[1]
    report["run_estimated_cpu_cost_usd"] = report["run_cpu_seconds"] * args.cpu_hour_usd / 3600
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    print(json.dumps({mode: data["summary"] for mode, data in report["modes"].items()}, indent=2))
    for case in report["modes"]["system"]["scenarios"]:
        print(case["id"], "PASS" if case["passed"] else [turn["failed_checks"] for turn in case["turns"]])
    return 0 if all(case["passed"] for case in report["modes"]["system"]["scenarios"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())

