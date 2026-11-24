#!/usr/bin/env python3
"""Evaluate one-candidate LiveCodeBench-lite Medium results with LiveCodeBench."""

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import lcb_runner.benchmarks.code_generation as code_generation_module
from lcb_runner.benchmarks.code_generation import (
    CodeGenerationProblem,
    Difficulty,
)
from lcb_runner.evaluation import extract_instance_results
from lcb_runner.utils.scenarios import Scenario


# This fixed LiveCodeBench revision reads few-shot files relative to its source
# root while importing scenario_router. Import there, then restore the caller's
# working directory so CM-CodeAgents-relative input and output paths remain valid.
_CALLER_WORKING_DIRECTORY = Path.cwd()
_LIVECODEBENCH_ROOT = Path(code_generation_module.__file__).resolve().parents[2]
try:
    os.chdir(_LIVECODEBENCH_ROOT)
    from lcb_runner.runner.scenario_router import (  # noqa: E402
        get_metrics,
        sort_and_extract_save_results,
    )
finally:
    os.chdir(_CALLER_WORKING_DIRECTORY)


EXPECTED_TASK_COUNT = 167
DEFAULT_NUM_PROCESSES = 12
DEFAULT_TIMEOUT_SECONDS = 6


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def load_medium_benchmark(
    dataset_path: Path,
    expected_task_count: int = EXPECTED_TASK_COUNT,
) -> list[CodeGenerationProblem]:
    """Load and validate the prepared local release_v1 Medium JSONL."""

    if not dataset_path.is_file():
        raise FileNotFoundError(f"LiveCodeBench dataset not found: {dataset_path}")

    benchmark: list[CodeGenerationProblem] = []
    seen_ids: set[str] = set()
    with dataset_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at {dataset_path}:{line_number}: {exc.msg}"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected a JSON object at {dataset_path}:{line_number}"
                )

            question_id = record.get("question_id")
            if not isinstance(question_id, str) or not question_id.strip():
                raise ValueError(
                    f"Invalid question_id at {dataset_path}:{line_number}: "
                    f"{question_id!r}"
                )
            if question_id in seen_ids:
                raise ValueError(
                    f"Duplicate dataset question_id at {dataset_path}:{line_number}: "
                    f"{question_id}"
                )

            try:
                problem = CodeGenerationProblem(**record)
            except Exception as exc:
                raise ValueError(
                    f"Failed to construct CodeGenerationProblem for "
                    f"question_id={question_id} at {dataset_path}:{line_number}: {exc}"
                ) from exc
            if problem.difficulty != Difficulty.MEDIUM:
                raise ValueError(
                    f"Expected difficulty=medium for question_id={question_id}; "
                    f"found {problem.difficulty.value}"
                )
            if not problem.private_test_cases:
                raise ValueError(
                    f"private_test_cases is empty for question_id={question_id}"
                )

            seen_ids.add(question_id)
            benchmark.append(problem)

    if len(benchmark) != expected_task_count:
        raise ValueError(
            f"Expected {expected_task_count} LiveCodeBench-lite Medium tasks; "
            f"found {len(benchmark)} in {dataset_path}"
        )
    return sorted(benchmark, key=lambda problem: problem.question_id)


def load_and_align_candidates(
    samples_path: Path,
    benchmark: Sequence[CodeGenerationProblem],
) -> list[list[str]]:
    """Validate one candidate per task and align candidates by question_id."""

    if not samples_path.is_file():
        raise FileNotFoundError(f"LiveCodeBench candidates not found: {samples_path}")
    try:
        with samples_path.open("r", encoding="utf-8") as handle:
            records = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid candidate JSON in {samples_path}: {exc.msg}") from exc
    if not isinstance(records, list):
        raise ValueError(
            f"Expected candidate file to contain a JSON array: {samples_path}"
        )

    candidates_by_id: dict[str, list[str]] = {}
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"Candidate item {index} must be a JSON object")
        question_id = record.get("question_id")
        if not isinstance(question_id, str) or not question_id.strip():
            raise ValueError(
                f"Candidate item {index} has invalid question_id: {question_id!r}"
            )
        if question_id in candidates_by_id:
            raise ValueError(f"Duplicate candidate question_id: {question_id}")

        code_list = record.get("code_list")
        if not isinstance(code_list, list) or len(code_list) != 1:
            count = len(code_list) if isinstance(code_list, list) else "non-list"
            raise ValueError(
                f"Expected exactly one candidate for question_id={question_id}; "
                f"found {count}"
            )
        code = code_list[0]
        if not isinstance(code, str) or not code.strip():
            raise ValueError(
                f"Candidate code must be a non-empty string for "
                f"question_id={question_id}"
            )
        candidates_by_id[question_id] = [code]

    benchmark_ids = {problem.question_id for problem in benchmark}
    candidate_ids = set(candidates_by_id)
    missing = sorted(benchmark_ids - candidate_ids)
    extra = sorted(candidate_ids - benchmark_ids)
    if missing or extra:
        raise ValueError(
            "Candidate question_id set does not match benchmark: "
            f"missing={missing}, extra={extra}"
        )

    return [candidates_by_id[problem.question_id] for problem in benchmark]


def evaluate_candidates(
    benchmark: Sequence[CodeGenerationProblem],
    aligned_candidates: Sequence[list[str]],
    num_process_evaluate: int,
    timeout: int,
) -> tuple[list[dict[str, Any]], list[Any], list[dict[str, Any]]]:
    """Run the official LiveCodeBench code-generation evaluation pipeline."""

    if len(benchmark) != len(aligned_candidates):
        raise ValueError(
            "Benchmark and aligned candidate lengths differ: "
            f"{len(benchmark)} != {len(aligned_candidates)}"
        )

    save_results = [
        problem.insert_output(code_list, code_list)
        for problem, code_list in zip(benchmark, aligned_candidates, strict=True)
    ]
    save_results, combined_results = sort_and_extract_save_results(
        Scenario.codegeneration,
        save_results,
    )
    args = SimpleNamespace(
        scenario=Scenario.codegeneration,
        num_process_evaluate=num_process_evaluate,
        timeout=timeout,
    )
    metrics = get_metrics(
        Scenario.codegeneration,
        args,
        list(benchmark),
        combined_results,
    )
    if not isinstance(metrics, list) or len(metrics) != 3:
        raise RuntimeError(
            f"Unexpected LiveCodeBench metrics structure: {type(metrics).__name__}"
        )

    graded = extract_instance_results(metrics[1])
    metadatas = metrics[2]
    if len(graded) != len(benchmark) or len(metadatas) != len(benchmark):
        raise RuntimeError(
            "LiveCodeBench returned incomplete per-task results: "
            f"benchmark={len(benchmark)}, graded={len(graded)}, "
            f"metadata={len(metadatas)}"
        )

    save_eval_results = [
        problem.insert_output_evaluation(
            outputs_list,
            extracted_list,
            graded_list,
            metadata=metadata,
        )
        for problem, (outputs_list, extracted_list), graded_list, metadata in zip(
            benchmark,
            combined_results,
            graded,
            metadatas,
            strict=True,
        )
    ]
    return save_results, metrics, save_eval_results


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=4, ensure_ascii=False)


def run_evaluation(
    dataset_path: Path,
    samples_path: Path,
    output_dir: Path,
    num_process_evaluate: int = DEFAULT_NUM_PROCESSES,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    expected_task_count: int = EXPECTED_TASK_COUNT,
) -> dict[str, Any]:
    benchmark = load_medium_benchmark(dataset_path, expected_task_count)
    aligned_candidates = load_and_align_candidates(samples_path, benchmark)
    save_results, metrics, save_eval_results = evaluate_candidates(
        benchmark,
        aligned_candidates,
        num_process_evaluate,
        timeout,
    )

    prefix = output_dir / f"{samples_path.stem}_codegeneration_output"
    output_path = Path(f"{prefix}.json")
    eval_path = Path(f"{prefix}_eval.json")
    eval_all_path = Path(f"{prefix}_eval_all.json")
    write_json(output_path, save_results)
    write_json(eval_path, metrics)
    write_json(eval_all_path, save_eval_results)

    pass_at_1 = metrics[0].get("pass@1")
    if not isinstance(pass_at_1, (int, float)):
        raise RuntimeError(
            f"LiveCodeBench did not return a numeric pass@1: {pass_at_1!r}"
        )
    return {
        "task_count": len(benchmark),
        "pass@1": float(pass_at_1),
        "output_path": str(output_path),
        "eval_path": str(eval_path),
        "eval_all_path": str(eval_all_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate CM-CodeAgents candidates on the prepared LiveCodeBench-lite "
            "release_v1 Medium test set."
        )
    )
    parser.add_argument("--dataset_path", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--num_process_evaluate",
        type=positive_int,
        default=DEFAULT_NUM_PROCESSES,
    )
    parser.add_argument(
        "--timeout",
        type=positive_int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="Per-test timeout in seconds.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_evaluation(
        dataset_path=args.dataset_path,
        samples_path=args.samples,
        output_dir=args.output_dir,
        num_process_evaluate=args.num_process_evaluate,
        timeout=args.timeout,
    )
    print(
        f"LiveCodeBench-lite Medium: tasks={summary['task_count']} "
        f"pass@1={summary['pass@1']:.6f}"
    )
    print(f"Saved aligned outputs: {summary['output_path']}")
    print(f"Saved aggregate metrics: {summary['eval_path']}")
    print(f"Saved per-task results: {summary['eval_all_path']}")


if __name__ == "__main__":
    main()
