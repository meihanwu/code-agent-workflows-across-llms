#!/usr/bin/env python3
"""Run EvalPlus or LiveCodeBench inside the evaluation container."""

from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path


EVALPLUS_DATASETS = {"humaneval_plus", "mbpp_plus"}
LIVECODEBENCH_DATASETS = {"livecodebench", "livecodebench_hard"}


def evalplus_candidate(name, problem, solution, oracle):
    from evalplus.eval import untrusted_check

    statuses = {}
    for suite in ("base", "plus"):
        status, _ = untrusted_check(
            name, solution, problem[f"{suite}_input"], problem["entry_point"],
            expected=oracle[suite], atol=problem["atol"],
            ref_time=oracle[f"{suite}_time"], fast_check=True,
        )
        if status not in ("pass", "fail", "timeout"):
            raise ValueError(f"Invalid EvalPlus verdict for {problem['task_id']}")
        statuses[suite] = status
        if status != "pass":
            break
    plus = statuses.get("plus", "skipped")
    return {"task_id": problem["task_id"], "passed": plus == "pass",
            "base_status": statuses["base"], "plus_status": plus}


def evalplus_verdicts(dataset, data_path, candidates, workers):
    human = dataset == "humaneval_plus"
    os.environ["HUMANEVAL_OVERRIDE_PATH" if human else "MBPP_OVERRIDE_PATH"] = str(data_path)
    from evalplus import evaluate as backend
    from evalplus.data import (
        get_human_eval_plus, get_human_eval_plus_hash, get_mbpp_plus, get_mbpp_plus_hash,
    )
    if human:
        name = "humaneval"
        problems, data_hash = get_human_eval_plus(), get_human_eval_plus_hash()
        non_none = []
    else:
        name = "mbpp"
        problems, data_hash = get_mbpp_plus(noextreme=True), get_mbpp_plus_hash(noextreme=True)
        non_none = backend.MBPP_OUTPUT_NOT_NONE_TASKS
    if (Path("/groundtruth") / f"{data_hash}.pkl").is_file():
        backend.CACHE_DIR = "/groundtruth"
    oracle = backend.get_groundtruth(problems, data_hash, non_none)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(evalplus_candidate, name, problems[task_id], solution, oracle[task_id])
            for task_id, solution in candidates.items()
        ]
        for future in futures:
            yield future.result()


def lcb_verdicts(dataset, data_path, candidates, workers, timeout):
    from evaluate_lcb_medium import evaluate_candidates, load_medium_benchmark
    if dataset == "livecodebench_hard":
        from lcb_runner.benchmarks.code_generation import CodeGenerationProblem, Difficulty
        benchmark = []
        with data_path.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                record = json.loads(line)
                if record["question_id"] not in candidates:
                    continue
                problem = CodeGenerationProblem(**record)
                if problem.difficulty != Difficulty.HARD or not problem.private_test_cases:
                    raise ValueError(f"Invalid LiveCodeBench Hard task {problem.question_id}")
                benchmark.append(problem)
        benchmark.sort(key=lambda problem: problem.question_id)
    else:
        benchmark = [p for p in load_medium_benchmark(data_path) if p.question_id in candidates]
    _, _, results = evaluate_candidates(
        benchmark, [[candidates[p.question_id]] for p in benchmark], workers, timeout
    )
    for row in results:
        graded = row["graded_list"]
        if len(graded) != 1 or type(graded[0]) is not bool:
            raise ValueError(f"Invalid LiveCodeBench verdict for {row['question_id']}")
        yield {"task_id": row["question_id"], "passed": graded[0], "metadata": row.get("metadata")}


def worker():
    request = json.loads(Path("/input/request.json").read_text())
    dataset = request["dataset"]
    path = Path(request["data_path"])
    candidates = request["candidates"]
    if dataset in LIVECODEBENCH_DATASETS:
        rows = lcb_verdicts(dataset, path, candidates, request["workers"], timeout=6)
    elif dataset in EVALPLUS_DATASETS:
        rows = evalplus_verdicts(dataset, path, candidates, request["workers"])
    else:
        raise ValueError(f"Unsupported worker dataset: {dataset}")
    with Path("/output/verdicts.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()


if __name__ == "__main__":
    worker()
