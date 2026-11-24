#!/usr/bin/env python3
"""Run final correctness evaluation for the saved one-candidate experiments."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from dataset_loader import DATASETS, get_task_id, load_dataset
from evaluate_worker import EVALPLUS_DATASETS, LIVECODEBENCH_DATASETS
from result_store import ResultStore

ROOT = Path(__file__).resolve().parent

JUDGES = ROOT / "judges"
DOCKER_CPUS = {"evalplus": 4, "livecodebench": 12, "apps": 4}
APPS_JUDGE = JUDGES / "apps_official_compare"
IMAGE = "sha256:f26809eb13339cbc59c3d0cc972f8c4997830dc8d2121cf18089cb122834e10d"


def write_atomic(path, text):
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
            stream.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def write_json(path, value):
    write_atomic(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def save_verdicts(path, rows):
    write_atomic(path, "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
                              for row in rows))


def isolated_verdicts(dataset, data, candidates, output, *, judge, workers):
    env = Path(sys.executable).resolve().parent.parent
    evaluator = JUDGES / judge
    with tempfile.TemporaryDirectory(prefix=".judge-", dir=output) as temporary:
        stage = Path(temporary)
        input_dir, output_dir = stage / "input", stage / "output"
        input_dir.mkdir()
        output_dir.mkdir()
        target = "/input/" + data.name
        shutil.copyfile(data, input_dir / data.name)
        write_json(input_dir / "request.json", dict(
            dataset=dataset, data_path=target, candidates=candidates,
            workers=min(workers, len(candidates)),
        ))
        command = [
            "docker", "run", "--rm", "--runtime", "runc", "--network", "none",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--read-only",
            "--tmpfs", "/tmp:rw,size=2g", "--pids-limit", "256",
            "--memory", "24g", "--memory-swap", "24g", "--cpus", str(DOCKER_CPUS[judge]),
            "--user", f"{os.getuid()}:{os.getgid()}", "--workdir", "/tmp",
            "--volume", f"{ROOT / 'evaluate_worker.py'}:/evaluate_worker.py:ro",
            "--volume", f"{env}:{env}:ro",
            "--volume", f"{evaluator}:/judge:ro",
            "--volume", f"{input_dir}:/input:ro",
            "--volume", f"{output_dir}:/output",
        ]
        if judge == "evalplus":
            command.extend(["--volume", f"{Path.home() / '.cache/evalplus'}:/groundtruth:ro"])
        command.extend([
            "--entrypoint", "/usr/bin/env", IMAGE, "-i", "PATH=/usr/bin:/bin",
            "HOME=/tmp", "XDG_CACHE_HOME=/tmp/cache", "PYTHONDONTWRITEBYTECODE=1",
            "PYTHONPATH=/judge",
            "OMP_NUM_THREADS=1", "OPENBLAS_NUM_THREADS=1", "MKL_NUM_THREADS=1",
            str(env / "bin/python"), "-u", "/evaluate_worker.py",
        ])
        subprocess.run(command, check=True)
        return [json.loads(line) for line in (output_dir / "verdicts.jsonl").read_text().splitlines()]


def apps_official_verdicts(tasks, candidates, workers):
    """Run the official comparator with script entry enabled for stdin candidates."""
    env = Path(sys.executable).resolve().parent.parent
    with tempfile.TemporaryDirectory(prefix=".apps-official-") as temporary:
        input_dir = Path(temporary)
        write_json(input_dir / "payload.json", {"workers": min(workers, len(candidates)), "tasks": [
            {"task_id": task_id, "input_output": json.loads(tasks[task_id]["input_output"]),
             "solution": solution, "generation_failed": False}
            for task_id, solution in candidates.items()
        ]})
        command = [
            "docker", "run", "--rm", "--runtime", "runc", "--network", "none",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--read-only",
            "--tmpfs", "/tmp:rw,size=1g", "--tmpfs", "/output:rw,size=128m,mode=1777",
            "--pids-limit", "128", "--memory", "8g", "--memory-swap", "8g", "--cpus", str(DOCKER_CPUS["apps"]),
            "--user", f"{os.getuid()}:{os.getgid()}", "--workdir", "/tmp",
            "--volume", f"{env}:{env}:ro",
            "--volume", f"{APPS_JUDGE}:/judge:ro",
            "--volume", f"{input_dir}:/input:ro",
            "--entrypoint", str(env / "bin/python"), IMAGE, "-u", "/judge/runner.py", "--stdin-main",
        ]
        with subprocess.Popen(command, stdout=subprocess.PIPE, text=True, encoding="utf-8") as process:
            for line in process.stdout:
                result = json.loads(line)
                task_id = result["task_id"]
                if "error" in result:
                    raise RuntimeError(f"{task_id}: APPS official runner: {result['error']}")
                passed = result["passed"]
                if type(passed) is not bool:
                    raise ValueError(f"Invalid APPS official verdict for {task_id}")
                outcome = "PASSED" if passed else "FAILED"
                if result.get("global_timeout"):
                    outcome = "TIME_LIMIT_EXCEEDED"
                yield {"task_id": task_id, "passed": passed, "outcome": outcome,
                       "executed_case_count": 0 if result.get("no_worker_result") else result["case_count"],
                       "final_case_count": result["total_cases"], "result_types": result["result_types"]}
            status = process.wait()
            if status:
                raise subprocess.CalledProcessError(status, command)


def evaluate_pair(args):
    dataset_name = args.dataset
    dataset_config = DATASETS[dataset_name]
    dataset_format = dataset_config.get("format", dataset_name)
    dataset_file, expected_count = dataset_config["default"]
    dataset_path = ROOT / "data" / dataset_file
    
    tasks_by_id = {
        get_task_id(dataset_format, task): task
        for task in load_dataset(dataset_path, dataset_format)
    }
    results_by_id = ResultStore(args.results_file, dataset_format).records
    
    selected = set(args.task_id) if args.task_id else tasks_by_id.keys()
    if (len(tasks_by_id) != expected_count or not selected <= tasks_by_id.keys()
            or not selected <= results_by_id.keys() or not results_by_id.keys() <= tasks_by_id.keys()):
        raise ValueError(f"{dataset_name}: task IDs and saved solutions do not align")
    
    # ["HumanEval/1", "HumanEval/3", ...]
    ids = [task_id for task_id in tasks_by_id if task_id in selected]

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        verdicts = {}
        candidates = {}
        for task_id in ids:
            result = results_by_id[task_id]
            if result.get("status") == "failed":
                verdicts[task_id] = {"task_id": task_id, "passed": False,
                                     "outcome": "GENERATION_FAILED",
                                     "finish_reason": result["finish_reason"]}
            else:
                candidates[task_id] = result["solution"]
        print(f"{dataset_name}: evaluating {len(candidates)}/{len(ids)} candidates", flush=True)
        
        if candidates:
            if dataset_name in EVALPLUS_DATASETS:
                rows = isolated_verdicts(dataset_name, dataset_path, candidates, output,
                                         judge="evalplus", workers=args.workers or 4)
            elif dataset_name in LIVECODEBENCH_DATASETS:
                rows = isolated_verdicts(dataset_name, dataset_path, candidates, output,
                                         judge="livecodebench", workers=args.workers or 12)
            elif dataset_format == "apps":
                rows = apps_official_verdicts(tasks_by_id, candidates, workers=args.workers or 1)
            else:
                raise ValueError(f"No judge configured for dataset: {dataset_name}")

            for row in rows:
                task_id = row["task_id"]
                if task_id not in candidates or task_id in verdicts or type(row["passed"]) is not bool:
                    raise ValueError(f"Invalid evaluator verdict for {task_id}")
                verdicts[task_id] = row
                if dataset_format == "apps":
                    print(f"{dataset_name}: {len(verdicts)}/{len(ids)} {task_id} {row['outcome']}", flush=True)
        if set(verdicts) != set(ids):
            raise RuntimeError(f"{dataset_name}: evaluator produced incomplete results")
        
        save_verdicts(output / "verdicts.jsonl", (verdicts[task_id] for task_id in ids))
        passed = sum(row["passed"] for row in verdicts.values())
        metrics = {"dataset": dataset_name, "task_count": len(ids),
                   "passed": passed, "pass@1": passed / len(ids),
                   "generation_failed": sum(row.get("outcome") == "GENERATION_FAILED" for row in verdicts.values())}
        if dataset_name in EVALPLUS_DATASETS:
            metrics["base_passed"] = sum(row.get("base_status") == "pass" for row in verdicts.values())
            metrics["base_pass@1"] = metrics["base_passed"] / len(ids)
        write_json(output / "metrics.json", metrics)
        print(json.dumps(metrics, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--results-file", type=Path, required=True,
                        help="Generated results JSONL file")
    parser.add_argument("--task-id", action="append", help="Evaluate a task by its dataset ID. Repeat to select multiple tasks.")
    parser.add_argument("--output-dir", type=Path,
                        help="Override evaluation/<model>/<dataset>/ inferred from the parent directory and stem of --results-file.")
    parser.add_argument("--workers", type=int, help="Maximum tasks judged concurrently (defaults: EvalPlus 4, LiveCodeBench 12, APPS 1).")
    args = parser.parse_args()
    if args.workers is not None and args.workers < 1:
        parser.error("--workers must be a positive integer")
    if args.output_dir is None:
        results_file = Path(os.path.abspath(args.results_file.expanduser()))
        args.output_dir = ROOT / "evaluation" / results_file.parent.name / results_file.stem
    evaluate_pair(args)


if __name__ == "__main__":
    main()
