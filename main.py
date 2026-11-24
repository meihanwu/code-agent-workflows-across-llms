"""Generate code, save one result file per dataset, and resume saved tasks."""

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import fcntl
import os
from pathlib import Path

from dataset_loader import DATASETS, get_task_id, load_dataset
from model_client import CodeGenerator, TruncatedResponse
from prompt_loader import PROFILE_KINDS, load_prompt
from result_store import ResultStore
from response_store import ResponseStore


ROOT = Path(__file__).resolve().parent


def run_tasks(dataset, tasks, profile=None, results=None, generate=None, responses=None):
    if results is None or generate is None:
        raise ValueError("Code generation requires a result store and a generator")
    if results.dataset != dataset:
        raise ValueError("Result store uses a different dataset")
    processed = 0
    for task in tasks:
        task_id = get_task_id(dataset, task)
        if results.is_completed(task_id):
            continue
        prompt = load_prompt(dataset, task, profile)
        try:
            if responses is None:
                solution, response = generate(prompt)
            elif task_id in responses.records:
                solution, response = generate.process_response(responses.records[task_id])
            else:
                solution, response = generate(
                    prompt, on_response=lambda raw: responses.save(task_id, raw)
                )
        except TruncatedResponse as exc:
            results.save_failure(task_id, exc.content)
            print(f"Failed {task_id}: output length limit; raw response saved", flush=True)
        else:
            if solution is None:
                results.save_no_code(task_id)
                print(f"Failed {task_id}: no code block", flush=True)
            else:
                results.save(task_id, solution)
                print(f"Saved {task_id}", flush=True)
        processed += 1
    return processed


def run_parallel(dataset, tasks, profile, results, generate, concurrency, responses=None):
    pending = iter(tasks)
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        active = {}

        def submit_next():
            task = next(pending, None)
            if task is not None:
                future = pool.submit(run_tasks, dataset, [task], profile, results, generate, responses)
                active[future] = get_task_id(dataset, task)

        for _ in range(concurrency):
            submit_next()
        while active:
            done, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in done:
                task_id = active.pop(future)
                try:
                    future.result()
                except Exception as exc:
                    raise RuntimeError(f"{dataset} {task_id}: generation failed: {exc}") from exc
            for _ in done:
                submit_next()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--data-path", required=True, help="Input JSONL dataset file.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--model", default="Qwen3-32B")
    parser.add_argument("--base-url", default=os.getenv("VLLM_BASE_URL", "http://127.0.0.1:18011/v1"))
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--min-p", type=float)
    parser.add_argument("--presence-penalty", type=float)
    parser.add_argument("--repetition-penalty", type=float)
    parser.add_argument("--reasoning-effort", choices=["low", "medium", "high"])
    parser.add_argument("--max-tokens", type=int, help="Output token limit (default: 4096; GPT-OSS uses the server limit).")
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--parallel-tasks", type=int, default=1,help="Maximum simultaneous tasks for this dataset.")
    parser.add_argument("--workers", "--concurrency", dest="parallel_tasks", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--task-id", help="Generate for one task by its dataset ID, e.g. HumanEval/0 or APPS/4023.")
    parser.add_argument("--profile", choices=PROFILE_KINDS)
    args = parser.parse_args()

    if args.parallel_tasks < 1:
        parser.error("--parallel-tasks must be positive")

    # Data
    dataset_config = DATASETS[args.dataset]
    dataset = dataset_config.get("format", args.dataset)
    tasks = load_dataset(args.data_path, dataset, dataset_config.get("apps_split"))
    
    # Save
    output_path = args.output_dir.expanduser() / args.model.replace("/", "_") / f"{args.dataset}.jsonl"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output_path.parent / f".{output_path.name}.lock"
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another process is writing {output_path}") from exc

        results = ResultStore(output_path, dataset)
        responses = ResponseStore(output_path.parent / "responses" / output_path.name)
        
        task_ids = {get_task_id(dataset, task) for task in tasks}
        unknown = set(results.records) - task_ids
        if unknown:
            raise ValueError(f"Saved task IDs do not belong to this data selection: {sorted(unknown)}")
        if args.task_id is not None:
            tasks = [task for task in tasks if get_task_id(dataset, task) == args.task_id]
            if not tasks:
                raise ValueError(f"Task ID {args.task_id!r} not found in the selected dataset")
        
        generate = CodeGenerator(
            args.model, args.base_url, args.temperature, args.max_tokens, args.timeout,
            top_p=args.top_p, top_k=args.top_k, min_p=args.min_p,
            presence_penalty=args.presence_penalty, repetition_penalty=args.repetition_penalty,
            reasoning_effort=args.reasoning_effort,
        )
        if args.parallel_tasks > 1:
            pending = [
                task for task in tasks
                if not results.is_completed(get_task_id(dataset, task))
            ]
            run_parallel(dataset, pending, args.profile, results, generate, args.parallel_tasks, responses)
            processed = len(pending)
        else:
            processed = run_tasks(dataset, tasks, args.profile, results, generate, responses)
        print(f"Processed {processed} new tasks; results saved to {output_path}")

if __name__ == "__main__":
    main()
