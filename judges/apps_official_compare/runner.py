"""Run the APPS tester in an offline sandbox, one child per candidate."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import contextlib
import inspect
import json
import multiprocessing as mp
import os
from pathlib import Path
import traceback
from types import SimpleNamespace

if not hasattr(inspect, "getargspec"):
    inspect.getargspec = inspect.getfullargspec
import testing_util


def evaluate_one(conn, task, solution, stdin_main=False):
    try:
        if stdin_main and task.get("fn_name") is None:
            # The official tester wraps the candidate in a function. Set its
            # global script name before that function is called in this child.
            from_string = testing_util.RuntimeModule.from_string

            def load_stdin_module(name, doc, source):
                return from_string(name, doc, '__name__ = "__main__"\n' + source)

            testing_util.RuntimeModule = SimpleNamespace(from_string=load_stdin_module)
        with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            if stdin_main:
                os.dup2(sink.fileno(), 1)
                os.dup2(sink.fileno(), 2)
            result = testing_util.run_test(problem={"input_output": task}, test=solution, debug=False)
        if not isinstance(result, list) or not result:
            raise RuntimeError("APPS returned no verdicts")
        normalized = []
        for value in result:
            if hasattr(value, "item"):
                try:
                    value = value.item()
                except ValueError:
                    value = value.item(0)
            normalized.append(value)
        passed = all(value > 0 for value in normalized)
        conn.send({"passed": passed, "case_count": len(result), "total_cases": len(task["inputs"]), "result_types": [str(x)[:30] for x in normalized if not x > 0][:3]})
    except BaseException:
        conn.send({"error": traceback.format_exc()[-3000:]})
    finally:
        conn.close()


def judge_task(row, stdin_main):
    if row["generation_failed"]:
        result = {"passed": False, "generation_failed": True}
    else:
        ctx = mp.get_context("fork")
        parent, child = ctx.Pipe(duplex=False)
        process = ctx.Process(
            target=evaluate_one,
            args=(child, row["input_output"], row["solution"], stdin_main),
        )
        process.start()
        child.close()
        timeout = 11  # Official test_one_solution.py: TIMEOUT=10 and join(timeout+1)
        result = None
        timed_out = not parent.poll(timeout)
        if timed_out:
            process.kill()
        else:
            try:
                result = parent.recv()
            except EOFError:
                pass
        process.join(10)
        if process.is_alive():
            process.kill()
            process.join()
        parent.close()
        if result is None:
            # The official driver counts a missing worker result as failure.
            result = {
                "passed": False, "case_count": 21, "result_types": ["-1"],
                "total_cases": len(row["input_output"]["inputs"]),
                "no_worker_result": True, "global_timeout": timed_out,
                "worker_exitcode": process.exitcode,
            }
    result["task_id"] = row["task_id"]
    if "current_passed" in row:
        result["current_passed"] = row["current_passed"]
    return result


def evaluate_tasks(tasks, workers, stdin_main):
    if workers == 1:
        for row in tasks:
            yield judge_task(row, stdin_main)
        return

    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("fork")) as pool:
        futures = [pool.submit(judge_task, row, stdin_main) for row in tasks]
        for future in as_completed(futures):
            yield future.result()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stdin-main", action="store_true", help="Run standard-input candidates with the Python script entry point")
    args = parser.parse_args()
    payload = json.loads(Path("/input/payload.json").read_text())
    filename = "official-stdin-main.jsonl" if args.stdin_main else "official.jsonl"
    dest = Path("/output") / filename
    saved = {}
    if dest.exists():
        for line in dest.read_text().splitlines():
            x = json.loads(line)
            if x["task_id"] in saved:
                raise RuntimeError("duplicate result")
            saved[x["task_id"]] = x
    tasks = [row for row in payload["tasks"] if row["task_id"] not in saved]
    with dest.open("a", encoding="utf-8") as out:
        for result in evaluate_tasks(tasks, payload.get("workers", 1), args.stdin_main):
            out.write(json.dumps(result, ensure_ascii=False) + "\n"); out.flush(); os.fsync(out.fileno())
            print(json.dumps(result, ensure_ascii=False), flush=True)
            if "error" in result:
                raise RuntimeError(f"{result['task_id']}: {result['error']}")


if __name__ == "__main__": main()
