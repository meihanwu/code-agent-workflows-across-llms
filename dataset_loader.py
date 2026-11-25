"""Read original dataset records from a supplied JSONL path."""

import gzip
import json
from pathlib import Path


DATASETS = {
    "humaneval_plus": {
        "default": ("HumanEvalPlus/humanevalplus.jsonl", 164),
    },
    "mbpp_plus": {
        "default": ("MBPPEval/MBPPPlus-NoExtreme.jsonl", 378),
    },
    "livecodebench": {
        "default": ("livecodebench/LiveCodeBench-lite-medium.jsonl", 167),
    },
    "livecodebench_hard": {
        "default": ("livecodebench/LiveCodeBench-lite-hard.jsonl", 90),
    },
    "apps-introductory": {
        "format": "apps",
        "apps_split": "introductory",
        "default": ("APPS/APPS-test-introductory.jsonl", 1000),
    },
    "apps-interview": {
        "format": "apps",
        "apps_split": "interview",
        "default": ("APPS/APPS-test-interview.jsonl", 3000),
    },
    "apps-competition": {
        "format": "apps",
        "apps_split": "competition",
        "default": ("APPS/APPS-test-competition.jsonl", 1000),
    },
}

TASK_ID_FIELDS = {
    "humaneval_plus": "task_id",
    "mbpp_plus": "task_id",
    "livecodebench": "question_id",
    "livecodebench_hard": "question_id",
    "apps": "id",
}
DIFFICULTIES = {
    "livecodebench": ("easy", "medium", "hard"),
    "livecodebench_hard": ("hard",),
    "apps": ("introductory", "interview", "competition"),
}


def get_task_id(dataset, task):
    task_id = task[TASK_ID_FIELDS[dataset]]
    expected_type = int if dataset == "apps" else str
    if type(task_id) is not expected_type or (isinstance(task_id, str) and not task_id.strip()):
        raise ValueError(f"Invalid {dataset} task ID: {task_id!r}")
    return f"APPS/{task_id}" if dataset == "apps" else task_id


def load_dataset(data_path, dataset, difficulty=None):
    """Preserve source order/fields and reject IDs that make resume ambiguous."""
    if difficulty and difficulty not in DIFFICULTIES.get(dataset, ()):
        raise ValueError(f"Invalid difficulty {difficulty!r} for {dataset}")
    path = Path(data_path).expanduser()
    tasks = []
    seen_ids = set()
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                task = json.loads(line)
                if not isinstance(task, dict):
                    raise ValueError("Expected a JSON object")
                task_id = get_task_id(dataset, task)
                if task_id in seen_ids:
                    raise ValueError(f"Duplicate task ID: {task_id!r}")
                seen_ids.add(task_id)
                if not difficulty or task["difficulty"] == difficulty:
                    tasks.append(task)
            except (ValueError, KeyError, TypeError) as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
    if not tasks:
        raise ValueError(f"No matching tasks in {path}")
    return tasks
