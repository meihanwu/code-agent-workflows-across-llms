"""Save successful code and terminal task failures in one resumable JSONL file."""

import json
import os
from pathlib import Path
import tempfile
from threading import Lock


class ResultStore:
    def __init__(self, output_path, dataset):
        self.path = Path(output_path).expanduser()
        self.dataset = dataset
        self.records = {}
        self._write_lock = Lock()
        if self.path.exists():
            with self.path.open(encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, 1):
                    try:
                        record = json.loads(line)
                        self._validate(record)
                        task_id = record["task_id"]
                        if task_id in self.records:
                            raise ValueError(f"Duplicate saved task ID: {task_id!r}")
                        self.records[task_id] = record
                    except (ValueError, KeyError, TypeError) as exc:
                        raise ValueError(f"{self.path}:{line_number}: {exc}") from exc

    def _validate(self, record):
        if not isinstance(record, dict):
            raise ValueError("Each result must be an object")
        success_fields = {"task_id", "solution"}
        success_with_response = success_fields | {"response"}
        failure_fields = success_fields | {"status", "finish_reason", "response"}
        no_code_fields = success_fields | {"status", "finish_reason"}
        if set(record) not in (success_fields, success_with_response, failure_fields, no_code_fields):
            raise ValueError("Unexpected result fields")
        if not isinstance(record["task_id"], str) or not record["task_id"].strip():
            raise ValueError("Saved task_id must be a non-empty string")
        if set(record) == no_code_fields:
            if record["status"] != "failed" or record["finish_reason"] != "no_code" or record["solution"] is not None:
                raise ValueError("Invalid no-code result")
        elif set(record) == failure_fields:
            if record["status"] != "failed" or record["finish_reason"] != "length":
                raise ValueError("Invalid truncated result status")
            if record["solution"] != "" or not isinstance(record["response"], str):
                raise ValueError("A truncated result requires empty solution and raw response")
        elif not isinstance(record["solution"], str) or not record["solution"].strip():
            raise ValueError(f"Task {record['task_id']!r}: solution must be a non-empty code string")
        elif "response" in record and not isinstance(record["response"], str):
            raise ValueError("Raw response must be a string")

    def is_completed(self, task_id):
        return task_id in self.records

    def save(self, task_id, solution, response=None):
        record = {"task_id": task_id, "solution": solution}
        if response is not None:
            record["response"] = response
        self._save(record)

    def save_no_code(self, task_id):
        self._save({"task_id": task_id, "solution": None, "status": "failed", "finish_reason": "no_code"})

    def save_failure(self, task_id, response):
        self._save({
            "task_id": task_id,
            "solution": "",
            "status": "failed",
            "finish_reason": "length",
            "response": response,
        })

    def _save(self, record):
        self._validate(record)
        with self._write_lock:
            task_id = record["task_id"]
            if self.is_completed(task_id):
                raise ValueError(f"Task {task_id!r} is already saved")
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd, temporary = tempfile.mkstemp(prefix="." + self.path.name + ".", dir=self.path.parent)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as stream:
                    for row in [*self.records.values(), record]:
                        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, self.path)
            finally:
                Path(temporary).unlink(missing_ok=True)
            self.records[task_id] = record
