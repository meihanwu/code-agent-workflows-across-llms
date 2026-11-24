"""Persist full API responses before extracting model-generated code."""

import json
import os
from pathlib import Path
import tempfile
from threading import Lock


class ResponseStore:
    def __init__(self, path):
        self.path = Path(path)
        self.records = {}
        self._lock = Lock()
        if self.path.exists():
            with self.path.open(encoding="utf-8") as stream:
                for number, line in enumerate(stream, 1):
                    try:
                        row = json.loads(line)
                        task_id = row["task_id"]
                        if not isinstance(task_id, str) or not task_id or not isinstance(row["response"], dict):
                            raise ValueError("Invalid response record")
                        if task_id in self.records:
                            raise ValueError("Duplicate task ID")
                        self.records[task_id] = row["response"]
                    except (ValueError, TypeError, KeyError) as exc:
                        raise ValueError(f"{self.path}:{number}: {exc}") from exc

    def save(self, task_id, response):
        if not isinstance(task_id, str) or not task_id or not isinstance(response, dict):
            raise ValueError("Invalid response record")
        with self._lock:
            if task_id in self.records:
                raise ValueError(f"Response already saved for {task_id}")
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd, temporary = tempfile.mkstemp(prefix="." + self.path.name + ".", dir=self.path.parent)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as stream:
                    for key, value in [*self.records.items(), (task_id, response)]:
                        stream.write(json.dumps({"task_id": key, "response": value}, ensure_ascii=False, allow_nan=False) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, self.path)
            finally:
                Path(temporary).unlink(missing_ok=True)
            self.records[task_id] = response
