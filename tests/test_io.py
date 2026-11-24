"""Check dataset identity, atomic saving, and per-task resume without model calls."""

import gzip
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dataset_loader import get_task_id, load_dataset
from main import main, run_tasks
from model_client import TruncatedResponse
from result_store import ResultStore


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix=".io-test-", dir=ROOT)
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name)
        self.tasks = [
            {"task_id": f"HumanEval/{i}", "prompt": f'def f{i}():\n    """Return {i}."""\n',
             "canonical_solution": "REFERENCE_MUST_NOT_BE_SAVED", "test": "HIDDEN_TEST"}
            for i in range(3)
        ]
        self.output = self.path / "Qwen3-32B/humaneval_plus.jsonl"

    def write_tasks(self, rows):
        path = self.path / "tasks.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        return path

    def store(self):
        return ResultStore(self.output, "humaneval_plus")

    def test_four_source_id_fields_and_raw_preservation(self):
        cases = {
            "humaneval_plus": {"task_id": "HumanEval/0", "prompt": "text", "test": "hidden"},
            "mbpp_plus": {"task_id": "Mbpp/2", "prompt": "text"},
            "livecodebench": {"question_id": "1883_B", "difficulty": "medium"},
            "apps": {"id": 3, "difficulty": "interview"},
        }
        for dataset, task in cases.items():
            with self.subTest(dataset=dataset):
                loaded = load_dataset(self.write_tasks([task]), dataset)
                self.assertEqual(loaded, [task])
                self.assertIsInstance(get_task_id(dataset, loaded[0]), str)
                if dataset == "apps":
                    self.assertEqual(get_task_id(dataset, loaded[0]), "APPS/3")

    def test_duplicate_missing_and_wrong_ids_stop(self):
        for rows, dataset in [
            ([self.tasks[0], self.tasks[0]], "humaneval_plus"),
            ([{"prompt": "x"}], "humaneval_plus"),
            ([{"id": True}], "apps"),
            ([{"id": "3"}], "apps"),
        ]:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                load_dataset(self.write_tasks(rows), dataset)

    def test_gzip_jsonl_preserves_records(self):
        path = self.path / "tasks.jsonl.gz"
        with gzip.open(path, "wt", encoding="utf-8") as stream:
            for task in self.tasks:
                stream.write(json.dumps(task) + "\n")
        self.assertEqual(load_dataset(path, "humaneval_plus"), self.tasks)

    def test_bad_json_and_empty_dataset_stop(self):
        path = self.write_tasks([])
        with self.assertRaises(ValueError):
            load_dataset(path, "humaneval_plus")
        path.write_text('{"task_id":')
        with self.assertRaisesRegex(ValueError, "tasks.jsonl:1"):
            load_dataset(path, "humaneval_plus")

    def test_difficulty_is_dataset_specific(self):
        tasks = [{"id": 0, "difficulty": "interview"}, {"id": 1, "difficulty": "competition"}]
        path = self.write_tasks(tasks)
        self.assertEqual(load_dataset(path, "apps", "competition"), tasks[1:])
        with self.assertRaises(ValueError):
            load_dataset(path, "apps", "medium")
        with self.assertRaises(ValueError):
            load_dataset(path, "humaneval_plus", "medium")

    def test_interrupted_generation_resumes_only_unsaved_tasks(self):
        calls = []
        def interrupted(prompt):
            calls.append(prompt.user)
            if len(calls) == 2:
                raise RuntimeError("simulated interruption")
            return "def fixture():\n    return 1\n", "raw first"
        with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
            run_tasks("humaneval_plus", self.tasks, results=self.store(), generate=interrupted)
        self.assertEqual(list(self.store().records), ["HumanEval/0"])
        resumed_calls = []
        def complete(prompt):
            resumed_calls.append(prompt.user)
            return "def fixture():\n    return 2\n", "raw second"
        self.assertEqual(run_tasks("humaneval_plus", self.tasks, results=self.store(), generate=complete), 2)
        self.assertEqual(len(resumed_calls), 2)
        self.assertEqual(len(self.store().records), 3)
        self.assertEqual(run_tasks("humaneval_plus", self.tasks, results=self.store(), generate=complete), 0)
        self.assertEqual(len(resumed_calls), 2)
        self.assertNotIn("REFERENCE_MUST_NOT_BE_SAVED", self.output.read_text())
        self.assertNotIn("HIDDEN_TEST", self.output.read_text())

    def test_failed_save_keeps_previous_file_and_progress(self):
        store = self.store()
        store.save("HumanEval/0", "first fixture code")
        before = self.output.read_bytes()
        with patch("result_store.os.replace", side_effect=OSError("simulated disk failure")):
            with self.assertRaises(OSError):
                store.save("HumanEval/1", "second fixture code")
        self.assertEqual(self.output.read_bytes(), before)
        self.assertFalse(store.is_completed("HumanEval/1"))
        self.assertEqual(list(self.output.parent.iterdir()), [self.output])

    def test_saved_result_contains_only_id_and_code(self):
        self.store().save("HumanEval/0", "fixture code")
        record = json.loads(self.output.read_text())
        self.assertEqual(record, {"task_id": "HumanEval/0", "solution": "fixture code"})
        record["prompt"] = "extra field"
        self.output.write_text(json.dumps(record) + "\n")
        with self.assertRaises(ValueError):
            self.store()

    def test_empty_solution_does_not_complete_but_wrong_code_is_recorded(self):
        store = self.store()
        for solution in ("", " ", None):
            with self.assertRaises(ValueError):
                store.save("HumanEval/0", solution)
        self.assertFalse(self.output.exists())
        store.save("HumanEval/0", "syntactically invalid code: !!!")
        self.assertTrue(self.store().is_completed("HumanEval/0"))

    def test_corrupt_or_duplicate_saved_rows_are_not_skipped(self):
        self.store().save("HumanEval/0", "fixture code")
        row = self.output.read_text()
        self.output.write_text(row + row)
        with self.assertRaisesRegex(ValueError, "Duplicate saved task"):
            self.store()
        self.output.write_text(row + '{"task_id":')
        with self.assertRaisesRegex(ValueError, "humaneval_plus.jsonl:2"):
            self.store()

    def test_cli_selection_does_not_expand_on_resume(self):
        data_path = self.write_tasks(self.tasks)
        self.store().save("HumanEval/0", "fixture code")
        argv = ["main.py", "--dataset", "humaneval_plus", "--data-path", str(data_path),
                "--output-dir", str(self.path), "--task-id", "HumanEval/1"]
        with patch.object(sys, "argv", argv), patch("main.CodeGenerator") as generator:
            generator.return_value.return_value = ("print(1)", "raw response")
            main()
            generator.return_value.assert_called_once()
        self.assertEqual(set(self.store().records), {"HumanEval/0", "HumanEval/1"})


    def test_generation_requires_a_result_store_before_calling_model(self):
        with self.assertRaises(ValueError):
            run_tasks("humaneval_plus", self.tasks, generate=lambda _: self.fail("must not call"))

    def test_result_dataset_mismatch_stops_before_calling_model(self):
        with self.assertRaisesRegex(ValueError, "different dataset"):
            run_tasks("mbpp_plus", [], results=self.store(), generate=lambda _: self.fail("must not call"))


    def test_length_failure_is_saved_and_next_task_runs_and_resume_skips_it(self):
        calls = []
        def generate(prompt):
            calls.append(prompt.user)
            if len(calls) == 1:
                raise TruncatedResponse("unfinished analysis")
            return "print(1)", "raw print"
        self.assertEqual(run_tasks("humaneval_plus", self.tasks, results=self.store(), generate=generate), 3)
        rows = self.store().records
        self.assertEqual(rows["HumanEval/0"], {
            "task_id": "HumanEval/0", "solution": "", "status": "failed",
            "finish_reason": "length", "response": "unfinished analysis",
        })
        self.assertEqual(rows["HumanEval/1"]["solution"], "print(1)")
        self.assertEqual(len(rows), 3)
        self.assertEqual(run_tasks("humaneval_plus", self.tasks, results=self.store(),
                                  generate=lambda _: self.fail("must skip processed tasks")), 0)

    def test_failure_save_is_atomic(self):
        store = self.store()
        store.save("HumanEval/0", "print(1)")
        before = self.output.read_bytes()
        with patch("result_store.os.replace", side_effect=OSError("disk failure")):
            with self.assertRaises(OSError):
                store.save_failure("HumanEval/1", "unfinished analysis")
        self.assertEqual(self.output.read_bytes(), before)
        self.assertFalse(store.is_completed("HumanEval/1"))


if __name__ == "__main__":
    unittest.main()
