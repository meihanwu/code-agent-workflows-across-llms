"""Verify raw API response durability and recovery before extraction."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from main import run_parallel, run_tasks
from model_client import CodeGenerator
from response_store import ResponseStore
from result_store import ResultStore


class ResponseTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix=".response-test-", dir=ROOT)
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.task = {"task_id": "HumanEval/0", "prompt": "def f(): pass"}
        self.results = ResultStore(self.root / "result.jsonl", "humaneval_plus")
        self.raw = ResponseStore(self.root / "responses/result.jsonl")

    def test_response_precedes_extraction_and_resume_reuses_it(self):
        payload = {"id": "test", "choices": [{"finish_reason": "stop", "message": {"content": "```python\ndef f(): return 1\n```", "reasoning": "reasoning text"}}], "usage": {"completion_tokens": 9}}
        with patch("model_client.OpenAI") as sdk:
            create = sdk.return_value.chat.completions.create
            create.return_value = SimpleNamespace(model_dump=lambda **kwargs: payload)
            generator = CodeGenerator("model", "http://localhost")
            with patch("model_client.extract_code", side_effect=RuntimeError("extraction failed")):
                with self.assertRaisesRegex(RuntimeError, "extraction failed"):
                    run_tasks("humaneval_plus", [self.task], results=self.results, generate=generator, responses=self.raw)
            self.assertEqual(json.loads(self.raw.path.read_text())["response"], payload)
            self.assertFalse(self.results.path.exists())
            run_tasks("humaneval_plus", [self.task], results=self.results, generate=generator, responses=ResponseStore(self.raw.path))
            self.assertEqual(self.results.records["HumanEval/0"]["solution"], "def f(): return 1")
            self.assertEqual(create.call_count, 1)

    def test_parallel_tasks_and_length_save_raw_before_results(self):
        payload = {"choices": [{"finish_reason": "length", "message": {"content": "partial", "reasoning": "analysis"}}]}
        task2 = {"task_id": "HumanEval/1", "prompt": "def g(): pass"}
        with patch("model_client.OpenAI") as sdk:
            sdk.return_value.chat.completions.create.return_value = SimpleNamespace(model_dump=lambda **kwargs: payload)
            generator = CodeGenerator("model", "http://localhost")
            run_parallel("humaneval_plus", [self.task, task2], None, self.results, generator, 2, self.raw)
        self.assertEqual(set(ResponseStore(self.raw.path).records), {"HumanEval/0", "HumanEval/1"})
        self.assertEqual({r["finish_reason"] for r in self.results.records.values()}, {"length"})
        self.assertEqual(self.raw.records["HumanEval/0"]["choices"][0]["message"]["reasoning"], "analysis")

    def test_no_code_block_saves_null_and_continues(self):
        payload = {"choices": [{"finish_reason": "stop", "message": {"content": "No solution found."}}]}
        with patch("model_client.OpenAI") as sdk:
            sdk.return_value.chat.completions.create.return_value = SimpleNamespace(model_dump=lambda **kwargs: payload)
            generator = CodeGenerator("model", "http://localhost")
            run_tasks("humaneval_plus", [self.task], results=self.results, generate=generator, responses=self.raw)
        self.assertEqual(self.results.records["HumanEval/0"], {"task_id": "HumanEval/0", "solution": None, "status": "failed", "finish_reason": "no_code"})
        self.assertEqual(self.raw.records["HumanEval/0"], payload)
        self.assertIsNone(json.loads(self.results.path.read_text())["solution"])
