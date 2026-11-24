"""Offline checks: every model request is mocked."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import main
from model_client import CodeGenerator, extract_code
from prompt_loader import load_prompt


class ModelTests(unittest.TestCase):
    def test_humaneval_prompt_excludes_test_fields_in_every_profile(self):
        from prompt_loader import list_profiles
        generation = 'def f():\n    """Return one."""\n'
        task = {
            "prompt": generation,
            "test": "HIDDEN_OFFICIAL_TEST",
            "public_test_cases": ["PUBLIC_TEST_MUST_NOT_BE_ADDED"],
            "base_input": ["HIDDEN_BASE_INPUT"],
            "plus_input": ["HIDDEN_PLUS_INPUT"],
        }
        for profile in list_profiles("humaneval_plus"):
            with self.subTest(profile=profile):
                actual = load_prompt("humaneval_plus", task, profile)
                expected = load_prompt("humaneval_plus", {"prompt": generation}, profile)
                self.assertEqual(actual, expected)
                self.assertNotIn("HIDDEN_", repr(actual))
                self.assertNotIn("PUBLIC_TEST_MUST_NOT_BE_ADDED", repr(actual))

    def test_extracts_only_answer_code(self):
        text = '<think>not an answer</think>\nExplanation\n```python\ndef f():\n    return 1\n```\nDone'
        self.assertEqual(extract_code(text), 'def f():\n    return 1')
        self.assertEqual(extract_code('```\npython\ndef f():\n    return 1\n```'), 'def f():\n    return 1')
        self.assertEqual(extract_code('```\nPY\nprint(1)\n```'), 'print(1)')
        self.assertEqual(extract_code('print(1)'), 'print(1)')
        self.assertEqual(extract_code('def add(a, b):\n    return a + b'), 'def add(a, b):\n    return a + b')
        self.assertIsNone(extract_code('This is not Python code.'))
        self.assertIsNone(extract_code('def broken(:'))
        self.assertIsNone(extract_code(''))
        self.assertIsNone(extract_code('<think>no answer</think>'))

    def test_python_block_beats_prior_formula_block(self):
        text = "Explanation\n```\nx = 1 – n\n```\n```python\nprint(1)\n```"
        self.assertEqual(extract_code(text), "print(1)")
        self.assertEqual(extract_code("```python\nprint(1)\n```\n```py\nprint(2)\n```"), "print(2)")

    def test_last_python_tests_do_not_replace_solution(self):
        text = "```python\ndef solve(x):\n    return x + 1\n```\nTests:\n```python\nassert solve(1) == 2\n```"
        self.assertEqual(extract_code(text), "def solve(x):\n    return x + 1")
        text = "```python\ndef solve():\n    return 1\n```\n```python\ndef test_solve():\n    assert solve() == 1\n```"
        self.assertEqual(extract_code(text), "def solve():\n    return 1")
        text = "```python\ndef solve(x):\n    assert x >= 0\n    return x\n```"
        self.assertEqual(extract_code(text), "def solve(x):\n    assert x >= 0\n    return x")

    def test_qwen_request_uses_system_role_and_disables_thinking(self):
        prompt = load_prompt('humaneval_plus', {'prompt': 'def f():\n    """Return one."""'})
        with patch('model_client.OpenAI') as sdk:
            create = sdk.return_value.chat.completions.create
            create.return_value = SimpleNamespace(choices=[SimpleNamespace(
                finish_reason='stop', message=SimpleNamespace(content='```python\ndef f():\n    return 1\n```')
            )])
            generator = CodeGenerator('Qwen3-32B', 'http://127.0.0.1:18011/v1', 0, 4096, 180)
            self.assertEqual(generator(prompt), ('def f():\n    return 1', '```python\ndef f():\n    return 1\n```'))
            request = create.call_args.kwargs
            self.assertEqual(request['model'], 'Qwen3-32B')
            self.assertEqual(request['messages'], prompt.to_messages())
            self.assertEqual(request['extra_body'], {'chat_template_kwargs': {'enable_thinking': False}})
            self.assertEqual(sdk.call_args.kwargs['max_retries'], 0)

    def test_starcoder_moves_system_instructions_into_user_message(self):
        prompt = load_prompt("humaneval_plus", {"prompt": "def f(): pass"})
        with patch("model_client.OpenAI") as sdk:
            create = sdk.return_value.chat.completions.create
            create.return_value = SimpleNamespace(choices=[SimpleNamespace(
                finish_reason="stop", message=SimpleNamespace(content="def f(): pass")
            )])
            CodeGenerator("StarCoder2-15B-Instruct-v0.1", "http://localhost/v1")(prompt)
            self.assertEqual(create.call_args.kwargs["messages"], [{
                "role": "user", "content": prompt.system + "\n\n" + prompt.user,
            }])

    def test_gpt_oss_reasoning_effort_defaults_high_and_can_be_overridden(self):
        with patch("model_client.OpenAI"):
            default = CodeGenerator("gpt-oss-20b", "http://localhost/v1")
            self.assertEqual(default.extra_body["reasoning_effort"], "high")
            low = CodeGenerator("gpt-oss-20b", "http://localhost/v1", reasoning_effort="low")
            self.assertEqual(low.extra_body["reasoning_effort"], "low")
            other = CodeGenerator("Qwen3-32B", "http://localhost/v1")
            self.assertNotIn("reasoning_effort", other.extra_body)

    def test_incomplete_or_empty_model_response_fails(self):
        prompt = load_prompt('humaneval_plus', {'prompt': 'def f(): pass'})
        with patch('model_client.OpenAI') as sdk:
            generator = CodeGenerator('Qwen3-32B', 'http://localhost/v1')
            create = sdk.return_value.chat.completions.create
            create.return_value = SimpleNamespace(choices=[SimpleNamespace(
                finish_reason='length', message=SimpleNamespace(content='partial')
            )])
            with self.assertRaises(RuntimeError):
                generator(prompt)
            create.return_value = SimpleNamespace(choices=[])
            with self.assertRaises(ValueError):
                generator(prompt)
            create.return_value = SimpleNamespace(choices=[SimpleNamespace(
                finish_reason='stop', message=SimpleNamespace(content=None)
            )])
            self.assertEqual(generator(prompt), (None, ''))

    def test_main_saves_one_minimal_file_per_dataset_and_resumes(self):
        fixtures = {
            'humaneval_plus': {'task_id': 'HumanEval/0', 'prompt': 'def f(): pass'},
            'mbpp_plus': {'task_id': 'Mbpp/2', 'prompt': '"""Write a function."""'},
            'livecodebench': {'question_id': '1883_B', 'question_content': 'Print one.', 'starter_code': ''},
            'apps-150': {'id': 4023, 'question': 'Print one.', 'starter_code': '', 'input_output': '{}'},
        }
        with tempfile.TemporaryDirectory(prefix='.model-test-', dir=ROOT) as directory:
            directory = Path(directory)
            with patch('main.CodeGenerator') as generator, redirect_stdout(io.StringIO()):
                generator.return_value.return_value = ('print(1)', '```python\nprint(1)\n```')
                for dataset, task in fixtures.items():
                    data = directory / (dataset + '.input.jsonl')
                    data.write_text(json.dumps(task) + '\n')
                    argv = ['main.py', '--dataset', dataset, '--data-path', str(data), '--output-dir', str(directory/'results')]
                    with patch.object(sys, 'argv', argv):
                        main.main()
                    output = directory/'results/Qwen3-32B'/f'{dataset}.jsonl'
                    rows = [json.loads(line) for line in output.read_text().splitlines()]
                    self.assertEqual(len(rows), 1)
                    self.assertEqual(set(rows[0]), {'task_id', 'solution'})
                    self.assertEqual(rows[0]['solution'], 'print(1)')
                    if dataset == 'apps-150':
                        self.assertEqual(rows[0]['task_id'], 'APPS/4023')
                    previous_calls = generator.return_value.call_count
                    with patch.object(sys, 'argv', argv):
                        main.main()
                    self.assertEqual(generator.return_value.call_count, previous_calls)
                self.assertEqual(len(list((directory/'results/Qwen3-32B').glob('*.jsonl'))), 4)


if __name__ == '__main__':
    unittest.main()
