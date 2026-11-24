"""Generate Python code through a vLLM chat endpoint."""

import ast
import os
import re

from openai import OpenAI


def is_test_only(code):
    """Recognize standalone assertions or test functions after a solution."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    definitions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
    if any(not node.name.startswith(("test_", "Test")) for node in definitions):
        return False
    return bool(definitions) or any(isinstance(node, ast.Assert) for node in ast.walk(tree))


def extract_code(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    python_blocks = re.findall(r"```(?:python|py)[ \t]*\r?\n(.*?)```", text, flags=re.S | re.I)
    blocks = python_blocks or re.findall(r"```[ \t]*\r?\n(.*?)```", text, flags=re.S)
    if python_blocks:
        solutions = [block for block in python_blocks if not is_test_only(block)]
        code = (solutions[-1] if solutions else python_blocks[-1]).strip()
    elif blocks:
        code = blocks[0].strip()
    else:
        try:
            ast.parse(text)
        except (SyntaxError, ValueError):
            return None
        code = text
    if blocks and re.match(r"^(?:python|py)[ \t]*\r?\n", code, flags=re.I):
        code = re.sub(r"^(?:python|py)[ \t]*\r?\n", "", code, count=1, flags=re.I).strip()
    return code or None


class TruncatedResponse(RuntimeError):
    def __init__(self, content):
        super().__init__("Generation did not finish: length")
        self.content = content


class CodeGenerator:
    def __init__(
        self, model, base_url, temperature=0, max_tokens=None, timeout=180, *,
        top_p=None, top_k=None, min_p=None, presence_penalty=None, repetition_penalty=None,
        reasoning_effort=None,
    ):
        self.client = OpenAI(
            base_url=base_url,
            api_key=os.getenv("VLLM_API_KEY") or "EMPTY",
            timeout=timeout,
            max_retries=0,
        )
        self.model = model
        self.temperature = temperature
        if max_tokens is None and model.lower() != "gpt-oss-20b":
            max_tokens = 4096
        self.max_tokens = max_tokens
        self.sampling = {
            key: value
            for key, value in {"top_p": top_p, "presence_penalty": presence_penalty}.items()
            if value is not None
        }
        self.extra_body = {"chat_template_kwargs": {"enable_thinking": False}}
        if reasoning_effort is None and model.lower() == "gpt-oss-20b":
            reasoning_effort = "high"
        if reasoning_effort is not None:
            self.extra_body["reasoning_effort"] = reasoning_effort
        self.extra_body.update({
            key: value
            for key, value in {"top_k": top_k, "min_p": min_p, "repetition_penalty": repetition_penalty}.items()
            if value is not None
        })

    def __call__(self, prompt, *, on_response=None):
        if prompt.kind != "chat":
            raise ValueError("Code generation uses a chat profile; choose the dataset's default profile")
        messages = prompt.to_messages()
        if self.model.lower() == "starcoder2-15b-instruct-v0.1" and messages[0]["role"] == "system":
            messages = [{"role": "user", "content": messages[0]["content"] + "\n\n" + messages[1]["content"]}]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            **({"max_tokens": self.max_tokens} if self.max_tokens is not None else {}),
            extra_body=self.extra_body,
            **self.sampling,
        )
        if on_response is not None:
            raw = response.model_dump(mode="json")
            on_response(raw)
            return self.process_response(raw)
        if not response.choices:
            raise ValueError("Model returned no choices")
        choice = response.choices[0]
        if choice.finish_reason == "length":
            raise TruncatedResponse(choice.message.content or "")
        if choice.finish_reason != "stop":
            raise RuntimeError(f"Generation did not finish: {choice.finish_reason}")
        content = choice.message.content or ""
        if not isinstance(content, str):
            raise ValueError("Model returned a non-text answer")
        return extract_code(content), content

    @staticmethod
    def process_response(response):
        choices = response.get("choices") or []
        if not choices:
            raise ValueError("Model returned no choices")
        choice = choices[0]
        message = choice.get("message") or {}
        content = message.get("content")
        if choice.get("finish_reason") == "length":
            raise TruncatedResponse(content or "")
        if choice.get("finish_reason") != "stop":
            raise RuntimeError(f"Generation did not finish: {choice.get('finish_reason')}")
        content = content or ""
        if not isinstance(content, str):
            raise ValueError("Model returned a non-text answer")
        return extract_code(content), content
