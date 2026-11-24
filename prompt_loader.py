"""Load code-generation prompts saved next to this module."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10: use the reference project's TOML reader.
    import toml as tomllib

from jinja2 import Environment, StrictUndefined


ROOT = Path(__file__).resolve().parent
PROMPT_FILE = ROOT / "prompts.toml"
TEMPLATES = tomllib.loads(PROMPT_FILE.read_text(encoding="utf-8"))
DEFAULT_PROFILES = {'humaneval_plus': 'prompt_codegen_chat_api', 'mbpp_plus': 'prompt_codegen_chat_api', 'livecodebench': 'prompt_codegen_generic', 'apps': 'prompt_codegen_chat_api'}
TEMPLATES["livecodebench_hard"] = TEMPLATES["livecodebench"]
DEFAULT_PROFILES["livecodebench_hard"] = DEFAULT_PROFILES["livecodebench"]
PROFILE_KINDS = {'prompt_codegen_chat_api': 'chat', 'prompt_codegen_local_chat': 'evalplus_prefill', 'prompt_codegen_completion': 'completion', 'prompt_codegen_generic': 'chat', 'prompt_codegen_user_only': 'chat', 'prompt_codegen_openai_reason': 'chat', 'prompt_codegen_gemini': 'completion', 'prompt_codegen_gemini_thinking': 'completion', 'prompt_codegen_deepseek_coder': 'completion', 'prompt_codegen_codeqwen': 'completion', 'prompt_codegen_qwq': 'completion', 'prompt_codegen_deepseek_r1': 'completion', 'prompt_codegen_base': 'completion'}
PREFILL_SPLITTER = "-[[]]-this-is-really-our-highest-priority-[[]]-"
ENVIRONMENT = Environment(
    undefined=StrictUndefined, autoescape=False, keep_trailing_newline=True
)


@dataclass(frozen=True)
class LoadedPrompt:
    kind: str
    system: str
    user: str
    assistant_prefix: str
    raw_prompt: str
    splitter: str
    profile: str

    def to_messages(self):
        """Return API messages; an absent system is omitted, not sent empty."""
        if self.kind != "chat":
            raise ValueError(
                f"{self.profile} uses {self.kind}; call to_text() instead."
            )
        messages = []
        if self.system:
            messages.append({"role": "system", "content": self.system})
        messages.append({"role": "user", "content": self.user})
        return messages

    def to_text(self, tokenizer=None):
        """Render completion input or tokenize chat with the supplied tokenizer."""
        if self.kind == "completion":
            return self.user
        if tokenizer is None:
            raise ValueError(f"{self.profile} requires a tokenizer for to_text().")
        if self.kind == "evalplus_prefill":
            # Preserve the upstream no-chat-template branch and assistant truncation.
            if tokenizer.chat_template is None:
                return self.raw_prompt
            messages = [
                {"role": "user", "content": self.user},
                {
                    "role": "assistant",
                    "content": self.assistant_prefix + self.splitter + "\n```\n",
                },
            ]
            return tokenizer.apply_chat_template(
                messages, tokenize=False
            ).split(self.splitter)[0]
        return tokenizer.apply_chat_template(
            self.to_messages(), tokenize=False, add_generation_prompt=True
        )


def list_profiles(dataset: str):
    """Return available TOML section names for the selected dataset."""
    config = TEMPLATES[dataset]
    return list(config)


def load_prompt(dataset: str, task: Dict[str, Any], profile: Optional[str] = None):
    """Render a dataset template without adding tests."""
    config = TEMPLATES[dataset]
    selected = profile or DEFAULT_PROFILES[dataset]
    template = config[selected]

    if dataset in ("humaneval_plus", "mbpp_plus"):
        context = {"task_prompt": task["prompt"].strip()}
    elif dataset in ("livecodebench", "livecodebench_hard"):
        context = {
            "question_content": task["question_content"],
            "starter_code": task["starter_code"],
        }
    else:
        input_output = task["input_output"]
        if isinstance(input_output, str):
            input_output = json.loads(input_output)
        context = {
            "question": task["question"],
            "starter_code": task["starter_code"],
            "fn_name": input_output.get("fn_name"),
        }

    def render(key):
        return ENVIRONMENT.from_string(template.get(key, "")).render(context)

    return LoadedPrompt(
        kind=PROFILE_KINDS[selected],
        system=render("system"),
        user=render("user"),
        assistant_prefix=render("assistant_prefix"),
        raw_prompt=(context["task_prompt"] + "\n" if PROFILE_KINDS[selected] == "evalplus_prefill" else ""),
        splitter=(PREFILL_SPLITTER if PROFILE_KINDS[selected] == "evalplus_prefill" else ""),
        profile=selected,
    )
