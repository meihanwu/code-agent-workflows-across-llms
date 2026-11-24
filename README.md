# code-agent-workflows-across-llms

Comparing coding agent workflows across locally deployed LLMs on code generation benchmarks.

## 1. Install the environment

Run the client and judges on Linux x86-64 with Conda and a local Docker Engine. Run the following commands from the repository root. The judges run on CPU.

```bash
conda env create -f environment.yml
conda activate code-agent-eval
python -m pip check
```

Pull the official vLLM 0.25.1 image used for evaluation. The judges mount the active Conda environment and the sources under `judges/` into the container. Docker must be able to bind-mount the repository and environment directories from the same host.

```bash
docker pull --platform linux/amd64 vllm/vllm-openai:v0.25.1
```

## 2. Models tested

| Model family | Models used in experiments |
| --- | --- |
| GPT-OSS | `gpt-oss-20b`, `gpt-oss-120b` |
| Qwen | `Qwen2.5-Coder-32B-Instruct`, `Qwen3-32B`, `Qwen3.8-27B` |
| Gemma | `gemma-3-27b-it`, `gemma-4-31b-it` |
| GLM | `GLM-4-32B-0414` |

We served these models locally with vLLM in Docker containers.

## 3. Prepare data

| Dataset | Download | Processing | Tasks | JSONL path under `data/` |
| --- | --- | --- | ---: | --- |
| `humaneval_plus` | [HumanEval+ v0.1.10][he] · [adapted prompts and public tests][adapted] | Decompress the release. Join by `task_id`, using the adapted prompts and public tests alongside the official evaluation fields. The experiment includes prompt corrections for tasks 32, 47, 114, 115, 149 and 158. | 164 | `HumanEvalPlus/humanevalplus.jsonl` |
| `mbpp_plus` | [MBPP+ v0.2.0 NoExtreme][mbpp] | Decompress the JSONL file and keep every record. | 378 | `MBPPEval/MBPPPlus-NoExtreme.jsonl` |
| `livecodebench` | [LiveCodeBench-lite test split][lcb] | Select `medium` records and exclude question `abc311_c`. | 167 | `livecodebench/LiveCodeBench-lite-medium.jsonl` |
| `livecodebench_hard` | [LiveCodeBench-lite test split][lcb] | Select all `hard` records. | 90 | `livecodebench/LiveCodeBench-lite-hard.jsonl` |
| `apps-150` | [APPS test split][apps] | The experiment uses a fixed selection of 50 standard-input tasks per difficulty. Reproducing this selection requires the original task IDs. | 150 | `APPS/APPS-test-stratified150.jsonl` |
| `apps-introductory` | [APPS test split][apps] | Select `introductory` records. The experiment corrects an expected boolean in task 4751 and an integer input in 4752. | 1,000 | `APPS/APPS-test-introductory.jsonl` |
| `apps-interview` | [APPS test split][apps] | Select `interview` records. The experiment corrects extra quotes or list nesting in the test inputs or outputs of tasks 2468, 2745–2748, 2882 and 2884. | 3,000 | `APPS/APPS-test-interview.jsonl` |
| `apps-competition` | [APPS test split][apps] | Select all `competition` records and preserve their contents. | 1,000 | `APPS/APPS-test-competition.jsonl` |

[he]: https://github.com/evalplus/humanevalplus_release/releases/download/v0.1.10/HumanEvalPlus.jsonl.gz
[adapted]: https://raw.githubusercontent.com/XZC888/MavenCoder/main/data/humanevalplus.jsonl
[mbpp]: https://github.com/evalplus/mbppplus_release/releases/download/v0.2.0/MbppPlus-NoExtreme.jsonl.gz
[lcb]: https://huggingface.co/datasets/livecodebench/code_generation_lite/resolve/0fe84c3912ea0c4d4a78037083943e8f0c4dd505/test.jsonl
[apps]: https://huggingface.co/datasets/codeparrot/apps/resolve/21e74ddf8de1a21436da12e3e653065c5213e9d1/test.jsonl

## 4. Generate and evaluate one solution

```bash
python main.py --dataset mbpp_plus \
  --data-path data/MBPPEval/MBPPPlus-NoExtreme.jsonl \
  --model gpt-oss-20b --base-url http://127.0.0.1:18020/v1 \
  --reasoning-effort low --temperature 0 \
  --timeout 1200 --task-id Mbpp/2

python evaluate.py --dataset mbpp_plus \
  --results-file results/gpt-oss-20b/mbpp_plus.jsonl \
  --task-id Mbpp/2 --workers 4
```

| Output | Location |
| --- | --- |
| Extracted solutions | `results/gpt-oss-20b/mbpp_plus.jsonl` |
| Raw model responses | `results/gpt-oss-20b/responses/mbpp_plus.jsonl` |
| Per-task verdicts and summary | `evaluation/gpt-oss-20b/mbpp_plus/{verdicts.jsonl,metrics.json}` |

Omit `--task-id` from both commands for the full dataset. EvalPlus reuses matching reference outputs in `~/.cache/evalplus/` when available. On a cache miss, it computes reference outputs for the entire dataset inside the temporary container, even for a one-task run.
