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

### 3.1 Included in the repository

The MBPP+ example in section 4 uses the file included here.

| JSONL file | Tasks | Upstream source | Processing |
| --- | ---: | --- | --- |
| [HumanEval+](data/HumanEvalPlus/humanevalplus.jsonl) | 164 | [HumanEval+ v0.1.10][he] and [adapted prompts and public tests][adapted] | Join by `task_id`. Keep the official evaluation fields, adapted prompts and public tests, including prompt corrections for tasks 32, 47, 114, 115, 149 and 158. |
| [MBPP+ NoExtreme](data/MBPPEval/MBPPPlus-NoExtreme.jsonl) | 378 | [MBPP+ v0.2.0 NoExtreme][mbpp] | Decompress and retain all records. |
| [APPS-150](data/APPS/APPS-test-stratified150.jsonl) | 150 | [APPS test split][apps] | Select 50 standard-input tasks from each of the three difficulty levels. The file contains the exact experiment selection. |
| [LiveCodeBench-lite Easy](data/livecodebench/LiveCodeBench-lite-easy.jsonl) | 142 | [LiveCodeBench-lite test split][lcb] | Select all easy records. Supplementary split, outside the current experiment set. |

### 3.2 Download larger splits

The links point to the upstream test sets. Each experiment reads its corresponding difficulty split from the path shown below.

| Experiment split | Upstream download | Expected path under `data/` |
| --- | --- | --- |
| LiveCodeBench-lite Medium (167 tasks, excluding `abc311_c`) | [LiveCodeBench-lite][lcb] | `livecodebench/LiveCodeBench-lite-medium.jsonl` |
| LiveCodeBench-lite Hard (90 tasks) | [LiveCodeBench-lite][lcb] | `livecodebench/LiveCodeBench-lite-hard.jsonl` |
| APPS Introductory (1,000 tasks) | [APPS][apps] | `APPS/APPS-test-introductory.jsonl` |
| APPS Interview (3,000 tasks) | [APPS][apps] | `APPS/APPS-test-interview.jsonl` |
| APPS Competition (1,000 tasks) | [APPS][apps] | `APPS/APPS-test-competition.jsonl` |

The introductory and interview experiments used corrected test cases for APPS tasks 4751, 4752, 2468, 2745–2748, 2882 and 2884.

[he]: https://github.com/evalplus/humanevalplus_release/releases/download/v0.1.10/HumanEvalPlus.jsonl.gz
[adapted]: https://raw.githubusercontent.com/XZC888/MavenCoder/main/data/humanevalplus.jsonl
[mbpp]: https://github.com/evalplus/mbppplus_release/releases/download/v0.2.0/MbppPlus-NoExtreme.jsonl.gz
[lcb]: https://huggingface.co/datasets/livecodebench/code_generation_lite/resolve/0fe84c3912ea0c4d4a78037083943e8f0c4dd505/test.jsonl
[apps]: https://huggingface.co/datasets/codeparrot/apps/resolve/21e74ddf8de1a21436da12e3e653065c5213e9d1/test.jsonl

## 4. Generate and evaluate code

### 4.1 Generate solutions with GPT-OSS-20B

Generate one solution for each of the 378 MBPP+ tasks. From the repository root, run the command below with the Conda environment active. Set `--base-url` to your vLLM endpoint and `--model` to the name served there.

```bash
python main.py --dataset mbpp_plus \
  --data-path data/MBPPEval/MBPPPlus-NoExtreme.jsonl \
  --model gpt-oss-20b --base-url http://127.0.0.1:18020/v1 \
  --reasoning-effort low --temperature 0 \
  --timeout 1200
```

The example uses low reasoning effort, a sampling temperature of zero, and a request timeout of 1,200 seconds. If generation is interrupted, rerun the same command to resume from the saved results.

### 4.2 Evaluate the saved solutions

Once generation finishes, pass the complete JSONL file to `--results-file`. EvalPlus runs each solution against the base tests, then runs the additional tests if the base tests pass.

```bash
python evaluate.py --dataset mbpp_plus \
  --results-file results/gpt-oss-20b/mbpp_plus.jsonl \
  --workers 4
```

`--workers 4` allows up to four tasks to be judged concurrently. In the printed summary, `base_passed` counts solutions that passed the base tests, `passed` counts those that passed both suites, and `pass@1` is the fraction of evaluated tasks that passed both suites.

For a single task, add the optional `--task-id Mbpp/2` to either command. During generation it selects the prompt to send to the model. During evaluation it selects the saved solution to judge.
