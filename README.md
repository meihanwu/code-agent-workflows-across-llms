# code-agent-workflows-across-llms

Comparing coding agent workflows across locally deployed LLMs on code generation benchmarks.

## 1. Install the environment

Run the client and judges on Linux x86-64 with Conda and a local Docker Engine. Run the following commands from the repository root. The judges run on CPU. Self-hosting the model in section 2 also requires NVIDIA GPUs and the NVIDIA Container Toolkit.

```bash
conda env create -f environment.yml
conda activate code-agent-eval
python -m pip check
python -m unittest discover -s tests
```

Pull the exact image used for evaluation. The judges mount the active Conda environment and the sources under `judges/` into the container. Docker must be able to bind-mount the repository and environment directories from the same host.

```bash
docker pull --platform linux/amd64 vllm/vllm-openai@sha256:e4f88a835143cd22aee2397a26ec6bb80b3a4a6fe0c882bcbc63822904766089
mkdir -p ~/.cache/evalplus
```

## 2. Connect a model

If a model is already running, use its served model name and OpenAI-compatible `/v1` URL in section 4. Set `VLLM_API_KEY` if the endpoint requires authentication.

To serve GPT-OSS-120B locally, run the following in a separate terminal with four suitable GPUs. Set `MODEL_DIR` to the directory containing the downloaded weights. The server listens on localhost port 18023. [Model weights](https://huggingface.co/openai/gpt-oss-120b) · [vLLM Docker guide](https://docs.vllm.ai/en/stable/deployment/docker/)

```bash
export MODEL_DIR=/absolute/path/to/gpt-oss-120b
docker run --rm --gpus '"device=0,1,2,3"' --ipc=host \
  -p 127.0.0.1:18023:8000 \
  -v "${MODEL_DIR}:/model:ro" \
  vllm/vllm-openai@sha256:e4f88a835143cd22aee2397a26ec6bb80b3a4a6fe0c882bcbc63822904766089 \
  --model /model --served-model-name gpt-oss-120b \
  --tensor-parallel-size 4 --max-model-len 32768 \
  --gpu-memory-utilization 0.90 --max-num-seqs 8 --enable-prefix-caching
```

## 3. Prepare data

For the one-task example, download and decompress the official MBPP+ NoExtreme release. Keep all 378 records in the file. The evaluator checks the full dataset and then selects the requested task.

```bash
mkdir -p data/MBPPEval
curl -fL --retry 3 -o data/MBPPEval/MbppPlus-NoExtreme.jsonl.gz \
  https://github.com/evalplus/mbppplus_release/releases/download/v0.2.0/MbppPlus-NoExtreme.jsonl.gz
gzip -dc data/MBPPEval/MbppPlus-NoExtreme.jsonl.gz > data/MBPPEval/MBPPPlus-NoExtreme.jsonl
```

| Dataset argument | JSONL path under `data/` | Tasks | Source and experiment split |
| --- | --- | ---: | --- |
| `humaneval_plus` | `HumanEvalPlus/humanevalplus.jsonl` | 164 | [HumanEval+ v0.1.10](https://github.com/evalplus/humanevalplus_release/releases/tag/v0.1.10). The experiments used corrected prompts and public tests from an [adapted dataset](https://github.com/XZC888/MavenCoder/blob/main/data/humanevalplus.jsonl), joined by task ID. |
| `mbpp_plus` | `MBPPEval/MBPPPlus-NoExtreme.jsonl` | 378 | [MBPP+ v0.2.0](https://github.com/evalplus/mbppplus_release/releases/tag/v0.2.0), NoExtreme. |
| `livecodebench` | `livecodebench/LiveCodeBench-lite-medium.jsonl` | 167 | Medium records from the [pinned LiveCodeBench-lite split](https://huggingface.co/datasets/livecodebench/code_generation_lite/blob/0fe84c3912ea0c4d4a78037083943e8f0c4dd505/test.jsonl), excluding `abc311_c`. |
| `livecodebench_hard` | `livecodebench/LiveCodeBench-lite-hard.jsonl` | 90 | All hard records from the same split. |
| `apps-150` | `APPS/APPS-test-stratified150.jsonl` | 150 | Fixed selection of 50 standard-input tasks per difficulty from the [pinned APPS test set](https://huggingface.co/datasets/codeparrot/apps/blob/21e74ddf8de1a21436da12e3e653065c5213e9d1/test.jsonl). Exact reproduction requires the experiment subset. |
| `apps-introductory` | `APPS/APPS-test-introductory.jsonl` | 1,000 | Filter APPS by `introductory`. The experiment copy has two corrected test records. |
| `apps-interview` | `APPS/APPS-test-interview.jsonl` | 3,000 | Filter APPS by `interview`. The experiment copy has seven corrected test records. |
| `apps-competition` | `APPS/APPS-test-competition.jsonl` | 1,000 | Filter APPS by `competition`. |

## 4. Generate and evaluate one solution

```bash
python main.py --dataset mbpp_plus \
  --data-path data/MBPPEval/MBPPPlus-NoExtreme.jsonl \
  --model gpt-oss-120b --base-url http://127.0.0.1:18023/v1 \
  --reasoning-effort low --temperature 0 --max-tokens 8192 \
  --timeout 1200 --task-id Mbpp/2

python evaluate.py --dataset mbpp_plus \
  --results-file results/gpt-oss-120b/mbpp_plus.jsonl \
  --task-id Mbpp/2 --workers 4
```

| Output | Location |
| --- | --- |
| Extracted solutions | `results/gpt-oss-120b/mbpp_plus.jsonl` |
| Raw model responses | `results/gpt-oss-120b/responses/mbpp_plus.jsonl` |
| Per-task verdicts and summary | `evaluation/gpt-oss-120b/mbpp_plus/{verdicts.jsonl,metrics.json}` |

Omit `--task-id` from both commands for the full dataset. EvalPlus reuses matching reference outputs in `~/.cache/evalplus/` when available. On a cache miss, it computes reference outputs for the entire dataset inside the temporary container, even for a one-task run.
