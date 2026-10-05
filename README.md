# TRAIL: Embedding-Based Scheduling for LLM Serving

Code for the ICLR 2025 paper **"Don't Stop Me Now: Embedding Based Scheduling for LLMs"** by Rana Shahout, Eran Malach, Chunwei Liu, Weifan Jiang, Minlan Yu and Michael Mitzenmacher.

Most LLM servers run requests first-come, first-served (FCFS), so short requests get stuck behind long ones. TRAIL schedules requests by their predicted remaining output length instead:

- **Length prediction from the model's own embeddings.** A small MLP (about 2.1M parameters) reads the layer-11 hidden state of the serving model at each decode step and predicts how many tokens are left. Bayesian smoothing combines each step's prediction with the previous ones. Before a request runs, a DistilBERT classifier gives it an initial estimate from the prompt text.
- **SPRPT with limited preemption.** The scheduler runs the request with the shortest predicted remaining time first. A running request can be preempted only during its first `c × (initial predicted length)` steps, while its KV cache is still small.

This repository implements TRAIL on top of [SGLang](https://github.com/sgl-project/sglang) as a patch, and includes the trained predictors.

## Setup

```bash
git lfs install
git clone https://github.com/harvard-cns/TRAIL-llm-scheduling.git
cd TRAIL-llm-scheduling
bash setup.sh
source .venv/bin/activate
```

`setup.sh` installs [uv](https://docs.astral.sh/uv/) if needed, creates a Python 3.12 environment, and installs the patched SGLang in editable mode.

## Quick start

Start a TRAIL server on port 30000:

```bash
scripts/launch_server.sh trail
```

and send it requests like any SGLang server, for example:

```bash
curl http://127.0.0.1:30000/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model": "meta-llama/Meta-Llama-3-8B-Instruct",
       "messages": [{"role": "user", "content": "Write a haiku about GPUs."}]}'
```

The scheduling only matters under load, when requests queue. The benchmark below shows the effect.

## Reproducing the latency comparison

Run one policy at a time on the same GPU. For each policy, start the server in one terminal and run the benchmark in another:

```bash
# terminal 1
scripts/launch_server.sh fcfs --max-running-requests 48

# terminal 2
python scripts/benchmark.py --label fcfs --rates 10,20,30 --out results.jsonl
```

Stop the server (Ctrl-C), then repeat with `trail` and `trail_plus`:

```bash
scripts/launch_server.sh trail --max-running-requests 48
python scripts/benchmark.py --label trail --rates 10,20,30 --out results.jsonl

scripts/launch_server.sh trail_plus --max-running-requests 48
python scripts/benchmark.py --label trail_plus --rates 10,20,30 --out results.jsonl
```

## Policies and settings

`scripts/launch_server.sh {fcfs|trail|trail_plus} [extra sglang args]`. Arguments after the policy are passed to `sglang.launch_server`.

| Policy | What it does |
|---|---|
| `fcfs` | SGLang's default first-come, first-served scheduler |
| `trail` | TRAIL with the released predictors |
| `trail_plus` | TRAIL+ in the paper: the same scheduler, given each request's exact remaining length. The true length comes from `max_tokens`, so requests must set `ignore_eos: true` and `max_tokens` to the real output length (`benchmark.py` does this) |

Environment variables read by `launch_server.sh`:

| Variable | Default | Meaning |
|---|---|---|
| `MODEL` | `meta-llama/Meta-Llama-3-8B-Instruct` | Model to serve. The predictors were trained on this model |
| `PORT` | `30000` | Server port |
| `TRAIL_C` | `0` | Limited-preemption threshold `c`. `0` turns preemption off, so TRAIL only decides which waiting request runs next. See the note below |
| `PREDICT_EVERY` | `8` | Run the embedding MLP every K decode steps. Between runs the estimate counts down by one per token. `1` refines every step at a higher cost |
| `ORACLE_MAE` | `0` | `trail_plus` only: add Gaussian noise with this mean absolute error (tokens) to the oracle |

These map to the patch's own variables, which you can also set directly when calling `python -m sglang.launch_server --schedule-policy trail`: `SGLANG_TRAIL_INITIAL_PREDICTOR_DIR`, `SGLANG_TRAIL_PREDICTOR_DIR`, `SGLANG_TRAIL_PREEMPTION_LIMIT` (default `0.8` when set directly), `SGLANG_TRAIL_PREDICT_EVERY`, `SGLANG_TRAIL_ORACLE`, `SGLANG_TRAIL_ORACLE_MAE`. Two more knobs apply when preemption is on: `SGLANG_TRAIL_PREEMPTION_MARGIN` (preempt only if the waiting request is shorter by at least this many tokens) and `SGLANG_TRAIL_MAX_PREEMPTIONS_PER_REQ`.

## Predictors

Both predictors classify output length into 10 equal-width bins over 0–512 tokens, and the scheduler uses the expected value. They were trained on Llama-3-8B-Instruct outputs for the Alpaca dataset, with 10% of requests held out for validation (split by request).

| Predictor | Input | Validation MAE |
|---|---|---|
| `initial_distilbert` | Prompt text | 46.1 tokens (total length) |
| `mlp_layer11` | Layer-11 hidden state at a decode step | 33.9 tokens (remaining length) |

Training histories are in each directory (`training_history.json`). The MLP is `Linear(4096, 512) → ReLU → Linear(512, 10)`; `config.json` describes it.

## Citation

```bibtex
@inproceedings{shahout2025trail,
  title     = {Don't Stop Me Now: Embedding Based Scheduling for {LLMs}},
  author    = {Shahout, Rana and Malach, Eran and Liu, Chunwei and Jiang, Weifan and
               Yu, Minlan and Mitzenmacher, Michael},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2025}
}
```

## License

Apache 2.0 (see `LICENSE`). The Alpaca prompts in `data/` come from [tatsu-lab/alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca) (CC BY-NC 4.0).
