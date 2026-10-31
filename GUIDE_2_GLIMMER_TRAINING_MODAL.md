# Guide 2: Train the Glimmer decision model on Modal

Owner: Brooklyn. Executor: Codex. Source of truth for everything this guide does not restate: the Brier build plan (section "Models and training"), `AGENTS.md` and `SCOPE.md`. Where this guide is more specific than the plan, this guide wins, and Brooklyn authorizes every file, flag and dependency it names.

Input: `compiled.jsonl` from Guide 1 (`gs://$BRIER_BUCKET/release/v1/compiled.jsonl`).

When you finish this guide, these exist and pass their checks:

1. Muse Glimmer 30B weights in a Modal Volume, pinned to one Hugging Face commit.
2. Zero-shot option probabilities from base Glimmer for the calibration, test and OOD splits (the "Glimmer base, zero-shot" ablation row).
3. Last-token embeddings from base Glimmer for every example, and a trained frozen-head MLP with its probabilities (the "Glimmer frozen head" row and the safety net).
4. A LoRA adapter fine-tuned with the proper-scoring loss (cross-entropy plus Brier, ranked probability score for the bucket question), with checkpoints and a resume path.
5. Fine-tuned probabilities for calibration, test and OOD, before and after temperature scaling (the "Glimmer fine-tuned" and "Fine-tuned + temperature" rows).
6. A metrics file per model (ECE, Brier, NLL, AUROC and the rest) that the eval harness (T13) turns into `report.json`.
7. The adapter published on the Hugging Face Hub with a model card.

Serving the adapter behind `POST /decide` (T20) is a separate card. Section 10 gives the contract it must follow so that serving reproduces training exactly.

---

## 0. How to run this guide

- Work through the task cards in section 6 in order, one Codex session per card. Run the runbook in section 7 after the cards it needs have landed and passed `make check`.
- If a target file already exists from an earlier session (T14, T15, T16, T17), keep it and close the gaps against this guide inside the card's files and line budget. List every gap you fixed.
- Stop and report instead of improvising when: a credential is missing, a gate in section 8 fails twice, Modal spend would pass the cap in section 3, or the installed `transformers` behaves differently from what section 2 describes and the source does not make the fix obvious.
- No fake numbers. Every probability in the outputs comes from a real forward pass of Glimmer on the real compiled prompts.

### Card to plan mapping (use the plan id in commit messages)

| This guide | Plan task |
| --- | --- |
| M1 Modal skeleton and weights, M2 data, M3 model | T14 and T15 (shared) |
| M4 losses, M5 training loop, M7 Modal functions | T15 |
| M6 base pass and scoring | T14 (zero-shot row) and T16 (embeddings) |
| M8 frozen head | T16 |
| M9 temperature scaling and metrics | T17 |

Commit as, for example, `T15: finetune loop (guide 2, M5)`.

### Run Codex where the network works

The runbook uses the Modal CLI, `gcloud storage` and the Hugging Face Hub. Run Codex locally with network access. All GPU work runs on Modal.

### Deliberate changes from the plan, and why

| Plan says | This guide does | Reason |
| --- | --- | --- |
| QLoRA 4-bit NF4 on A100-80GB by default | BF16 LoRA on one H100 80GB by default; QLoRA NF4 on A100-80GB is the fallback | The text decoder in BF16 is about 56 GB and fits on one 80 GB card with gradient checkpointing (Hugging Face's own Glimmer post lists LoRA SFT in BF16 on 1 x 80 GB H100). BF16 avoids dequantization overhead and bitsandbytes support risk on a brand-new architecture, and is cheaper per token on Modal. The smoke test decides with measured numbers |
| Zero-shot row via the vLLM endpoint (T14) | Zero-shot scored with the same HF forward pass used for training | Identical tokenization by construction, no serving stack needed to fill the row. The vLLM endpoint still gets built in T14/T20 and must match these numbers (section 10) |
| Checkpoint every 200 steps | Every 50 steps | One epoch is only a few hundred optimizer steps at this dataset size |
| Sequence 1,536 tokens | Max length is measured; compiled prompts are a few hundred tokens. Hard limit stays 1,536 | Nothing is truncated; a longer prompt is an error |

Everything else follows the plan: custom loop in plain PyTorch plus PEFT, no Trainer subclasses, no Unsloth, LoRA rank 16, alpha 32, dropout 0.05 on attention and MLP projections, AdamW lr 1e-4 cosine with 3% warmup, effective batch 32, one epoch (a second only if calibration NLL still falls), loss = cross-entropy plus a Brier term (RPS for score questions), temperature per question type fitted on the calibration split only.

---

## 1. Credentials Brooklyn provides

| Name | What it is | Used by |
| --- | --- | --- |
| `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET` | Modal API token for Brooklyn's workspace (or `modal token set` done once on the machine) | Modal CLI |
| Modal secret `huggingface` | created once with `modal secret create huggingface HF_TOKEN=<token>` | weight download, adapter publish |
| `HF_TOKEN` | Hugging Face token with write scope (for the adapter publish at the end) | local publish step |
| `HF_USER` | Hugging Face user or org that will own `brier-glimmer-lora` | publish |
| `BRIER_BUCKET`, `GOOGLE_APPLICATION_CREDENTIALS` | from Guide 1 | fetching `compiled.jsonl` |

Before any GPU call, Brooklyn confirms in the Modal dashboard that the workspace budget alert is set at $25 (plan rule). Codex asks for this confirmation if it has not been stated.

Env vars allowed in addition to the `AGENTS.md` list: `HF_TOKEN`, `HF_USER`, `BRIER_BUCKET`, `GOOGLE_APPLICATION_CREDENTIALS`, `HF_HOME`, `TOKENIZERS_PARALLELISM`, `PYTORCH_CUDA_ALLOC_CONF`.

---

## 2. Facts about Muse Glimmer this guide relies on

From the Hugging Face model card, config and launch posts (verify each in card M1 against the downloaded files, fail loudly if any is false):

- Repo: `meta-models/Muse-Glimmer-30B`, Apache 2.0, not gated.
- Architecture `MuseGlimmerForConditionalGeneration` (`model_type: muse_glimmer`), text config `model_type: muse_glimmer_text`. About 29.6B parameters in total, of which about 1.8B are the perception encoder (ViT-G/14). Text decoder: 52 layers, hidden size 6,656, 32 query heads and 2 KV heads (head dim 128), SwiGLU MLP (intermediate 19,968), attention pattern of 3 local (sliding window 2,048) layers then 1 global, gated attention, vocabulary 202,048.
- Full BF16 weights are about 58 GB on disk. Transformers support shipped in `transformers` 5.15. The official loading example uses `AutoProcessor` and `AutoModelForMultimodalLM` with `dtype="auto"`.
- vLLM serves it natively from 0.28 (or with `--model-impl transformers`).

This guide uses the model as a **text-only** scorer: input ids and an attention mask, no images. It never uses the chat template (section 2.1).

### 2.1 Prompt and answer position

`brier compile` writes the prompt; Python only tokenizes it. Training, scoring and serving all feed the **raw compiled prompt string** with the tokenizer's default special tokens (`add_special_tokens=True`), no chat template, no reasoning mode. The prompt ends with `<answer>` and nothing after it. The model's next token is the answer.

Option tokens: for each letter L in `ABCDEF`, the answer token is the single token of `" " + L` (space then letter). Card M2 proves for every example that:

```
tokenize(prompt + " " + L) == tokenize(prompt) + [token_id(" " + L)]
```

If any letter is not a single token, or the boundary property fails for any example, stop and report: the fix is a template change in `brier compile`, which is Brooklyn's call.

Option probabilities are a softmax over the logits of the K valid letters at the last position (K = 2 for `regressed`, 6 for `failure_mode`, 5 for `success_bucket`). Never over the whole vocabulary.

---

## 3. Budget

Modal pool: $30 (plan ceiling), alert at $25. Rates used for planning (Modal pricing page, per second): H100 $0.001097 (about $3.95 per hour), A100-80GB $0.000694 (about $2.50 per hour).

| Step | GPU | Planned | Hard cap |
| --- | --- | --- | --- |
| Weight download to the Volume | CPU | < $0.50 | $1 |
| M-check (load, tokenizer, memory) | H100 | ~$0.50 | $1 |
| Base pass (zero-shot + embeddings, all splits) | H100 | ~$1.50 | $3 |
| Smoke test (50 steps) | H100 | ~$1.50 | $3 |
| Full fine-tune (1 epoch) | H100 | ~$4 to $8 | $12 |
| Optional second epoch | H100 | ~$4 | only if under the total cap |
| Scoring fine-tuned (cal, test, OOD) | H100 | ~$1 | $2 |
| **Total** | | **~$10 to $15** | **$22** |

Every GPU function takes `--max-usd` and stops itself (after saving a checkpoint) when its own elapsed GPU seconds times the rate reach it. Every function has a Modal `timeout`. The full fine-tune runs only if the smoke test's projected epoch cost is under its cap.

---

## 4. Code layout (inside existing directories)

```
py/brier_ml/
  glimmer_data.py     # read compiled.jsonl, tokenize, letter ids, boundary checks, batching
  glimmer_model.py    # load base (BF16 or NF4), LoRA wrap, last-position option logits, embeddings
  losses.py           # proper-scoring losses (CE + Brier, CE + RPS), metrics helpers
  finetune.py         # training loop: smoke and full, checkpoints, resume, budget guard
  score.py            # base pass (zero-shot probs + embeddings) and adapter scoring
  modal_train.py      # Modal app: image, volumes, functions, local entrypoint
  frozen_head.py      # MLP head on embeddings (CPU)
  calibrate.py        # temperature scaling per question type + metrics JSON
  tests/              # CPU tests with a tiny random model
```

No new top-level files or directories. Python 3.11 locally and in the Modal image.

Dependencies allowed for this guide: `modal` (local CLI and in-image), `torch`, `transformers>=5.15`, `peft`, `accelerate`, `safetensors`, `huggingface_hub`, `bitsandbytes` (fallback path only), `numpy`, `scipy`, `scikit-learn` (AUROC only), `pytest`. Pin exact versions in card M1 (resolve the newest mutually compatible set on the day, then freeze them in `modal_train.py` and `py/pyproject.toml`).

---

## 5. Data contract (from Guide 1)

Each line of `compiled.jsonl`:

```json
{"id": "c042:aloha-s07:regressed:ev", "pair_id": "c042:aloha-s07", "change_id": "c042", "scenario_id": "aloha-s07",
 "env": "aloha", "variant": "ev", "split": "train", "question": "regressed",
 "options": ["no", "yes"], "prompt": "...<answer>", "label": "B"}
```

- `split` in `train`, `calibration`, `test`, `ood`. Train on `train` only. Fit temperatures and the frozen head's early stopping on `calibration` only. Never fit anything on `test` or `ood`.
- `question` in `regressed` (K=2), `failure_mode` (K=6), `success_bucket` (K=5).
- `options` are the option texts in displayed order. Position i has letter `"ABCDEF"[i]`.
- For `success_bucket` the ordinal order is the canonical order `0-19%`, `20-39%`, `40-59%`, `60-79%`, `80-100%`. The RPS loss and metric map displayed positions back to this order through the option texts.
- For `regressed` metrics, `p_yes` is the probability of the letter whose option text is `yes`.

---

## 6. Task cards

### M1: Modal app skeleton, pinned image, weight download

```
TASK M1: modal_train.py skeleton + weights
Goal: Modal app with a pinned image, two volumes, and a CPU function that downloads Glimmer at a pinned revision.
Files (only these): py/brier_ml/modal_train.py, py/pyproject.toml (deps only)
Dependencies allowed: modal, huggingface_hub (+ the pins listed in section 4)
Line budget: 140
```

Spec:

```python
import modal

APP = modal.App("brier-train")
MODELS = modal.Volume.from_name("brier-models", create_if_missing=True)   # /models
DATA = modal.Volume.from_name("brier-data", create_if_missing=True)       # /data
HF = modal.Secret.from_name("huggingface")
BASE_REPO = "meta-models/Muse-Glimmer-30B"
BASE_DIR = "/models/base/muse-glimmer-30b"          # weights live here, revision recorded in revision.txt

IMAGE = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(  # exact pins resolved in this card, for example "torch==X.Y.Z", "transformers==5.N.M", ...
        "torch==<pin>", "transformers==<pin>", "peft==<pin>", "accelerate==<pin>", "safetensors==<pin>",
        "huggingface_hub==<pin>", "bitsandbytes==<pin>", "numpy==<pin>", "scipy==<pin>", "scikit-learn==<pin>",
    )
    .env({"HF_HOME": "/models/hf-home", "TOKENIZERS_PARALLELISM": "false",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .add_local_python_source("brier_ml")
)
```

If the installed `modal` client has no `Image.uv_pip_install`, use `Image.pip_install` with the same pins. Run every `modal` command with `PYTHONPATH=py` so `add_local_python_source("brier_ml")` finds the package.

- `download_weights()` runs on CPU (`cpu=4`, `memory=16384`, `timeout=3600`, secret `HF`, volume `MODELS`). It resolves `huggingface_hub.model_info(BASE_REPO).sha` once, writes it to `BASE_DIR/revision.txt`, and calls `snapshot_download(BASE_REPO, revision=sha, local_dir=BASE_DIR, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja", "tokenizer*"])`. Then `MODELS.commit()`. If `revision.txt` exists, it only verifies that the files are complete (every shard named in `model.safetensors.index.json` exists) and exits.
- Every later function loads with `revision` read from `revision.txt` and `local_files_only=True`.
- Run `python -c "import modal"` and `modal --version` locally; the CLI version goes into `run_meta.json` later.

Acceptance:

```bash
PYTHONPATH=py modal run py/brier_ml/modal_train.py::download_weights
modal volume ls brier-models base/muse-glimmer-30b
```

The listing shows the config, tokenizer files and every safetensors shard; `revision.txt` holds a 40-character sha.

### M2: Data loading, tokenization checks, batching

```
TASK M2: glimmer_data
Goal: Load compiled.jsonl, tokenize prompts exactly once, prove the answer-token boundary, build length-grouped batches.
Files (only these): py/brier_ml/glimmer_data.py, py/brier_ml/tests/test_glimmer_data.py
Dependencies allowed: transformers, torch, numpy
Line budget: 220
```

Spec:

- `load_examples(path, splits) -> list[dict]` validates every field in section 5 and fails on any missing or unknown value.
- `letter_token_ids(tokenizer) -> list[int]` for `" A"` to `" F"`. Assert each encodes to exactly one token with `add_special_tokens=False`.
- `tokenize(examples, tokenizer, max_len=1536)`: `tokenizer(prompt, add_special_tokens=True)`. Assert every prompt ends with `<answer>`. Assert the boundary property of section 2.1 for **every** example and every valid letter. Assert length <= `max_len`. Returns input ids, the label index (0-based letter position), K, and for `success_bucket` a `pos_to_ordinal` array mapping displayed position to ordinal index.
- `collate(batch, pad_id)`: **left padding**. Returns `input_ids`, `attention_mask`, `position_ids = (attention_mask.cumsum(-1) - 1).clamp(min=0)`, `label_idx`, `k`, `ordinal_map` (padded with -1), and a `valid` mask of shape [B, 6] (True for j < K).
- `length_grouped_order(lengths, micro_batch, seed)`: shuffle indices with the seed, cut into chunks of `micro_batch * 64`, sort each chunk by length, cut into micro-batches, shuffle the micro-batch order with the seed. Deterministic for a given seed.
- `pad_id`: the tokenizer's pad id if it has one, else its EOS id (masked out anyway). Record which.
- Tests use a small local tokenizer fixture (any small BPE tokenizer saved under `py/brier_ml/tests/fixtures/`) to test the logic. The real Glimmer tokenizer is checked in the runbook step N3 on Modal.

Acceptance: `pytest -q py/brier_ml/tests/test_glimmer_data.py`.

### M3: Model loading, LoRA, last-position option logits

```
TASK M3: glimmer_model
Goal: Load Glimmer text-only in BF16 (or NF4 fallback), attach LoRA to the text decoder's attention and MLP projections, and compute option logits and embeddings at the last position only.
Files (only these): py/brier_ml/glimmer_model.py, py/brier_ml/tests/test_glimmer_model.py
Dependencies allowed: torch, transformers, peft, bitsandbytes
Line budget: 230
```

Spec:

`load_base(path, revision, quant: "bf16" | "nf4") -> (model, tokenizer, info)`:

1. Try `AutoModelForCausalLM.from_pretrained(path, revision=revision, local_files_only=True, dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa")`. Accept it only if it loaded no perception-encoder parameters (no parameter name containing `vision`, `visual` or `perception`) and the parameter count is between 27.0B and 28.6B.
2. Otherwise load `AutoModelForMultimodalLM` the same way and use it whole. Only the text decoder receives LoRA; the perception encoder stays frozen and unused (text-only inputs).
3. `nf4` adds `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16, llm_int8_skip_modules=[<lm_head name>, <vision module names>])` and calls `peft.prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)`.
4. `info` records the class used, parameter counts, dtype, quantization, `transformers` version, whether `forward` accepts `logits_to_keep`, and the final-norm location used for embeddings.

`add_lora(model) -> model`:

```python
skip = ("vision", "visual", "perception", "projector", "multi_modal", "lm_head", "embed")
leaf_names = sorted({n.rsplit(".", 1)[-1] for n, m in model.named_modules()
                     if isinstance(m, torch.nn.Linear) and not any(s in n for s in skip)})
# expect attention projections (q/k/v/o, plus any attention gate projection) and MLP projections (gate/up/down)
# fail loudly if no attention-like or no MLP-like names are found; log the final list
target = rf".*({'|'.join(map(re.escape, leaf_names))})$" if <text-only CausalLM> \
    else rf".*language_model.*\.({'|'.join(map(re.escape, leaf_names))})$"
cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM", target_modules=target)
model = get_peft_model(model, cfg)
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.enable_input_require_grads()
model.config.use_cache = False
```

Read the real module names from the loaded model and adjust the `language_model` prefix in the regex to whatever path the multimodal class uses. Log trainable and total parameter counts.

`last_logits(model, batch) -> Tensor[B, V]` (float32):

```python
out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
            position_ids=batch["position_ids"], use_cache=False, logits_to_keep=1)
return out.logits[:, -1, :].float()
```

If `forward` does not accept `logits_to_keep`, run the decoder only (`model.get_decoder()` or the multimodal class's language model), take the final hidden state at position -1, and apply the output embedding (`model.get_output_embeddings()`) to that one position. Never materialize logits for every position: at 202k vocabulary that is gigabytes per batch.

`option_logits(last_logits, letter_ids) -> Tensor[B, 6]` selects the 6 letter columns.

`last_hidden(model, batch) -> Tensor[B, 6656]`: the final-layer hidden state at position -1 after the final norm (`output_hidden_states=True`, take `hidden_states[-1][:, -1]`; check in the source whether that entry is pre- or post-norm and apply the model's final norm if it is pre-norm; record the choice in `info`).

Left padding plus explicit `position_ids` is required. Test it: on a tiny random model (`transformers` can build any small config locally, for example a 2-layer Llama-style config), the last-position logits of a sequence must be identical (to 1e-4) whether it is padded on the left inside a batch or run alone.

Acceptance: `pytest -q py/brier_ml/tests/test_glimmer_model.py` (tiny model on CPU: padding invariance, LoRA attaches only to the allowed names, `option_logits` shape).

### M4: Proper-scoring losses and metrics

```
TASK M4: losses
Goal: CE + Brier, CE + RPS, masked over the K valid options; metric helpers.
Files (only these): py/brier_ml/losses.py, py/brier_ml/tests/test_losses.py
Dependencies allowed: torch, numpy, scikit-learn
Line budget: 160
```

Spec:

```python
def option_probs(opt_logits, valid, temperature=1.0):        # [B, 6], [B, 6] bool
    z = (opt_logits / temperature).masked_fill(~valid, float("-inf"))
    return torch.softmax(z, dim=-1)                          # zeros on invalid positions

def proper_loss(opt_logits, valid, label_idx, question_is_score, ordinal_map, lam=1.0):
    p = option_probs(opt_logits, valid)
    ce = -torch.log(p.gather(1, label_idx[:, None]).squeeze(1).clamp_min(1e-12))
    onehot = F.one_hot(label_idx, 6).float() * valid
    brier = ((p - onehot) ** 2 * valid).sum(-1)
    # RPS for score questions: reorder p and onehot into ordinal order with ordinal_map, then
    # rps = sum_{i=1}^{K-1} (cumsum(p_ord)[i-1] - cumsum(y_ord)[i-1])^2 / (K - 1)
    extra = torch.where(question_is_score, rps, brier)
    return (ce + lam * extra).mean(), {"ce": ce.mean(), "extra": extra.mean()}
```

- No class weights and no label smoothing: both break calibration.
- Metrics (NumPy, used by `calibrate.py`): `nll`, `brier` (multi-class sum of squares), `rps`, `accuracy`, `auroc` (binary, `regressed` only, on `p_yes`), and `ece_equal_mass(p, y, bins=15)` for the binary `regressed` question: sort by `p_yes`, split into 15 bins of equal count, ECE = sum over bins of (bin size / N) x |mean predicted - mean observed|.

Acceptance: `pytest -q py/brier_ml/tests/test_losses.py` with hand-computed values for each loss and ECE on a toy array.

### M5: Training loop

```
TASK M5: finetune
Goal: Smoke and full LoRA training with the proper-scoring loss, checkpoints every 50 steps, resume, budget guard.
Files (only these): py/brier_ml/finetune.py, py/brier_ml/tests/test_finetune.py
Dependencies allowed: torch, peft, transformers
Line budget: 280
```

Spec, `train(cfg)` where `cfg` holds `run_id`, `quant`, `micro_batch`, `effective_batch=32`, `lr=1e-4`, `warmup_frac=0.03`, `epochs=1`, `max_steps` (None, or 50 for smoke), `ckpt_every=50`, `eval_every=50`, `seed=2026`, `max_usd`, `usd_per_sec`, and `init_adapter` (None, or a path to an adapter to start from; used only by the second-epoch rule in N7):

1. Seed everything (`torch`, `numpy`, `random`). Load base (M3), add LoRA, load and tokenize the train split and a fixed evaluation subset: the first 256 `calibration` examples of question `regressed`, variant `ev`, sorted by id.
2. Optimizer: `torch.optim.AdamW(trainable_params, lr, betas=(0.9, 0.999), weight_decay=0.0, fused=True)`. Scheduler: linear warmup over `ceil(0.03 * total_steps)` steps then cosine to 0 at `total_steps = ceil(n_train / effective_batch) * epochs` (or `max_steps`).
3. One optimizer step = `effective_batch / micro_batch` micro-batches from `length_grouped_order` (seeded per epoch with `seed + epoch`). Loss per micro-batch is divided by the accumulation count. Forward under `torch.autocast("cuda", dtype=torch.bfloat16)`. Clip grad norm to 1.0.
4. Log every step to `runs/<run_id>/train_log.jsonl`: step, loss, ce, extra, lr, grad norm, examples per second, non-pad tokens per second, peak GPU memory (`torch.cuda.max_memory_reserved`), elapsed seconds, spend so far.
5. Every `eval_every` steps: NLL and Brier of the eval subset (no grad), logged.
6. Every `ckpt_every` steps and at the end: write `runs/<run_id>/ckpt-<step:06d>/` with the adapter (`model.save_pretrained`), `optimizer.pt`, `scheduler.pt`, `state.json` (step, epoch, micro-batch cursor, RNG states, config, data hash), then commit the Volume. Keep the 2 newest checkpoints plus the one with the best eval NLL.
7. Resume: if `runs/<run_id>/` has a checkpoint, load the newest one (adapter weights with `set_peft_model_state_dict`, optimizer, scheduler, RNG) and continue from its cursor. The same command restarts a preempted or timed-out run.
8. Budget guard: before each step, if `elapsed_seconds * usd_per_sec` would pass `max_usd`, save a checkpoint and stop with a clear message.
9. NaN or inf loss: save nothing, stop with the step number and the batch ids.
10. At the end: copy the final adapter to `runs/<run_id>/adapter/` and write `runs/<run_id>/run_meta.json` (base repo and revision, class used, quant, LoRA config and target names, library versions, data sha256, letter token ids, pad id, steps, epochs, final eval NLL, tokens per second, GPU type, seconds, spend).

Smoke mode (`max_steps=50`) also writes `smoke_report.json`: median seconds per step after step 5, tokens per second, peak memory, first-10 and last-10 mean loss, and the projected cost of a full epoch: `steps_per_epoch * median_step_seconds * usd_per_sec`.

Acceptance: `pytest -q py/brier_ml/tests/test_finetune.py` runs 6 steps on CPU with a tiny random model and 24 fake compiled examples, kills itself after step 3 (simulate with `max_steps`), resumes and checks the step counter and that the resumed loss trajectory equals an uninterrupted run.

### M6: Base pass and adapter scoring

```
TASK M6: score
Goal: Zero-shot probabilities and embeddings from the base model; probabilities from an adapter.
Files (only these): py/brier_ml/score.py, py/brier_ml/tests/test_score.py
Line budget: 160
```

Spec:

- `base_pass(out_dir)`: base model without LoRA, `torch.inference_mode()`, micro-batch 32 (halve on OOM until it fits, log the size used). For every example in all four splits, compute last-position option logits and the last hidden state in **one** forward. Write:
  - `out_dir/probs/<split>.jsonl` with `{"id", "pair_id", "split", "question", "variant", "options", "label", "logits": [6 floats], "probs": {"A": .., ...}, "model": "glimmer-base-zeroshot", "revision": <sha>}` (only the K valid letters in `probs`).
  - `out_dir/emb/<split>.npy` (float16, [N, hidden]) and `out_dir/emb/<split>.ids.json` (ids in row order).
- `score_adapter(run_dir, splits=("calibration", "test", "ood"))`: same forward with the adapter attached, writes `run_dir/probs/<split>.jsonl` with `"model": "glimmer-ft-<run_id>"`.
- Raw logits are saved so temperature scaling happens later on CPU without the GPU.

Acceptance: `pytest -q py/brier_ml/tests/test_score.py` on the tiny model (file formats, row order matches ids).

### M7: Modal functions and entrypoint

```
TASK M7: modal_train functions
Goal: Wire M2 to M6 into Modal GPU functions with timeouts and budget caps.
Files (only these): py/brier_ml/modal_train.py
Line budget: 160 (on top of M1)
```

Functions (all with `image=IMAGE`, `volumes={"/models": MODELS, "/data": DATA}`, `secrets=[HF]`):

| Function | GPU | timeout | What it does |
| --- | --- | --- | --- |
| `check(quant="bf16")` | `"H100"` (`"A100-80GB"` when quant is nf4) | 1,800 s | load base (M3), print `info`, run the real-tokenizer checks of M2 on all of `/data/compiled.jsonl`, run one forward on 8 examples, print peak memory and the LoRA target list after `add_lora` on a throwaway copy, write `/models/runs/check.json` |
| `base_pass_fn()` | `"H100"` | 7,200 s | M6 `base_pass` into `/models/runs/base/` |
| `train_fn(run_id, quant, micro_batch, max_steps, epochs, max_usd)` | `"H100"` or `"A100-80GB"` by quant | 14,400 s | M5 `train` |
| `score_fn(run_id)` | `"H100"` | 3,600 s | M6 `score_adapter` |

- GPU rates for the budget guard: H100 0.001097, A100-80GB 0.000694 USD per second (constants in the file, with the date they were read).
- Each function commits the `MODELS` Volume before returning and calls `MODELS.reload()` at start.
- `@APP.local_entrypoint() def main(step: str, run_id: str = "glimmer-ft-v1", quant: str = "bf16", micro_batch: int = 8, max_steps: int = 0, epochs: int = 1, max_usd: float = 12.0, init_adapter: str = "", lr: float = 1e-4)` dispatches to the function for `step` in `download`, `check`, `base`, `smoke`, `train`, `score`. `smoke` means `max_steps=50` and `max_usd=3`.
- Two GPU functions never run at once from this guide (budget and simplicity).

Acceptance: `PYTHONPATH=py modal run py/brier_ml/modal_train.py --step check` succeeds (this is runbook step N3).

### M8: Frozen head (CPU)

```
TASK M8: frozen_head
Goal: Train the MLP head on base embeddings with the same proper-scoring loss; write probabilities.
Files (only these): py/brier_ml/frozen_head.py, py/brier_ml/tests/test_frozen_head.py
Line budget: 160
```

Spec (plan: the DRS recipe and the safety net):

- Inputs: `runs/base/emb/<split>.npy` and ids, joined to `compiled.jsonl` for labels, question, K and ordinal maps.
- Standardize embeddings with train mean and std (float32).
- Model: shared trunk `Linear(hidden, 256) -> GELU -> Dropout(0.1)`, then one linear head per question type with K outputs. Loss: `losses.proper_loss` with the same masks.
- AdamW lr 1e-3, weight decay 1e-4, batch 256, up to 40 epochs, early stopping on calibration NLL of the `regressed` question (patience 5), seed 2026. CPU only.
- Writes `runs/frozen-head/probs/<split>.jsonl` (calibration, test, ood) in the M6 format with `"model": "glimmer-frozen-head"`, plus `runs/frozen-head/head.pt` and `meta.json`.

Acceptance: `pytest -q py/brier_ml/tests/test_frozen_head.py` (learns a toy separable problem; early stopping triggers).

### M9: Temperature scaling and metrics

```
TASK M9: calibrate
Goal: Fit one temperature per question type on the calibration split; compute metrics before and after; write metrics JSON for the eval harness.
Files (only these): py/brier_ml/calibrate.py, py/brier_ml/tests/test_calibrate.py
Line budget: 180
```

Spec:

- `fit_temperatures(probs_cal_jsonl) -> {"regressed": T, "failure_mode": T, "success_bucket": T}`: for each question type minimize calibration NLL over `log T` with `scipy.optimize.minimize_scalar` (bounded, T in [0.05, 20]) using the saved raw logits.
- `apply(probs_jsonl, temps) -> probs_jsonl` writes the same rows with recomputed `probs` and `"temperature"`.
- `metrics(probs_jsonl) -> dict` per question type: `nll`, `brier`, `accuracy`; plus `ece` (15 equal-mass bins) and `auroc` for `regressed`; plus `rps` for `success_bucket`. Compute separately for variant `ev`, variant `noev` and both.
- CLI: `python -m brier_ml.calibrate --model-dir <dir> --name <model name>` fits on `calibration`, applies to `test` and `ood`, writes `<dir>/temps.json`, `<dir>/probs_temp/<split>.jsonl` and `<dir>/metrics.json` with this shape:

```json
{"model": "glimmer-ft-glimmer-ft-v1", "revision": "<sha>", "temperatures": {"regressed": 1.37, "failure_mode": 1.1, "success_bucket": 1.2},
 "splits": {"test": {"before": {"regressed": {"ece": 0.0, "brier": 0.0, "nll": 0.0, "auroc": 0.0, "accuracy": 0.0}, "...": {}},
                     "after":  {"regressed": {"ece": 0.0, "brier": 0.0, "nll": 0.0, "auroc": 0.0, "accuracy": 0.0}}},
            "ood":  {"before": {}, "after": {}}}}
```

The T13 eval harness reads `metrics.json` and the `probs_temp` files for the ablation rows (ECE, AUROC, OOD ECE) and for the replay (`p_yes` per test pair, variant `ev`).

Acceptance: `pytest -q py/brier_ml/tests/test_calibrate.py` (temperature recovers a known T on synthetic logits within 5%; ECE equals a hand-computed value).

---

## 7. Runbook

Each step lists its exit condition and its spend. After every GPU step, print the Modal spend so far (from the function logs' own spend line) and stop if the total would pass $22.

**N0 Setup.**

```bash
pip install modal==<pin>           # same pin as the image
modal token set --token-id "$MODAL_TOKEN_ID" --token-secret "$MODAL_TOKEN_SECRET"   # skip if already set
modal secret create huggingface HF_TOKEN="$HF_TOKEN"                                  # skip if it exists
export PYTHONPATH=py
```

Exit: `modal profile current` shows Brooklyn's workspace; Brooklyn has confirmed the $25 budget alert.

**N1 Data to Modal.**

```bash
gcloud storage cp "gs://$BRIER_BUCKET/release/v1/compiled.jsonl" data/compiled.jsonl
gcloud storage cp "gs://$BRIER_BUCKET/release/v1/SHA256SUMS" data/SHA256SUMS
(cd data && grep compiled.jsonl SHA256SUMS | sha256sum -c -)
modal volume put brier-data data/compiled.jsonl compiled.jsonl --force
```

Exit: checksum OK; `modal volume ls brier-data` shows the file.

**N2 Weights.** `modal run py/brier_ml/modal_train.py --step download`. Exit: M1 acceptance listing. Spend < $1.

**N3 Check.** `modal run py/brier_ml/modal_train.py --step check`. Exit: gate G1 passes (section 8). Spend ~$0.50. If BF16 does not fit for a single forward of 8 examples, run `--quant nf4` (A100-80GB) and continue with NF4 for every later step.

**N4 Base pass.** `modal run --detach py/brier_ml/modal_train.py --step base`. Follow with `modal app logs brier-train`. Exit: gate G2. Then copy results down:

```bash
modal volume get brier-models runs/base data/runs/base --force
```

**N5 Frozen head (CPU, local, can run while N6 trains).** `python -m brier_ml.frozen_head --base data/runs/base --compiled data/compiled.jsonl --out data/runs/frozen-head`, then `python -m brier_ml.calibrate --model-dir data/runs/frozen-head --name glimmer-frozen-head`. Also run calibrate on the base pass: `python -m brier_ml.calibrate --model-dir data/runs/base --name glimmer-base-zeroshot`. Exit: both `metrics.json` files exist.

**N6 Smoke.** `modal run py/brier_ml/modal_train.py --step smoke --run-id glimmer-ft-smoke --micro-batch 8`. If it runs out of memory, retry with `--micro-batch 4`, then `2`. If `2` still fails on H100 BF16, use `--quant nf4` (A100-80GB) with `--micro-batch 8`. Read `smoke_report.json`. Exit: gate G3. Decide the full-run settings: the quant and micro-batch that passed, and its projected epoch cost must be under $12. If the BF16 run passed but its projected cost is over $12 while NF4's (measure it with one more smoke only if needed) is under, use NF4. Delete `runs/glimmer-ft-smoke` checkpoints after reading the report (keep the report).

**N7 Full fine-tune.**

```bash
modal run --detach py/brier_ml/modal_train.py --step train --run-id glimmer-ft-v1 --quant <q> --micro-batch <m> --epochs 1 --max-usd 12
```

Watch `train_log.jsonl` every 15 minutes (`modal volume get brier-models runs/glimmer-ft-v1/train_log.jsonl /tmp/` and tail it). If the function times out or is preempted, run the same command again; it resumes. Exit: `runs/glimmer-ft-v1/adapter/` and `run_meta.json` exist; gate G4 training checks pass.

Second epoch rule (plan): compare the eval NLL logged at the end of epoch 1 with the value at 75% of epoch 1. Only if it is still falling by more than 1% relative, and the total spend so far plus one more epoch stays under $22, run `--run-id glimmer-ft-v1e2` with `init_adapter=/models/runs/glimmer-ft-v1/adapter`, lr 3e-5 and cosine to 0 (the entrypoint passes `--init-adapter` and `--lr` through to M5). Keep epoch 2 only if its calibration NLL is lower; otherwise the v1 adapter is final.

**N8 Score the fine-tuned model.** `modal run py/brier_ml/modal_train.py --step score --run-id glimmer-ft-v1`, then:

```bash
modal volume get brier-models runs/glimmer-ft-v1 data/runs/glimmer-ft-v1 --force
python -m brier_ml.calibrate --model-dir data/runs/glimmer-ft-v1 --name glimmer-ft-v1
```

Exit: gate G4 metric checks.

**N9 Hand off to the eval harness.** Upload the three model folders (probs, probs_temp, temps, metrics, run_meta, no checkpoints) to `gs://$BRIER_BUCKET/release/v1/models/`. Run T13's eval command if it exists so `report.json` gains the rows "Glimmer base, zero-shot logprobs", "Glimmer frozen head", "Glimmer fine-tuned", "Fine-tuned + temperature".

**N10 Publish the adapter.**

```bash
python - <<'EOF'
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
repo = f'{os.environ["HF_USER"]}/brier-glimmer-lora'
api.create_repo(repo, repo_type="model", exist_ok=True, private=False)
api.upload_folder(repo_id=repo, folder_path="data/runs/glimmer-ft-v1/adapter")
api.upload_file(repo_id=repo, path_or_fileobj="data/runs/glimmer-ft-v1/temps.json", path_in_repo="temps.json")
api.upload_file(repo_id=repo, path_or_fileobj="data/runs/glimmer-ft-v1/run_meta.json", path_in_repo="run_meta.json")
EOF
```

Then write and upload the model card (`README.md` in that repo): base model and pinned revision, Apache 2.0, what the adapter predicts, the exact input format (raw compiled prompt, no chat template, answer token `" A"` to `" F"` at the last position, softmax over valid letters, then temperature per question), the training data (link to the Guide 1 dataset), hyperparameters from `run_meta.json`, and the test and OOD metrics before and after temperature from `metrics.json`. Exit: the repo opens logged out.

**N11 Clean up.** Delete smoke checkpoints and all but the final and best checkpoints of `glimmer-ft-v1` from the Volume. Keep the base weights (serving needs them). Print the final Modal spend.

---

## 8. Gates

| Gate | Check | If it fails |
| --- | --- | --- |
| G1 check | Weights load; class and parameter count as M3 expects; every option letter is one token; the boundary property holds for 100% of examples; max prompt length <= 1,536 tokens (report the max and the median); one forward runs; LoRA targets include attention and MLP projections of the text decoder only | Report the exact failure. Template or tokenizer problems go back to Brooklyn (they need a `brier compile` change) |
| G2 base pass | Probability files for calibration, test and OOD, embeddings for all four splits, row counts equal the compiled counts, no NaN | Re-run once; then report |
| G3 smoke | No NaN; mean loss of the last 10 steps below the first 10; peak memory below 78 GB; projected epoch cost under $12 | Follow N6's fallback order; if every option fails, stop and report with the numbers |
| G4 trained | Run finished with no budget stop; on calibration, the fine-tuned `regressed` NLL is lower than zero-shot's; on test after temperature, report ECE, Brier, NLL and AUROC for `regressed` (the plan's acceptance target is ECE <= 0.05 on Nov 2, which is reported here, not enforced) | If fine-tuned NLL is not below zero-shot, report both and the training curve. The frozen head is the fallback model for the ablation and the replay |

---

## 9. Troubleshooting

- `Unrecognized configuration class` or a missing `muse_glimmer` model type: the image's `transformers` is older than the release that added Glimmer. Bump the pin to the first version that loads it, rebuild, and record the change.
- Out of memory at load with BF16: check that nothing else holds GPU memory, that `device_map={"": 0}` is set and that the multimodal class is not loading duplicated tied weights. Then use the NF4 fallback.
- Out of memory during training: lower micro-batch (effective batch stays 32 via accumulation). Confirm logits are only computed at the last position.
- Loss is NaN in the first steps: check that no example has all options masked, that `label_idx < K` everywhere and that LoRA parameters are float32 (PEFT's default `autocast_adapter_dtype=True` keeps them so).
- Very slow steps: confirm `attn_implementation="sdpa"`, gradient checkpointing with `use_reentrant=False`, and that sequences are left-padded only to the batch maximum (length-grouped batches).
- Volume writes missing after a crash: every checkpoint calls `MODELS.commit()`; reading functions call `MODELS.reload()` first.
- Modal function killed at the timeout: rerun the same command; it resumes from the newest checkpoint.

---

## 10. Serving contract for T20 (so serving matches training)

This guide does not build serving, but whoever builds `POST /decide` on vLLM must keep these invariants, and must prove them with the parity test below before the endpoint is used by `brier decide`.

- Base weights at the exact revision in `revision.txt`, adapter from `glimmer-ft-v1/adapter`.
- Use the completions API with the raw compiled prompt (no chat template, no reasoning parser), `max_tokens=1`, `temperature=0`, `logprobs=20`, prefix caching on. The tokenizer must add the same special tokens as training (`add_special_tokens=True`); verify with vLLM's `/tokenize` endpoint on 20 prompts that the ids equal the training ids exactly.
- Probabilities: take the logprobs of `" A"` to the K-th letter from the top-20 list, renormalize over the valid letters, then apply the temperature for that question type from `temps.json`. If a valid letter is missing from the top-20 list, fail the request loudly; never fill in a default.
- Parity test: score 100 test examples through the endpoint and compare with `probs_temp/test.jsonl`. Maximum absolute difference in `p_yes` must be below 0.02. BF16 kernels differ slightly between engines; a larger gap means the prompts or tokens differ.
- vLLM's LoRA support for this architecture must be confirmed on the pinned vLLM version. If it is not supported, merge the adapter into a BF16 copy of the base weights for serving and keep the unmerged adapter as the published artifact.

---

## 11. Final message checklist (what Codex reports at the end)

- Pinned versions: torch, transformers, peft, accelerate, bitsandbytes, modal; base revision sha.
- Quant and micro-batch used and why (smoke numbers).
- Steps, tokens per second, wall time, Modal spend per step and total.
- Test and OOD metrics for zero-shot, frozen head, fine-tuned and fine-tuned plus temperature (`regressed`: ECE, Brier, NLL, AUROC; the other questions: NLL and accuracy).
- Paths: Volume run folders, `gs://$BRIER_BUCKET/release/v1/models/`, the Hugging Face adapter URL.
- Every deviation from this guide and the reason.
