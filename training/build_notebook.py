"""Builds training/nlcli_wizard_train_v2.ipynb.

The notebook is generated rather than hand-edited so its cells stay reviewable in
diffs and cannot drift into invalid JSON.

Design goals, all of them reactions to how the previous notebook went wrong:

1. **The evaluation inside the notebook is the same code as the evaluation on the
   laptop.** It imports `eval/` from the cloned repo rather than reimplementing
   scoring in a cell. The old notebook had its own copy of the eval logic, which
   is how a contaminated metric survived for nine months.

2. **Baselines run before training, in the same session.** Base model zero-shot and
   few-shot are measured against the same held-out set with the same scorer. Without
   this a fine-tune number means nothing.

3. **A contamination gate aborts the notebook** if the training file leaks the
   held-out set. Not a warning - an exception.

4. **Controlled comparison.** Default base model is Gemma 3 4B, the same model that
   scored 46.6% on the v1 dataset, so the v1 -> v2 ablation changes one variable.

5. **`train_on_responses_only`** so loss is computed on the answer, not on the
   prompt tokens the model is given anyway.

6. **The validation split is command-level**, via eval/splits.py. A random split of
   a generated dataset puts paraphrases of the same command on both sides.

    python training/build_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

NB_PATH = Path(__file__).parent / "nlcli_wizard_train_v2.ipynb"

REPO = "https://github.com/pranavkumaarofficial/nlcli-wizard.git"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip("\n").splitlines(True),
    }


cells: List[dict] = []

# ---------------------------------------------------------------------------

cells.append(md("""
# nlcli-wizard — v2 training run

Trains a Docker NL→CLI translator and **measures it honestly in the same notebook**.

## What is different from the previous notebook

The earlier notebook reported 94% accuracy. That number was measured on training
data — the eval script scored the last 100 lines of the same JSONL the notebook
trained on. The corrected figure for that model is **46.6%**.
See `docs/EVAL_METHODOLOGY.md` in the repo.

This notebook is built so that cannot happen again:

| | Old notebook | This notebook |
|---|---|---|
| Eval code | its own copy, in a cell | imports `eval/` from the repo — same code as local |
| Eval set | last 100 rows of the training file | 116 hand-written held-out prompts |
| Contamination check | none | gate that **raises** and stops the run |
| Baseline | none | base model zero-shot + few-shot, before training |
| Validation split | random rows | command-level (`eval/splits.py`) |
| Loss | on prompt + answer | answer only (`train_on_responses_only`) |
| Confidence field | `random.uniform(0.90, 0.97)` | removed |

## What you get at the end

An ablation table. Every row measured by the same scorer on the same held-out set:

```
config                     overall   unseen_cmd   unseen_phrasing
base, zero-shot                  ?            ?                 ?
base, 8-shot                     ?            ?                 ?
v1 fine-tune (known)         46.6%        38.0%             53.0%
v2 fine-tune                     ?            ?                 ?
```

**Runtime:** ~50–70 min on a free T4. Baselines ~15 min, training ~25 min,
evaluation ~15 min, GGUF export ~10 min.

**Before you start:** Runtime → Change runtime type → **T4 GPU**.
"""))

# ---------------------------------------------------------------------------
cells.append(md("## 1. Setup"))

cells.append(code(f"""
# Clone the repo (or refresh it if the runtime already has it)
import os, subprocess, sys

REPO = "{REPO}"

if not os.path.exists('/content/nlcli-wizard'):
    subprocess.run(['git', 'clone', REPO, '/content/nlcli-wizard'], check=True)

os.chdir('/content/nlcli-wizard')
subprocess.run(['git', 'pull', '--ff-only'], check=False)
sys.path.insert(0, '/content/nlcli-wizard')

print("cwd:", os.getcwd())
print("HEAD:", subprocess.run(['git','rev-parse','--short','HEAD'],
                              capture_output=True, text=True).stdout.strip())
"""))

cells.append(code("""
import torch
assert torch.cuda.is_available(), (
    "No GPU. Runtime -> Change runtime type -> T4 GPU, then re-run from the top."
)
print("GPU:", torch.cuda.get_device_name(0))
print("VRAM: %.1f GB" % (torch.cuda.get_device_properties(0).total_memory / 1e9))
"""))

cells.append(code("""
%%capture
# Unsloth + training stack
!pip install -q "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install -q --no-deps xformers trl peft accelerate bitsandbytes
"""))

# ---------------------------------------------------------------------------
cells.append(md("""
## 2. Configuration

`BASE_MODEL` defaults to Gemma 3 4B — the *same* model that scored 46.6% on the v1
dataset. Keeping it fixed means the v1→v2 comparison changes exactly one variable.

Swap to `unsloth/Qwen3-4B-Instruct-2507` for a second run **after** the controlled
one, so model and dataset effects stay separable.
"""))

cells.append(code("""
CLI_TOOL      = "docker"
BASE_MODEL    = "unsloth/gemma-3-4b-it"     # controlled: same model as the 46.6% run
TRAIN_FILE    = "data/docker_train_v2.jsonl"
TEST_FILE     = "data/docker_test_handwritten.jsonl"
OUTPUT_PREFIX = "docker_gemma3_4b_v2"

MAX_SEQ_LEN   = 512
EPOCHS        = 2          # 5k examples; 3 epochs on this size overfits
LR            = 2e-4
LORA_R        = 32         # up from 16: more capacity for flag composition
LORA_ALPHA    = 64
SEED          = 42

# Known reference point for the ablation table (docs/EVAL_METHODOLOGY.md)
V1_RESULT = {"overall": 0.466, "unseen_command": 0.380, "unseen_phrasing": 0.530}

print(f"{BASE_MODEL}  |  {TRAIN_FILE}  |  {EPOCHS} epochs, lr={LR}, r={LORA_R}")
"""))

# ---------------------------------------------------------------------------
cells.append(md("""
## 3. Contamination gate

Runs **before** anything is trained. If the training file leaks the held-out set,
this raises and the notebook stops. That is deliberate: a run that cannot be
measured honestly is not worth the GPU minutes.
"""))

cells.append(code("""
from pathlib import Path
from eval.contamination import audit, load_jsonl, self_audit

train_examples = load_jsonl(Path(TRAIN_FILE))
test_examples  = load_jsonl(Path(TEST_FILE))

print("Training set self-audit")
for k, v in self_audit(train_examples).items():
    print(f"   {k:<28} {v}")
print()

report = audit(train_examples, test_examples)
print(report.format())

if not report.is_clean:
    raise RuntimeError(
        "CONTAMINATED: the training file contains held-out prompts. "
        "Regenerate with:  python -m nlcli_wizard.dataset_v2 "
        f"--out {TRAIN_FILE} --exclude {TEST_FILE}"
    )
print("\\nGate passed - the held-out set is disjoint from training.")
"""))

# ---------------------------------------------------------------------------
cells.append(md("""
## 4. Shared evaluation helper

This wraps the repo's scorer. Every row of the ablation table goes through it —
baselines and fine-tunes alike — so no configuration gets a favourable code path.
"""))

cells.append(code("""
import json, time, torch
from eval.metrics import categorize, score_all
from eval.normalize import normalized_string
from eval.run_eval import extract_command

TRAIN_COMMANDS = {normalized_string(e.command) for e in train_examples}

FEW_SHOT = [
    ("run nginx on port 8080 in the background", "docker run -d -p 8080:80 nginx"),
    ("drop me into a shell on the api container", "docker exec -it api bash"),
    ("build it and tag it myapp:2.0", "docker build -t myapp:2.0 ."),
    ("what containers are running", "docker ps"),
    ("bring the stack up detached", "docker-compose up -d"),
    ("postgres named db with password secret, backgrounded",
     "docker run -d --name db -e POSTGRES_PASSWORD=secret postgres"),
    ("delete unused volumes", "docker volume prune"),
    ("follow the logs on web", "docker logs -f web"),
]

SYSTEM_HINT = (
    "You translate natural language into a single Docker CLI command. "
    "Reply with exactly one line in the form 'COMMAND: <command>'. "
    "Use short flags (-d, -p, -e, -v, -it, --name, --rm, --restart, --network). "
    "Do not explain."
)


def build_messages(instruction, mode):
    \"\"\"mode: 'plain' | 'system' | 'fewshot'\"\"\"
    msgs = []
    if mode == "system":
        msgs.append({"role": "user", "content": SYSTEM_HINT + "\\n\\n" + instruction})
        return msgs
    if mode == "fewshot":
        shots = []
        for q, a in FEW_SHOT:
            shots.append(f"Translate to docker command: {q}\\nCOMMAND: {a}")
        prefix = SYSTEM_HINT + "\\n\\n" + "\\n\\n".join(shots) + "\\n\\n"
        msgs.append({"role": "user", "content": prefix + instruction})
        return msgs
    msgs.append({"role": "user", "content": instruction})
    return msgs


@torch.no_grad()
def evaluate(model, tokenizer, label, mode="plain", max_new_tokens=64, limit=None):
    \"\"\"Score a model on the held-out set. Returns a summary dict.\"\"\"
    examples = test_examples[:limit] if limit else test_examples
    preds, golds, cats, novs, records = [], [], [], [], []

    t0 = time.time()
    for i, ex in enumerate(examples, 1):
        messages = build_messages(ex.instruction, mode)
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        text = tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        pred = extract_command(text)

        nov = ("unseen_phrasing"
               if normalized_string(ex.command) in TRAIN_COMMANDS
               else "unseen_command")

        preds.append(pred); golds.append(ex.command)
        cats.append(categorize(ex.command)); novs.append(nov)
        records.append({"instruction": ex.instruction, "gold": ex.command,
                        "predicted": pred, "raw_output": text, "novelty": nov,
                        "category": cats[-1]})
        if i % 25 == 0:
            print(f"    {i}/{len(examples)}  ({time.time()-t0:.0f}s)", flush=True)

    overall = score_all(preds, golds, cats)
    by_nov  = score_all(preds, golds, novs)

    print(f"\\n=== {label} ===")
    print(overall.format_table())
    for nov in sorted(by_nov.per_category):
        s = by_nov.per_category[nov]; n = s["n"]
        print(f"  {nov:<18} n={n:<4} func {s['functional']/n:>6.1%}")

    summary = {
        "label": label,
        "mode": mode,
        "n": overall.n,
        "overall": {m: overall.rate(m) for m in ("exact", "normalized", "functional")},
        "by_novelty": {k: {"n": v["n"], "functional": v["functional"]/v["n"]}
                       for k, v in by_nov.per_category.items()},
        "by_category": {k: {"n": v["n"], "functional": v["functional"]/v["n"]}
                        for k, v in overall.per_category.items()},
    }
    Path("results").mkdir(exist_ok=True)
    with open(f"results/{label}_generations.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\\n")
    with open(f"results/{label}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


RESULTS = {}
print("evaluate() ready")
"""))

# ---------------------------------------------------------------------------
cells.append(md("""
## 5. Baselines — before any training

If the base model with few-shot prompting already matches the fine-tune, the
fine-tuning is not earning its keep, and that is a finding worth having.

Baselines run on a subset (`BASELINE_LIMIT`) to save GPU time. Set it to `None`
for the full 116 if you want the baseline directly comparable to the final number.
"""))

cells.append(code("""
from unsloth import FastLanguageModel

BASELINE_LIMIT = 60   # None for all 116

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL,
    max_seq_length=MAX_SEQ_LEN,
    dtype=None,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)

print("Chat template check — this is what the model will actually receive:")
print(repr(tokenizer.apply_chat_template(
    [{"role": "user", "content": "TEST"}],
    tokenize=False, add_generation_prompt=True)))
"""))

cells.append(code("""
RESULTS["base_zeroshot"] = evaluate(
    model, tokenizer, "base_zeroshot", mode="plain", limit=BASELINE_LIMIT)
"""))

cells.append(code("""
RESULTS["base_system"] = evaluate(
    model, tokenizer, "base_system", mode="system", limit=BASELINE_LIMIT)
"""))

cells.append(code("""
RESULTS["base_fewshot"] = evaluate(
    model, tokenizer, "base_fewshot", mode="fewshot", max_new_tokens=48,
    limit=BASELINE_LIMIT)
"""))

# ---------------------------------------------------------------------------
cells.append(md("""
## 6. Fine-tune on v2

The model is reloaded from scratch so the baseline inference above cannot affect
training state.
"""))

cells.append(code("""
import gc
del model
gc.collect(); torch.cuda.empty_cache()

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL,
    max_seq_length=MAX_SEQ_LEN,
    dtype=None,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_R,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=LORA_ALPHA,
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=SEED,
)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"trainable {trainable:,} / {total:,} = {100*trainable/total:.2f}%")
"""))

cells.append(md("""
### Command-level validation split

A random split of a generated dataset puts paraphrases of the same command on both
sides, so validation loss stops being a signal. `eval/splits.py` partitions on the
target command instead.
"""))

cells.append(code("""
from datasets import Dataset
from eval.splits import split_by_command

split = split_by_command(train_examples, test_fraction=0.05, seed=SEED)
print(split.summary())

def to_text(examples_list):
    rows = []
    for e in examples_list:
        messages = [
            {"role": "user", "content": e.instruction},
            {"role": "assistant", "content": f"COMMAND: {e.command}"},
        ]
        rows.append({"text": tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False)})
    return Dataset.from_list(rows)

train_ds = to_text(split.train)
val_ds   = to_text(split.test)
print(f"\\ntrain {len(train_ds)}   val {len(val_ds)}")
print("\\nFormatted example:\\n" + train_ds[0]["text"])
"""))

cells.append(md("""
### Mask the prompt

`train_on_responses_only` computes loss on the answer alone. Without it the model
spends capacity learning to reproduce instructions it is always given.

The turn markers differ between model families, so they are detected from the
tokenizer's own template rather than hardcoded — the previous notebook hardcoded
Gemma 3 markers while pointing at a Gemma 4 model.
"""))

cells.append(code("""
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth.chat_templates import train_on_responses_only

probe = tokenizer.apply_chat_template(
    [{"role": "user", "content": "U"}, {"role": "assistant", "content": "A"}],
    tokenize=False, add_generation_prompt=False)

if "<|turn>" in probe:                       # Gemma 4
    INSTR_PART, RESP_PART = "<|turn>user\\n", "<|turn>model\\n"
elif "<start_of_turn>" in probe:             # Gemma 2 / 3
    INSTR_PART, RESP_PART = "<start_of_turn>user\\n", "<start_of_turn>model\\n"
elif "<|im_start|>" in probe:                # Qwen
    INSTR_PART, RESP_PART = "<|im_start|>user\\n", "<|im_start|>assistant\\n"
elif "<|start_header_id|>" in probe:         # Llama 3
    INSTR_PART = "<|start_header_id|>user<|end_header_id|>\\n\\n"
    RESP_PART  = "<|start_header_id|>assistant<|end_header_id|>\\n\\n"
else:
    raise RuntimeError(f"Unrecognised chat template:\\n{probe}")

print(f"instruction marker {INSTR_PART!r}")
print(f"response marker    {RESP_PART!r}")
assert INSTR_PART in probe and RESP_PART in probe, "markers not found in template"
"""))

cells.append(code("""
args = TrainingArguments(
    output_dir="./outputs",
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=LR,
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    optim="adamw_8bit",
    fp16=True,
    logging_steps=20,
    eval_strategy="steps",
    eval_steps=50,
    per_device_eval_batch_size=4,
    save_strategy="steps",
    save_steps=50,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    gradient_checkpointing=True,
    max_grad_norm=1.0,
    seed=SEED,
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LEN,
    args=args,
    packing=False,
)

trainer = train_on_responses_only(
    trainer, instruction_part=INSTR_PART, response_part=RESP_PART
)
print("trainer ready — loss is computed on responses only")
"""))

cells.append(code("""
# Verify the mask before spending 25 minutes on it: the decoded labels should
# contain ONLY the answer. If the instruction appears here, the mask is wrong.
sample = trainer.train_dataset[0]
labels = [t for t in sample["labels"] if t != -100]
print("Supervised tokens decode to:")
print(repr(tokenizer.decode(labels)))
print()
assert "COMMAND:" in tokenizer.decode(labels), "response masking looks wrong"
print("Mask verified.")
"""))

cells.append(code("""
stats = trainer.train()

print("\\nRuntime: %.1f min" % (stats.metrics['train_runtime'] / 60))
print("Final train loss: %.4f" % stats.metrics['train_loss'])
evals = [l for l in trainer.state.log_history if 'eval_loss' in l]
if evals:
    print("Best val loss:    %.4f" % min(e['eval_loss'] for e in evals))
"""))

# ---------------------------------------------------------------------------
cells.append(md("## 7. Evaluate the fine-tune — same scorer, same held-out set"))

cells.append(code("""
FastLanguageModel.for_inference(model)
RESULTS["v2_finetune"] = evaluate(model, tokenizer, "v2_finetune", mode="plain")
"""))

cells.append(md("## 8. Ablation table"))

cells.append(code("""
rows = [
    ("base, zero-shot",   RESULTS.get("base_zeroshot")),
    ("base, +system",     RESULTS.get("base_system")),
    ("base, 8-shot",      RESULTS.get("base_fewshot")),
    ("v1 fine-tune",      {"n": 116, "overall": {"functional": V1_RESULT["overall"]},
                           "by_novelty": {
                               "unseen_command":  {"functional": V1_RESULT["unseen_command"]},
                               "unseen_phrasing": {"functional": V1_RESULT["unseen_phrasing"]}}}),
    ("v2 fine-tune",      RESULTS.get("v2_finetune")),
]

print(f"{'config':<22}{'n':>5}{'overall':>10}{'unseen_cmd':>13}{'unseen_phr':>13}")
print("-" * 63)
for name, r in rows:
    if not r:
        print(f"{name:<22}{'—':>5}{'not run':>10}")
        continue
    o = r["overall"]["functional"]
    uc = r["by_novelty"].get("unseen_command", {}).get("functional")
    up = r["by_novelty"].get("unseen_phrasing", {}).get("functional")
    n = r.get("n", "")
    print(f"{name:<22}{n:>5}{o:>9.1%}"
          f"{(f'{uc:.1%}' if uc is not None else '—'):>13}"
          f"{(f'{up:.1%}' if up is not None else '—'):>13}")
print("-" * 63)
print("Baselines may use a subset (BASELINE_LIMIT); compare with that in mind.")
print("n=116 gives roughly +/-9 points at 95% confidence near 50%.")
"""))

cells.append(code("""
# Per-flag-count accuracy — the metric v2 was built to move.
# v1 fine-tune scored: 0 flags 74.0% | 1 flag 47.1% | 2 flags 5.0% | 3+ flags 0.0%
import collections, json
from eval.metrics import score_one
from eval.normalize import parse_command

with open("results/v2_finetune_generations.jsonl") as f:
    recs = [json.loads(l) for l in f]

buckets = collections.defaultdict(lambda: [0, 0])
for r in recs:
    k = min(len(parse_command(r["gold"]).flags), 3)
    buckets[k][1] += 1
    if score_one(r["predicted"], r["gold"]).functional:
        buckets[k][0] += 1

V1_BY_FLAGS = {0: 0.740, 1: 0.471, 2: 0.050, 3: 0.000}
print(f"{'flags':<8}{'n':>5}{'v2':>9}{'v1':>9}{'delta':>9}")
print("-" * 40)
for k in sorted(buckets):
    c, t = buckets[k]
    v2 = c / t
    v1 = V1_BY_FLAGS.get(k)
    label = f"{k}+" if k == 3 else str(k)
    print(f"{label:<8}{t:>5}{v2:>8.1%}{v1:>8.1%}{v2-v1:>+9.1%}")
"""))

# ---------------------------------------------------------------------------
cells.append(md("""
## 9. Export to GGUF

Only worth running if the ablation table above shows an improvement.
"""))

cells.append(code("""
lora_dir = f"{OUTPUT_PREFIX}_lora"
model.save_pretrained(lora_dir); tokenizer.save_pretrained(lora_dir)

merged_dir = f"{OUTPUT_PREFIX}_merged"
model.save_pretrained_merged(merged_dir, tokenizer, save_method="merged_16bit")
print("merged ->", merged_dir)
"""))

cells.append(code("""
%%capture
!git clone https://github.com/ggml-org/llama.cpp /content/llama.cpp
!cd /content/llama.cpp && cmake -B build -DCMAKE_BUILD_TYPE=Release \\
    && cmake --build build --config Release --target llama-quantize -j 4
"""))

cells.append(code("""
fp16 = f"{OUTPUT_PREFIX}_fp16.gguf"
q4   = f"{OUTPUT_PREFIX}_q4km.gguf"

!python /content/llama.cpp/convert_hf_to_gguf.py {merged_dir} --outfile {fp16} --outtype f16
!/content/llama.cpp/build/bin/llama-quantize {fp16} {q4} Q4_K_M

import os
print("%s  %.2f GB" % (q4, os.path.getsize(q4) / 1e9))
"""))

cells.append(code("""
# Bundle results for committing back to the repo.
!mkdir -p results && tar -czf v2_run_results.tar.gz results/

from google.colab import files
files.download('v2_run_results.tar.gz')
print("\\nAlso download the model if the numbers justify it:")
print(f"   files.download('{q4}')")
"""))

cells.append(md("""
## 10. After the run

1. Extract `v2_run_results.tar.gz` into the repo's `results/`.
2. Commit the `*_summary.json` files — they are the published record.
3. Update the ablation table in `docs/EVAL_METHODOLOGY.md` and `notes/PROGRESS.md`.
4. Only update the README headline number if the held-out result actually improved.

If v2 did **not** improve, that is a result too, and the next hypothesis is that the
generated phrasing distribution still does not resemble real user input — which
would point at collecting genuine prompts rather than generating more.
"""))

# ---------------------------------------------------------------------------

notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

NB_PATH.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
print(f"wrote {NB_PATH}  ({len(cells)} cells)")
