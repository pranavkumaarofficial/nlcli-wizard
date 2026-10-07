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

4. **Controlled comparison.** The v1 46.6% was produced by a Gemma 3 *1B*, not the
   4B the old README claimed: the shipped GGUF reports size_label 1000M and
   999,885,952 parameters (`python scripts/gguf_header.py <model>`). So the
   controlled run holds the base model at 1B and moves only the dataset. Model
   capacity is a separate second run on the fixed v2 dataset.

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

**Runtime:** not measured yet. The ~50–70 min figure previously quoted here was an
estimate for a 4B; run 1 is a 1B and should be well under that. Record the real
numbers on the first run and replace this line with them.

The llama.cpp build in section 9 is the slowest single step and is only worth
paying for if the ablation table shows an improvement.

**Before you start:** Runtime → Change runtime type → **T4 GPU**.
"""))

# ---------------------------------------------------------------------------
cells.append(md("## 1. Setup"))

cells.append(code(r"""
# Clone the repo at a named branch.
#
# Experiments run from a branch, not from main: this notebook has to be executed
# before anyone can claim it works, and main should not carry training code that
# has never completed a run. Merge after the ablation table lands.
#
# Every number below should be traceable to one commit, so the branch, the commit
# and its subject are printed here and recorded into each results summary.
import os, subprocess, sys

REPO   = "__REPO__"
BRANCH = "train/v2-run"          # <- the branch you pushed. "main" once merged.
CLONE  = "/content/nlcli-wizard"


def git(*args):
    r = subprocess.run(["git", "-C", CLONE, *args], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("git " + " ".join(args) + "\n" + r.stderr.strip())
    return r.stdout.strip()


if not os.path.exists(CLONE):
    r = subprocess.run(["git", "clone", "--branch", BRANCH, REPO, CLONE],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(
            f"Could not clone branch {BRANCH!r}.\n"
            "Most likely it has not been pushed yet. From your laptop:\n"
            f"    git push -u origin {BRANCH}\n\n" + r.stderr.strip())
else:
    # Runtime already has a clone, possibly on another branch or another commit.
    git("fetch", "origin", BRANCH)
    git("checkout", "-B", BRANCH, f"origin/{BRANCH}")
    git("reset", "--hard", f"origin/{BRANCH}")

os.chdir(CLONE)
sys.path.insert(0, CLONE)

COMMIT  = git("rev-parse", "--short", "HEAD")
SUBJECT = git("log", "-1", "--format=%s")

print(f"branch : {BRANCH}")
print(f"commit : {COMMIT}  {SUBJECT}")

assert not git("status", "--porcelain"), (
    "working tree is dirty - results from it would not be reproducible"
)
print("clean checkout")
""".replace("__REPO__", REPO)))

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

Two open questions, one variable each. Do not try to answer both in one run.

| run | `BASE_MODEL` | holds fixed | measures |
|---|---|---|---|
| **1 (this one)** | `unsloth/gemma-3-1b-it` | model size | does the v2 dataset fix flag composition? |
| 2 | `unsloth/gemma-3-4b-it` | v2 dataset | was the ceiling capacity after all? |

Run 1 is 1B because **the 46.6% was produced by a 1B**. The file is named
`docker_gemma3_4b_q4km.gguf` and the old README said "Gemma 3 4B", but its header
reports `general.size_label 1000M` and 999,885,952 parameters over 340 tensors.
Verify with `python scripts/gguf_header.py models/docker_gemma3_4b_q4km.gguf`.

Training a 4B here and subtracting 46.6% would move dataset *and* model size at
once and tell you nothing about either. Run 2 then settles the withdrawn
1B-vs-4B "capacity ceiling" claim properly, because by then the dataset is fixed.

Qwen3-4B is a third run, not a variant of either.
"""))

cells.append(code("""
CLI_TOOL      = "docker"
BASE_MODEL    = "unsloth/gemma-3-1b-it"     # run 1: matches the 1B behind 46.6%
TRAIN_FILE    = "data/docker_train_v2.jsonl"
TEST_FILE     = "data/docker_test_handwritten.jsonl"
OUTPUT_PREFIX = "docker_gemma3_1b_v2"

MAX_SEQ_LEN   = 512
EPOCHS        = 2          # 5k examples; 3 epochs on this size overfits
LR            = 2e-4
LORA_R        = 32         # up from 16: more capacity for flag composition
LORA_ALPHA    = 64
SEED          = 42

# Reference point for the ablation table (docs/EVAL_METHODOLOGY.md, section 4).
# Measured on all 116 with a Gemma 3 1B fine-tuned on the v1 dataset. Comparable to
# this run only while BASE_MODEL is the 1B and EVAL_LIMIT is None.
V1_RESULT = {"overall": 0.466, "unseen_command": 0.380, "unseen_phrasing": 0.530}
V1_BASE   = "unsloth/gemma-3-1b-it"

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
def evaluate(model, tokenizer, label, mode="plain", max_new_tokens=64):
    \"\"\"Score a model on EVAL_SUBSET. Returns a summary dict.

    There is no per-call `limit`. Every row of the ablation table is scored on the
    same examples or the table compares nothing: `test_examples` is ordered by
    category, so a prefix is not a sample. The first 60 rows hold 22 of the 29
    `run` and 9 of the 11 `exec` examples (the two hardest categories, 20.7% and
    9.1% for v1) and zero `system` or `network`. Scored on that prefix the v1 model
    gets 40.0% against its true 46.6% — a 6.6 point penalty applied to whichever
    configs happened to use it.
    \"\"\"
    examples = EVAL_SUBSET
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
        records.append({"prompt": prompt, "instruction": ex.instruction,
                        "gold": ex.command, "predicted": pred, "raw_output": text,
                        "latency_s": None, "mean_logprob": None,
                        "novelty": nov, "category": cats[-1]})
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
        # Provenance: which code and which model produced this row. Without it a
        # summary.json is a number with no way to check what made it.
        "branch": BRANCH,
        "commit": COMMIT,
        "base_model": BASE_MODEL,
        "train_file": TRAIN_FILE,
        # Colab installs are unpinned, so the resolved stack is part of the result.
        # TrainingArguments stopped accepting warmup_ratio between two runs.
        "stack": globals().get("STACK"),
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

cells.append(md("""
### The evaluation subset

`EVAL_SUBSET` is fixed once here and used by every configuration, baselines and
fine-tune alike. Leave `EVAL_LIMIT = None` unless you are short of GPU time.

If you do set a limit, it is drawn **stratified by category with a fixed seed**,
never as a prefix. `data/docker_test_handwritten.jsonl` is grouped by category, so
`test_examples[:60]` is not a sample of the test set: it is 22 of the 29 `run`
examples, 9 of the 11 `exec`, and none of the 16 `system` or 9 `network`. The v1
model scores 46.6% on all 116 and 40.0% on that prefix. Measuring baselines on the
prefix and the fine-tune on the full set would have credited fine-tuning with 6.6
points of sampling bias.
"""))

cells.append(code("""
import collections, random

EVAL_LIMIT = None      # None = all 116. An int = stratified sample of that size.

if EVAL_LIMIT is None or EVAL_LIMIT >= len(test_examples):
    EVAL_SUBSET = list(test_examples)
else:
    by_cat = collections.defaultdict(list)
    for ex in test_examples:
        by_cat[categorize(ex.command)].append(ex)

    rng = random.Random(SEED)
    EVAL_SUBSET, order = [], sorted(by_cat)
    # Largest-remainder allocation so the subset keeps the category proportions.
    quota = {c: len(by_cat[c]) * EVAL_LIMIT / len(test_examples) for c in order}
    take = {c: int(quota[c]) for c in order}
    for c in sorted(order, key=lambda c: quota[c] - take[c], reverse=True):
        if sum(take.values()) >= EVAL_LIMIT:
            break
        take[c] += 1
    for c in order:
        EVAL_SUBSET += rng.sample(by_cat[c], min(take[c], len(by_cat[c])))

EVAL_N = len(EVAL_SUBSET)
print(f"EVAL_SUBSET: {EVAL_N} of {len(test_examples)} examples")
print("  " + "  ".join(
    f"{c}={n}" for c, n in sorted(
        collections.Counter(categorize(e.command) for e in EVAL_SUBSET).items())))
if EVAL_N < len(test_examples):
    print("\\n  NOTE: subset run. The v1 reference row (46.6%) was measured on all")
    print("  116, so it is NOT comparable to these rows. The ablation table says so.")
"""))

# ---------------------------------------------------------------------------
cells.append(md("""
## 5. Baselines — before any training

If the base model with few-shot prompting already matches the fine-tune, the
fine-tuning is not earning its keep, and that is a finding worth having.

Baselines are scored on `EVAL_SUBSET`, the same examples the fine-tune is scored
on. That is the only way the rows of the ablation table mean anything next to each
other.
"""))

cells.append(code("""
from unsloth import FastLanguageModel

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
    model, tokenizer, "base_zeroshot", mode="plain")
"""))

cells.append(code("""
RESULTS["base_system"] = evaluate(
    model, tokenizer, "base_system", mode="system")
"""))

cells.append(code("""
RESULTS["base_fewshot"] = evaluate(
    model, tokenizer, "base_fewshot", mode="fewshot", max_new_tokens=48)
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

cells.append(md("""
### Training arguments, built against the installed signature

The Colab install is `pip install --no-deps trl peft accelerate`, which resolves
to whatever is current that day. That drift has already broken this cell once:
`TrainingArguments` rejected `warmup_ratio` on 2026-10-07.

So the arguments are filtered against the class's real signature instead of
passed blind. A rename with a defined equivalent is rewritten, and a warmup
*ratio* becomes the matching number of *steps* computed from the step count, so
the schedule itself does not change. Anything with no equivalent is printed
under DROPPED, and the six arguments that define what this run *is* are asserted
present: losing `seed` or `max_grad_norm` quietly would make the result
impossible to describe afterwards.
"""))

cells.append(code(r"""
import dataclasses, inspect, math

import torch, transformers, trl, peft, accelerate
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth.chat_templates import train_on_responses_only

try:
    from trl import SFTConfig
except ImportError:
    SFTConfig = None

# Recorded into the results summaries: a run is not reproducible without them.
STACK = {m.__name__: getattr(m, "__version__", "?")
         for m in (transformers, trl, peft, accelerate, torch)}
try:
    import unsloth
    STACK["unsloth"] = getattr(unsloth, "__version__", "?")
except Exception:
    pass
for _k, _v in STACK.items():
    print(f"{_k:<13} {_v}")

# TRL moved the training knobs onto SFTConfig; prefer it where it exists.
CONFIG_CLS = SFTConfig or TrainingArguments
print(f"\nconfig class : {CONFIG_CLS.__module__}.{CONFIG_CLS.__name__}")


def accepted(cls):
    # Init parameters cls will take. None means it accepts **kwargs.
    if dataclasses.is_dataclass(cls):
        return {f.name for f in dataclasses.fields(cls) if f.init}
    try:
        sig = inspect.signature(cls.__init__)
    except (TypeError, ValueError):
        return None
    if any(p.kind is p.VAR_KEYWORD for p in sig.parameters.values()):
        return None
    return {n for n in sig.parameters if n != "self"}


BATCH, ACCUM = 4, 4
STEPS_PER_EPOCH = math.ceil(len(train_ds) / (BATCH * ACCUM))
TOTAL_STEPS = STEPS_PER_EPOCH * EPOCHS
print(f"schedule     : {len(train_ds)} rows -> {STEPS_PER_EPOCH} steps/epoch "
      f"x {EPOCHS} epochs = {TOTAL_STEPS} steps")

WANTED = dict(
    output_dir="./outputs",
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH,
    gradient_accumulation_steps=ACCUM,
    learning_rate=LR,
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    optim="adamw_8bit",
    fp16=True,
    logging_steps=20,
    eval_strategy="steps",
    eval_steps=50,
    per_device_eval_batch_size=BATCH,
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

ok = accepted(CONFIG_CLS)
kwargs, DROPPED, rewritten = {}, [], []
for k, v in WANTED.items():
    if ok is None or k in ok:
        kwargs[k] = v
    elif k == "warmup_ratio" and "warmup_steps" in ok:
        n = max(1, round(v * TOTAL_STEPS))
        kwargs["warmup_steps"] = n
        rewritten.append(f"warmup_ratio={v} -> warmup_steps={n} "
                         f"({v:.0%} of {TOTAL_STEPS} steps, schedule unchanged)")
    elif k == "eval_strategy" and "evaluation_strategy" in ok:
        kwargs["evaluation_strategy"] = v
        rewritten.append(f"eval_strategy -> evaluation_strategy={v!r}")
    else:
        DROPPED.append(k)

for line in rewritten:
    print("  rewrote:", line)
if DROPPED:
    print(f"\n  !! DROPPED, unsupported by this version:", DROPPED)
    print("  Decide whether the run still measures what you intended.")

CRITICAL = {"num_train_epochs", "learning_rate", "max_grad_norm", "seed",
            "per_device_train_batch_size", "gradient_accumulation_steps"}
missing = CRITICAL - set(kwargs)
assert not missing, f"critical training args unsupported: {sorted(missing)}"

# Newer TRL owns the dataset and sequence knobs on the config; older on the trainer.
DATASET_KNOBS = (("dataset_text_field", "text"),
                 ("max_seq_length", MAX_SEQ_LEN),
                 ("packing", False))
for k, v in DATASET_KNOBS:
    if ok is not None and k in ok:
        kwargs[k] = v

args = CONFIG_CLS(**kwargs)

# tokenizer= was renamed processing_class=.
tr_ok = accepted(SFTTrainer)
tr = dict(model=model, train_dataset=train_ds, eval_dataset=val_ds, args=args)
if tr_ok is None or "tokenizer" in tr_ok:
    tr["tokenizer"] = tokenizer
elif "processing_class" in tr_ok:
    tr["processing_class"] = tokenizer
    print("  using processing_class= instead of tokenizer=")
else:
    raise RuntimeError(f"SFTTrainer takes no tokenizer argument: {sorted(tr_ok)}")
for k, v in DATASET_KNOBS:
    if tr_ok is not None and k in tr_ok and k not in kwargs:
        tr[k] = v

trainer = SFTTrainer(**tr)
trainer = train_on_responses_only(
    trainer, instruction_part=INSTR_PART, response_part=RESP_PART
)
print(f"\ntrainer ready - loss is computed on responses only")
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
# The v1 row is a historical number, not something measured in this session. It
# belongs in the table only when this run is actually comparable to it: same base
# model, same full 116 examples. Otherwise it is printed separately, below, so it
# cannot be read as a delta.
V1_COMPARABLE = (BASE_MODEL == V1_BASE) and (EVAL_N == 116)

v1_row = ("v1 fine-tune (v1 data)",
          {"n": 116,
           "overall": {"functional": V1_RESULT["overall"]},
           "by_novelty": {
               "unseen_command":  {"functional": V1_RESULT["unseen_command"]},
               "unseen_phrasing": {"functional": V1_RESULT["unseen_phrasing"]}}})

rows = [
    ("base, zero-shot",   RESULTS.get("base_zeroshot")),
    ("base, +system",     RESULTS.get("base_system")),
    ("base, 8-shot",      RESULTS.get("base_fewshot")),
]
if V1_COMPARABLE:
    rows.append(v1_row)
rows.append(("v2 fine-tune (v2 data)", RESULTS.get("v2_finetune")))

print(f"{'config':<22}{'n':>5}{'overall':>10}{'unseen_cmd':>13}{'unseen_phr':>13}")
print("-" * 65)
seen_n = set()
for name, r in rows:
    if not r:
        print(f"{name:<22}{'-':>5}{'not run':>10}")
        continue
    o = r["overall"]["functional"]
    uc = r["by_novelty"].get("unseen_command", {}).get("functional")
    up = r["by_novelty"].get("unseen_phrasing", {}).get("functional")
    n = r.get("n", "")
    seen_n.add(n)
    flag = "  <- different n" if n != EVAL_N else ""
    print(f"{name:<22}{n:>5}{o:>9.1%}"
          f"{(f'{uc:.1%}' if uc is not None else '-'):>13}"
          f"{(f'{up:.1%}' if up is not None else '-'):>13}{flag}")
print("-" * 65)

if len(seen_n) > 1:
    print()
    print("WARNING: rows were scored on different numbers of examples.")
    print("Only rows with the same n are comparable. The v1 reference was measured")
    print("on all 116; if this run used EVAL_LIMIT, do not subtract it from v2.")

if not V1_COMPARABLE:
    print()
    print("v1 reference EXCLUDED from the table above, because this run is not a")
    print("controlled comparison against it:")
    if BASE_MODEL != V1_BASE:
        print(f"   base model  {BASE_MODEL}  !=  {V1_BASE} (the model behind 46.6%)")
    if EVAL_N != 116:
        print(f"   examples    {EVAL_N}  !=  116 (the set 46.6% was measured on)")
    print("   For the record only, not a delta: v1 = "
          f"{V1_RESULT['overall']:.1%} overall on 116.")

print()
print(f"All measured rows scored on the same {EVAL_N} examples, same scorer.")
print("n=116 gives roughly +/-9 points at 95% confidence near 50%, so treat")
print("differences smaller than that as noise rather than as an improvement.")
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
