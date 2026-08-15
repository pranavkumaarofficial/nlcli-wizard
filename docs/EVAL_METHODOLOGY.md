# Evaluation Methodology

*Last updated: 2026-08-15*

This document exists because this project published accuracy numbers for nine months
that were measured on training data. It records what went wrong, how it was found,
what replaced it, and what the corrected numbers are.

If you are here to check whether the figures in the README can be trusted, the
short answer is: the old ones could not, and this is why.

---

## 1. What went wrong

### 1.1 The evaluation set was inside the training set

`test/evaluate_docker.py` (now `scripts/legacy/evaluate_docker_LEGACY.py`) selected
its evaluation examples like this:

```python
with open('data/docker_training.jsonl') as f:
    examples = [json.loads(line) for line in f][-100:]
```

The last 100 lines of the training file. Meanwhile the training notebook did:

```python
dataset = dataset.train_test_split(test_size=0.1, seed=42)   # of the same file
```

A random 90/10 split of a 594-row file. So approximately 90 of the 100 "evaluation"
rows were in the training partition. The reported **94% Docker accuracy is
training-set recall**, not translation accuracy.

The same pattern produced the **83% venvy** figure via `evaluate_accuracy.py`.

### 1.2 The datasets could not be randomly split without leaking

Even with the eval file separated, a random row-level split of these datasets leaks,
because the generators emit multiple paraphrases per target command.

Measured with `python -m eval.contamination --train <file>`:

| | docker | venvy |
|---|---|---|
| rows | 594 | 1,500 |
| unique instructions | 592 | **230** |
| unique target commands | 298 | **154** |
| instruction duplication factor | 1.00× | **6.52×** |
| target duplication factor | 1.99× | **9.74×** |
| most repeated target | `docker exec -it web bash` (8×) | **`venvy current` (225×)** |
| safe to split randomly? | no | no |

`venvy current` is 15% of the entire venvy dataset. Under the legacy harness,
**95 of the 100 venvy evaluation prompts appear verbatim in training.** The model was
shown the exact input string an average of 6.5 times and still got 17% of them wrong.

For docker, 69 of the 100 legacy evaluation rows share their exact target command
with the training portion. Under a random 10% split, 44 of 59 do.

### 1.3 Exact string match was the only metric

`predicted_cmd == expected_cmd`. This scores `docker run -d -p 8080:80 nginx` and
`docker run -p 8080:80 -d nginx` as a mismatch. Flag order is not semantics. The
field settled this years ago — NLC2CMD scores functional equivalence over utilities
and flag sets, and InterCode-Bash executes both commands in matched containers and
diffs the resulting state.

### 1.4 There was no baseline

No un-fine-tuned model was ever evaluated. Without that, "94%" cannot answer the
only question that matters about a fine-tune: *did the fine-tuning do anything?*
Addressed in Milestone 2, not in this document.

---

## 2. How it was found

An adversarial code review of the repository on 2026-08-15, starting from the
evaluation harness rather than the model. The `[-100:]` slice against the training
filename was visible in the first read of the file.

The lesson worth generalizing: **read the eval harness first.** A contaminated metric
does not announce itself in the training loss, the validation loss, or the sample
outputs. Everything looks healthy. The only place it shows is in how the test set
was constructed.

---

## 3. What replaced it

### 3.1 Contamination auditing is mandatory and automatic

`eval/contamination.py` audits any (train, test) pair across four leak channels:

| Channel | Detects | Severity |
|---|---|---|
| `PROMPT_VERBATIM` | test instruction appears exactly in train | fatal |
| `PROMPT_NEAR_DUP` | differs only by word order, casing, punctuation | fatal |
| `TEMPLATE_SHARED` | matches a train instruction's generator template with entities substituted (`run nginx on port 8080` ≡ `run redis on port 6379`) | high |
| `TARGET_OVERLAP` | gold command appears as a train target under different phrasing | disclose |

`TARGET_OVERLAP` is deliberately *not* fatal. Generalizing across phrasings of a known
command is the actual product task. But it is measured and printed, because a test set
made entirely of seen targets measures paraphrase robustness rather than translation,
and the reader is entitled to know which they are looking at.

The audit runs at the top of every evaluation and prints **above** the accuracy.
`--strict` makes a contaminated run refuse to report a headline number at all.

### 3.2 Splits are by target command, not by row

`eval/splits.py` partitions on the normalized gold command, so no command can appear
on both sides. Stratified by category so the held-out set is not accidentally all
`docker volume`.

### 3.3 A hand-written held-out set

`data/docker_test_handwritten.jsonl` — 116 prompts written by hand rather than emitted
by a template generator, covering the phrasings the generator never produces:

| Style | Example |
|---|---|
| telegraphic | `nginx, port 8080, detached` |
| conversational | `i need a throwaway ubuntu shell` |
| question | `how much cpu is everything using` |
| typo / informal | `whats in /var/log on the db container` |
| jargon | `exec into frontend` |
| minimal | `ps` |
| polite | `could you show me the containers that are currently up` |
| compositional | `postgres called pgmain, port 5432, password secret, data on the pgdata volume, detached` |
| unseen-flag | `start a container from ubuntu capped at 512m of memory` |

Audit result against `data/docker_training.jsonl`:

```
PROMPT_VERBATIM        0     0.0%   FATAL
PROMPT_NEAR_DUP        0     0.0%   FATAL
TEMPLATE_SHARED        0     0.0%   high
TARGET_OVERLAP        66    56.9%   disclose

VERDICT: CLEAN - no prompt leakage.
```

Zero prompt-level leakage on all three fatal/high channels. The 57% target overlap is
disclosed, and is the reason results are reported split by novelty (§3.5).

### 3.4 Three metrics, always together

`eval/metrics.py`:

- **exact** — byte-identical. Kept so old and new numbers are comparable.
- **normalized** — same command path, same flag set (order-insensitive), same
  positional sequence. Credits flag reordering.
- **functional** — normalized plus documented equivalences (`docker ps` ≡
  `docker container ls`, `docker compose` ≡ `docker-compose`).

The gap between `exact` and `normalized` is itself a diagnostic: a large gap means the
model learned correct flags but not the training data's arbitrary flag ordering, which
is a scoring artifact rather than a model failure.

Deliberately *not* treated as equivalent: `-d`, `-a`, `-i`, `-t`. These change
behaviour. Loosening a metric until the number improves is how the first set of
figures happened.

### 3.5 Results are reported split by novelty

Every run reports two partitions separately:

- **unseen_command** — the gold command never appeared as a training target.
  The hard generalization test.
- **unseen_phrasing** — the command was in training, the phrasing was not.
  The realistic deployment test.

Both are published. A single blended number lets the easier partition carry the
harder one.

### 3.6 The harness is tested

`tests/test_eval_harness.py` — 36 tests covering flag-order invariance, the
context-sensitivity of `-t` (tag under `build`, TTY under `run`), bundled short flags,
each contamination channel, and split integrity. Two genuine parser bugs were caught
by these tests during development:

- `-it` failed to expand, because `-t` was globally registered as a value-consuming
  flag.
- `docker run -t nginx` parsed as `-t=nginx` with no image at all, for the same reason.

Both would have silently mis-scored every `exec` and interactive `run` example. This is
the argument for testing a harness before trusting it.

One test asserts the shipped test set is clean against the shipped training set, so a
regenerated dataset that reintroduces a leak fails CI rather than quietly publishing.

---

## 4. Corrected results

Run: `results/gemma3_4b_docker_summary.json`, 2026-08-15, `models/docker_gemma3_4b_q4km.gguf`,
Q4_K_M, llama.cpp b10437 CPU, temperature 0, 116 held-out examples, 4.67 s/example.

| Model | Legacy (contaminated) | Corrected | unseen_command | unseen_phrasing |
|---|---|---|---|---|
| Gemma 3 4B | 94.0% | **46.6%** | 38.0% (n=50) | 53.0% (n=66) |
| Gemma 3 1B | 73–76% | *not run — weights unavailable locally* | | |

**exact, normalized, and functional all returned 46.6%.** This matters: the looser
metrics were built specifically to give the model every benefit of the doubt, and
they moved the number by zero points. Not one prediction was penalised for flag
ordering or for `docker ps` vs `docker container ls`. The 47-point drop from the
published figure is contamination, not a scoring artifact.

### Per-category

| Category | n | Corrected | Legacy claim | Δ |
|---|---|---|---|---|
| volume | 7 | 100.0% | 100% | 0 |
| system | 16 | 68.8% | 100% | −31 |
| ps_images | 20 | 60.0% | 87.5% | −28 |
| build | 9 | 55.6% | 90.0% | −34 |
| network | 9 | 55.6% | 100% | −44 |
| compose | 15 | 46.7% | 100% | −53 |
| run | 29 | 20.7% | 96.2% | −76 |
| exec | 11 | 9.1% | 84.6% | −76 |

**The category ranking inverted.** Under the contaminated eval, `compose` and
`network` looked perfect and `exec` was the identified weak spot. On clean data, the
categories that survive are the ones with the fewest distinct commands and the
simplest surface (`volume`, `system`), while the flag-composition-heavy categories
collapse.

### Failure modes

Of 62 misses:

| Mode | Count |
|---|---|
| right subcommand, wrong flags | 44 |
| wrong subcommand entirely | 18 |
| empty output | 0 |

The model always produces well-formed, plausible Docker. It fails at flag composition,
not at format or syntax.

The `exec` category shows this at its cleanest — **6 of its 10 misses are a single
missing `-it`:**

```
"drop me into a shell on the api container"
  want  docker exec -it api bash
  got   docker exec api bash

"attach a terminal to the cache container"
  want  docker exec -it cache bash
  got   docker exec cache bash
```

Right verb, right container, right shell, one flag dropped — and the resulting command
is useless for the request, because without `-it` there is no interactive terminal.

The diagnosis follows from the training data. Every `exec` shell example in
`data/docker_training.jsonl` is phrased with a fixed trigger vocabulary — *"open shell
in container X"*, *"run bash in container X"*, *"shell into X"*. The held-out prompts use
*"drop me into"*, *"poke around inside"*, *"attach a terminal to"*. **The model learned
which lexical triggers precede `-it`, not that interactive intent requires it.**

That is memorization of surface form, and it is what the 94% was measuring.

### What this implies

The dataset is the bottleneck, not the model size. 594 examples covering 298 unique
commands, mostly single-flag, cannot teach flag composition.

This puts the project's headline 1B-vs-4B "capacity ceiling" claim in doubt. That
ceiling was attributed to the 1B model's parameter count; the evidence now suggests it
was imposed by the dataset, and that the 4B model hit the same wall without anyone
noticing because the contaminated metric hid it. Re-scoring the 1B on this test set is
the cheap experiment that would settle it.

---

## 5. Reproducing

```bash
# Audit a dataset's internal redundancy
python -m eval.contamination --train data/docker_training.jsonl

# Audit a train/test pair
python -m eval.contamination --train data/docker_training.jsonl \
                             --test  data/docker_test_handwritten.jsonl --verbose

# Create a leak-free split of any dataset
python -m eval.splits --input data/docker_training.jsonl \
                      --train-out data/train.jsonl --test-out data/test.jsonl

# Score a GGUF (no Python build toolchain needed — uses a prebuilt llama.cpp binary)
python -m eval.run_eval --model models/docker_gemma3_4b_q4km.gguf \
                        --template gemma3 \
                        --llama-bin vendor/llama.cpp/llama-cli.exe \
                        --label gemma3_4b --strict

# Re-score a completed run under improved metrics, without re-running inference
python -m eval.run_eval --replay results/gemma3_4b_generations.jsonl
```

Every run writes `results/<label>_generations.jsonl` (every prompt, raw output, and
latency) and `results/<label>_summary.json` (scores plus the embedded contamination
report). Published numbers should always be traceable to one of these files.

---

## 6. Known limitations

Stated rather than discovered later by someone else.

1. **The test set is one author's phrasing.** 116 hand-written prompts by the person
   who also knows the training distribution. Genuinely independent prompts — scraped
   from Stack Overflow Docker questions, or collected from other developers — would be
   stronger. This is a real weakness, not a formality.
2. **Functional equivalence is a heuristic, not execution.** Two commands scored
   equivalent may still differ in effect. Execution-based scoring in disposable
   containers is Milestone 4.
3. **No baseline yet.** Until Milestone 2 lands, these numbers say how well the
   fine-tune does, not whether the fine-tuning helped.
4. **Single seed.** No variance estimates. Any 1B-vs-4B claim needs multiple seeds
   before it means anything.
5. **116 examples is small.** A binomial 95% CI at n=116 is roughly ±9 points near
   50% accuracy, tightening to ±4 near 95%. Differences smaller than that are noise.
