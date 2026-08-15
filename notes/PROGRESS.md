# PROGRESS — nlcli-wizard rebuild

Working log of deliverables for the V3 direction. Every completed item gets a
timestamp. Newest work at the top of each section.

**Convention**
- `[ ]` not started · `[~]` in progress · `[x]` done (timestamp required) · `[!]` blocked
- Timestamps are local (IST), format `YYYY-MM-DD HH:MM`.
- A deliverable is only `[x]` when it is *verified* — code runs, numbers reproduce,
  or the document is complete. Not when it is merely written.

---

## Direction (V3)

The project is being repositioned. Old framing: "another natural-language shell."
That category is crowded (aichat, shell-gpt, Warp, Lacy Shell) and the local-model
angle alone is not a differentiator.

**New framing:** an automated pipeline for distilling a CLI tool's own documentation
into a shippable, offline NL adapter — targeted at environments where cloud tools
structurally cannot go (air-gapped hosts, restricted VMs, internal/niche CLIs that
no general model has ever seen).

Two hard requirements for everything built from here:
1. **No metric is published without a contamination-free test set.**
2. **No fine-tune result is published without a baseline it is compared against.**

---

## Audit findings that triggered the rebuild — 2026-08-15

Recorded so the reasoning is not lost. Details in `docs/EVAL_METHODOLOGY.md` (pending).

| # | Finding | Severity |
|---|---------|----------|
| A1 | All published accuracy numbers measured on training data. `evaluate_docker.py` scored the last 100 lines of the same JSONL the notebook trained on (random 90/10 split) — ~90 of 100 eval rows were in train. | Critical |
| A2 | venvy set: 1500 rows → 230 unique instructions (6.5× duplication). 95/100 eval prompts appear verbatim in train. The "83%" is memorization recall. | Critical |
| A3 | docker set: 594 rows → only 299 unique commands. 69/100 eval rows share their exact target command with train. Even a clean random split leaks, because the generator emits 2–3 paraphrases per command and the split cuts between them. | Critical |
| A4 | No baseline anywhere. Base model zero-shot / few-shot never measured, so the value of fine-tuning is unknown. | Critical |
| A5 | `CONFIDENCE` field is `random.uniform(0.90, 0.97)` — a fabricated number, surfaced to users as a percentage and used to gate `success` in `agent.py`. | High |
| A6 | Exact string match scoring. Flag-order-equivalent commands scored wrong. Field standard is functional equivalence (NL2CMD) or execution-based (InterCode-Bash). | High |
| A7 | `MODEL_REGISTRY` filename/repo do not match the shipped model or the README. The CLI cannot load the model in `models/`. | High |
| A8 | `model.py` hardcodes Gemma 3 prompt format; notebook trains Gemma 4 via `apply_chat_template`. Inference format ≠ training format. | High |
| A9 | `_validate_command` rejects `$`, `>`, `<` — rejects the model's own correct output (`-v $(pwd):/app`). Blocklist is the wrong primitive. | Medium |
| A10 | `_normalize_input` lowercases everything, destroying case-sensitive env values, tags, container names. | Medium |
| A11 | No CI, no real tests. `pyproject` points `testpaths` at a directory that did not exist. | Medium |
| A12 | Repo hygiene: 1.2 GB `.git`, 72 tracked build-cache files, stray `llama.cpp` gitlink, 80 MB of tokenizer dirs. | Medium |

---

## Milestone 0 — Repo reset

- [x] Untrack build junk: `unsloth_compiled_cache/` (72 files), `venvy_gemma3_lora/`,
      `venvy_gemma3_merged/`, `venvy_imatrix.dat`, `imatrix_data.txt`,
      `models/venvy_gemma3_q4km.gguf`, `.claude/settings.local.json`.
      Tracked files 130 → 38. — **2026-08-15 14:09**
- [x] Remove stray `llama.cpp` gitlink (submodule entry with no `.gitmodules`,
      broke fresh clones). — **2026-08-15 14:09**
- [x] Rewrite `.gitignore` around a stated principle; add `notes/*` +
      `!notes/PROGRESS.md` negation. — **2026-08-15 14:09**
- [x] Restructure: new `eval/` and `tests/`; legacy scripts → `scripts/legacy/`
      with `_LEGACY` suffix; superseded guides → `docs/archive/`. — **2026-08-15 14:09**
- [x] Fix `pyproject.toml`: real author email, `testpaths` → `tests`, coverage
      moved off the default addopts. — **2026-08-15 14:09**
- [ ] Purge large blobs from git history (`.git` is 1.2 GB). Deferred — needs a
      force-push and a decision on rewriting public history.
- [ ] Move GGUF weights to HuggingFace and drop `models/*` from LFS.

---

## Milestone 1 — Contamination-free eval  ← ACTIVE

The piece everything else depends on. Nothing else ships until this does.

- [x] `eval/normalize.py` — structural command parsing (path / flag-set / positionals)
      so flag order stops counting as an error. — **2026-08-15 14:45**
- [x] `eval/contamination.py` — audit across 4 leak channels (PROMPT_VERBATIM,
      PROMPT_NEAR_DUP, TEMPLATE_SHARED, TARGET_OVERLAP). Independently reproduces
      A1–A3. Surfaced a finding the manual audit missed: `venvy current` is 225 of
      1500 venvy rows (15% of the dataset is one command). — **2026-08-15 14:40**
- [x] `eval/splits.py` — split by *target command*, not by row; stratified by
      category; asserts zero command overlap. — **2026-08-15 14:47**
- [x] `eval/metrics.py` — exact + normalized + functional, always reported together.
      `-d`/`-a`/`-i`/`-t` deliberately NOT treated as ignorable. — **2026-08-15 14:46**
- [x] `eval/backends.py` — llama-server / llama-cli / llama-cpp-python / transformers
      / replay behind one interface, so baselines and fine-tunes share a code path.
      — **2026-08-15 15:05**
- [x] `data/docker_test_handwritten.jsonl` — 116 hand-written held-out prompts across
      9 phrasing styles (telegraphic, conversational, question, typo, jargon, minimal,
      polite, compositional, unseen-flag). Audits CLEAN: 0 verbatim, 0 near-dup,
      0 shared-template. 57% target overlap, disclosed. — **2026-08-15 14:52**
- [x] `eval/run_eval.py` — replaces the legacy script. Contamination gate runs and
      prints *above* the accuracy; `--strict` refuses to report a number on a
      contaminated pair. — **2026-08-15 15:10**
- [x] `tests/test_eval_harness.py` — 36 tests, all passing. Caught two genuine parser
      bugs: `-it` failed to expand, and `docker run -t nginx` parsed as `-t=nginx`
      with no image, both from `-t` being globally value-consuming when it is only
      that under `build`. Would have silently mis-scored every exec/interactive-run
      example. — **2026-08-15 15:02**
- [x] `docs/EVAL_METHODOLOGY.md` — what was wrong, how it was found, how it is fixed.
      §4 (corrected numbers) pending the run. — **2026-08-15 15:00**
- [x] Re-score Gemma 3 4B docker on the clean set. **94.0% -> 46.6%.**
      unseen_command 38.0% (n=50), unseen_phrasing 53.0% (n=66). exact == normalized
      == functional == 46.6%, so the drop is contamination, not scoring.
      `results/gemma3_4b_docker_summary.json`. — **2026-08-15 15:32**
- [x] Publish corrected numbers in README + `docs/EVAL_METHODOLOGY.md` §4, with the
      legacy figures kept alongside rather than deleted. Withdrew the venvy 83% and
      the 1B-vs-4B comparison. — **2026-08-15 15:40**
- [!] Gemma 3 1B docker — **weights not available locally.** `models/` holds only
      `docker_gemma3_4b_q4km.gguf` and `venvy_gemma3_q4km.gguf`. Needed to settle
      whether the "capacity ceiling" was capacity or dataset. Ask: HuggingFace? Drive?
      Old Colab session?

### Findings from the corrected run — 2026-08-15

Category ranking **inverted** versus the contaminated eval. Survivors are the
categories with fewest distinct commands (volume 100%, system 68.8%); collapses are
the flag-composition-heavy ones (run 20.7%, exec 9.1%).

Failure modes over 62 misses: 44 right-subcommand/wrong-flags, 18 wrong subcommand,
0 malformed. The model always emits plausible Docker; it cannot compose flags.

Sharpest single result: **6 of the 10 `exec` misses are a dropped `-it`.** Training
phrases exec-shell requests with a fixed trigger vocabulary ("open shell in container
X", "run bash in X"); the held-out prompts say "drop me into", "poke around inside",
"attach a terminal to". The model learned which lexical triggers precede `-it`, not
that interactive intent requires it.

**Implication that changes the plan:** the dataset is the bottleneck, not model size.
594 examples over 298 mostly single-flag commands cannot teach flag composition. The
1B-vs-4B "capacity ceiling" was probably dataset-imposed, and the 4B hit the same wall
unnoticed behind the contaminated metric. Swapping to a newer base model without
fixing the dataset would repeat the original mistake with fresher weights.

### Infrastructure notes from this session

- `llama-cpp-python` has no Windows wheel for Python 3.9; building needs MSVC.
  Worked around with the official prebuilt llama.cpp release (b10437, CPU x64,
  18 MB) in `vendor/` (gitignored).
- `llama-cli` in b10437 **ignores `-no-cnv`** and drops into an interactive chat TUI,
  which silently breaks subprocess scraping. Hence `llama-server` + HTTP as the
  default backend: model loads once, returns JSON, and supplies per-token logprobs
  (which Milestone 3 needs anyway).
- Local host: 8 GB RAM, CPU-only. 4B Q4_K_M runs at ~4.2 tok/s, ~5.8s per example.
  116 examples ≈ 11 minutes. Workable but not for repeated sweeps — baselines
  (M2) should run on Colab GPU.

---

## Milestone 1.5 — Dataset v2 (composition coverage)  ← ACTIVE

Root cause of the 46.6%, quantified from the corrected run:

| flags in target | accuracy | share of v1 training data |
|---|---|---|
| 0 | 74.0% | 35.4% |
| 1 | 47.1% | 52.0% |
| 2 | **5.0%** | 12.0% |
| 3+ | **0.0%** | **0.7%** |

Accuracy collapses exactly where coverage ends. v1 has 24 distinct flags but only
**17 distinct flag pairs**, 47 occurrences of which are the trivial `-i`+`-t` bundle.
`--detach`+`--publish` appears 3 times. The model was asked to compose flags it had
never seen composed.

Sanity check: reweighting the per-flag-count accuracies to v1's own (easier)
distribution still gives ~51%, not 94%. The drop is real, not test difficulty.

- [x] `nlcli_wizard/dataset_v2.py` — composition-first generator. Flags declared as
      specs with many intent-based phrasings each; examples built by sampling flag
      *subsets*; sentence order and connectives randomised. — **2026-08-15 16:20**
- [x] `data/docker_train_v2.jsonl` — 5,000 rows. — **2026-08-15 16:35**

      | metric | v1 | v2 |
      |---|---|---|
      | rows | 594 | 5,000 |
      | unique prompts | 592 | 5,000 |
      | unique commands | 298 | 3,188 |
      | multi-flag share | 12.6% | **52.4%** |
      | distinct flag pairs | 17 | **72** |
      | 3+ flag examples | 4 | ~700 |

- [x] Generation-time exclusion of the held-out set. The first v2 build leaked
      **12 test prompts verbatim** — same author wrote the test set and the
      generator phrasings and reached for the same words twice. Care does not
      prevent this; a blocklist does. Now `--exclude` defaults to the test set and
      dedups on the same fingerprint the audit uses. — **2026-08-15 16:30**
- [x] Four content defects found by sampling and fixed, each with a regression test:
      doubled conjunctions ("and and"), ports published on non-serving images
      (`-p 3000:8080 busybox`), env vars from the wrong image
      (`MONGO_INITDB_ROOT_PASSWORD` on caddy), and build-arg name/value mismatches
      (`VERSION=production`). — **2026-08-15 16:40**
- [x] `tests/test_dataset_v2.py` — 14 tests. Suite now 50 passing. — **2026-08-15 16:45**
- [x] v2 drops the fabricated `CONFIDENCE` field entirely (was
      `random.uniform(0.90, 0.97)`). Asserted in tests. — **2026-08-15 16:45**
- [x] `training/build_notebook.py` -> `training/nlcli_wizard_train_v2.ipynb`
      (36 cells). Generated, not hand-edited, so cells stay diffable and the JSON
      stays valid. — **2026-08-15 17:10**
      - imports `eval/` from the clone instead of reimplementing scoring in a cell
        (the old notebook's private copy of the eval logic is how the contaminated
        metric survived nine months)
      - contamination gate **raises** before training if the train file leaks
      - baselines (zero-shot / +system / 8-shot) run BEFORE training, same scorer
      - command-level validation split via `eval/splits.py`
      - `train_on_responses_only`, with turn markers detected from the tokenizer
        rather than hardcoded, plus a cell that decodes the mask and asserts the
        instruction is excluded before spending 25 min on it
      - base model held at Gemma 3 4B so the v1->v2 ablation moves one variable
      - emits the ablation table and a per-flag-count delta vs the v1 run
- [x] Static verification of the notebook (all that is possible without a GPU):
      valid JSON, every code cell parses, all 10 repo imports resolve, the
      command-level split on v2 gives 0 command overlap (4766/234), and the
      per-flag-count reporting cell reproduces the v1 numbers exactly when run
      against the v1 generations. — **2026-08-15 17:20**
- [ ] **RUN IT.** Requires GPU + Google auth; cannot be executed from this machine
      (Python 3.9 only, no general-purpose WSL distro, and colab-cli needs
      interactive OAuth). Blocked on Pranav.
- [ ] Ablation table filled in: v1 vs v2 dataset, model and recipe held fixed.

Known remaining weakness: `volume` saturates at 71 unique examples and `run` is 44%
of the mix. `run` is deliberate — it is the hardest, most compositional category and
scored 20.7%. Volume already scores 100% and needs no more data.

## Milestone 2 — Baselines

- [ ] Base Gemma 4 E2B, zero-shot
- [ ] Base + system prompt describing flag conventions
- [ ] Base + 8-shot in-context examples
- [ ] Fine-tuned, clean test set
- [ ] One table, 3 seeds per config, variance reported

---

## Milestone 3 — Real confidence

- [ ] Replace `random.uniform` confidence with mean token logprob from llama.cpp
- [ ] Reliability diagram: predicted confidence vs. observed correctness
- [ ] Report calibration error honestly, whatever it is

---

## Milestone 4 — Execution-based scoring

- [ ] Run predicted vs. gold commands in disposable containers, diff resulting state
- [ ] Position against NL2Bash / NLC2CMD / InterCode-Bash in the writeup

---

## Milestone 5 — Make the CLI actually run

- [ ] Fix `MODEL_REGISTRY` ↔ shipped filename ↔ README mismatch (A7)
- [ ] Read chat template from GGUF metadata instead of hardcoding (A8)
- [ ] Replace the `$`/`>`/`<` blocklist with a subcommand+flag allowlist (A9)
- [ ] Stop lowercasing user input (A10)
- [ ] Fix `CommandHistory` timestamp (currently the ctime of `agent.py`)
- [ ] CI: dataset validation + eval-harness unit tests + CLI smoke test

---

## Milestone 6 — The actual thesis

- [ ] `--help` → parsed flag grammar → LLM-generated paraphrases → **validated
      against the grammar** → training set
- [ ] Prove it on a CLI never touched before, end to end
- [ ] Only after Milestones 1–2. Worthless on top of an eval no one can trust.

---

## Deferred / explicitly not doing

- Kubernetes and Git datasets — not until the pipeline in M6 generates them.
- PyPI release — not until M5 makes the CLI actually work.
- Any new training run — not until M1 gives a trustworthy number to compare against.
