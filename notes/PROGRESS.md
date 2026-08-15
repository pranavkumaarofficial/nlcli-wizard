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

- [ ] `eval/splits.py` — split by *target command*, not by row. Guarantee zero
      command overlap between train and test.
- [ ] `data/docker_test_handwritten.jsonl` — 100–150 held-out prompts written in
      natural human phrasing, not generator templates.
- [ ] `eval/contamination.py` — reusable audit: exact-prompt overlap, target-command
      overlap, near-duplicate detection. Runs on any train/test pair.
- [ ] `eval/metrics.py` — exact match + normalized match (flag-order invariant) +
      functional equivalence, reported side by side.
- [ ] `eval/run_eval.py` — replaces `scripts/legacy/evaluate_docker_LEGACY.py`.
- [ ] `docs/EVAL_METHODOLOGY.md` — what was wrong, how it was found, how it is fixed,
      and the corrected numbers next to the old ones.
- [ ] Re-score the existing 1B and 4B GGUFs on the clean set. Publish whatever
      comes out, including a drop.

---

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
