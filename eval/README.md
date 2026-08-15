# eval/

The evaluation harness. This directory is the trust boundary of the project — if
the code in here is wrong, every number the project publishes is wrong.

## Why this exists

The previous harness (`scripts/legacy/evaluate_docker_LEGACY.py`) scored the model
on the last 100 lines of the same JSONL file the notebook trained on. Roughly 90 of
those 100 rows were in the training split. Every accuracy figure produced by it —
Docker 94%, venvy 83% — measured memorization, not translation.

Nothing in this directory may read from a training file without going through a
declared, audited split.

## Rules

1. **Splits are by target command, never by row.** The dataset generator emits
   several paraphrases per command; splitting by row puts paraphrases of the same
   command on both sides.
2. **Every eval run emits a contamination report** alongside its accuracy number.
   An accuracy without a contamination report is not a result.
3. **Metrics are reported as a set**, never as a single number: exact match,
   normalized match (flag-order invariant), and functional equivalence.
4. **Baselines run through the same harness as fine-tunes.** No special-casing.

## Layout (in progress — see `notes/PROGRESS.md`, Milestone 1)

| File | Purpose |
|------|---------|
| `splits.py` | Command-level train/test partitioning |
| `contamination.py` | Overlap and near-duplicate auditing for any train/test pair |
| `metrics.py` | Exact / normalized / functional-equivalence scoring |
| `run_eval.py` | Entry point; replaces the legacy script |

## Prior art this follows

- **NL2Bash** (LREC 2018) — 9,305 human-curated NL/command pairs.
- **NLC2CMD** (NeurIPS 2020) — functional-equivalence heuristic over utilities and
  flag sets rather than string comparison.
- **InterCode-Bash** — executes predicted and gold commands in matched containers
  and compares resulting state.
