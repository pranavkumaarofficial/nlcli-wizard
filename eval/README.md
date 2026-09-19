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
| `normalize.py` | Structural command parsing: path, flag set, positionals |
| `metrics.py` | Exact / normalized / functional-equivalence scoring |
| `backends.py` | One `generate()` interface over llama-server, llama-cli, llama-cpp-python, transformers, replay |
| `run_eval.py` | Entry point; replaces the legacy script |

## Backends

`--backend llama-server` is the default and the one the published numbers were
produced with. It loads the model once, returns JSON rather than a TUI that changes
between llama.cpp releases, and gives per-token logprobs for free.

`--llama-bin` must point at the binary matching `--backend`. Passing `llama-cli.exe`
while `--backend` is `llama-server` spawns a process that never answers `/health`,
and the run blocks for the full 180 s startup timeout before failing.

`--backend llama-cpp-python` does not report logprobs unless you pass `--logprobs`.
The bindings refuse `logprobs=` unless the model was built with `logits_all=True`,
which keeps the logits for every prompt position: roughly 1 MB per prompt token at
Gemma 3's 262k vocabulary. Use llama-server if you want confidence numbers.

`--replay` re-scores a recorded run with no inference and no model file. It is the
fastest way to check a published number:

```bash
python -m eval.run_eval --replay results/<label>_generations.jsonl
```

## Prior art this follows

- **NL2Bash** (LREC 2018) — 9,305 human-curated NL/command pairs.
- **NLC2CMD** (NeurIPS 2020) — functional-equivalence heuristic over utilities and
  flag sets rather than string comparison.
- **InterCode-Bash** — executes predicted and gold commands in matched containers
  and compares resulting state.
