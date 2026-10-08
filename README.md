# nlcli-wizard

[![tests](https://github.com/pranavkumaarofficial/nlcli-wizard/actions/workflows/tests.yml/badge.svg)](https://github.com/pranavkumaarofficial/nlcli-wizard/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Natural language to CLI commands, offline, on CPU.** A fine-tuned 1B small language
model turns plain English into Docker commands with no API key, no network call, and
no GPU. Built for machines that cannot reach a hosted model: air-gapped hosts,
customer-site VMs, restricted networks.

Stack: QLoRA fine-tuning, GGUF quantization, llama.cpp inference, Python 3.10+.

https://github.com/user-attachments/assets/2d7ca418-d6b2-4449-a81e-417df9666d44

<sub>Recorded February 2026, before the correction below. The `CONFIDENCE` value it
shows is not real; see [Limitations](#limitations).</sub>

> **This project previously published 94% Docker accuracy. That number was wrong.**
> It was measured on training data. The corrected figure is 46.6%. The full account
> is in [`docs/EVAL_METHODOLOGY.md`](docs/EVAL_METHODOLOGY.md): what broke, how it
> was found, and what replaced it. Both numbers are kept side by side below rather
> than the old one being quietly deleted.

## Results

Every row below is the same 116 hand-written held-out prompts, scored by the same
code. Zero prompt overlap with training.

| config | accuracy |
|---|---|
| base 1B, zero-shot | 9.5% |
| base 1B, with a system prompt | 15.5% |
| base 1B, 8-shot | 19.0% |
| fine-tuned, v1 dataset | 46.6% |
| **fine-tuned, v2 dataset** | **62.9%** |

Paired McNemar test on v1 vs v2: p = 0.006.

**Read the 62.9% carefully.** The v2 training set covers 3,188 unique commands against
v1's 298, so part of the gain is coverage rather than generalization. On the 33 test
commands that appear in neither training set, v1 scores 36.4% and v2 scores 45.5%, and
that difference is not significant (p = 0.58).

The result that does hold up is narrower and more interesting:

| flags in the target command | n | v1 | v2 |
|---|---|---|---|
| 2 flags, unseen command | 7 | 0/7 | **4/7** |
| 3 or more flags, unseen command | 11 | 0/11 | 1/11 |

The v2 dataset was built specifically to teach flag composition. It did, at two flags,
on commands the model had never seen. At three or more flags the problem is unsolved.

### By category

| category | n | v1 | v2 |
|---|---|---|---|
| system | 16 | 68.8% | 81.2% |
| compose | 15 | 46.7% | 80.0% |
| network | 9 | 55.6% | 77.8% |
| ps/images | 20 | 60.0% | 60.0% |
| volume | 7 | 100.0% | 57.1% |
| build | 9 | 55.6% | 55.6% |
| exec | 11 | 9.1% | 54.5% |
| run | 29 | 20.7% | 48.3% |

`volume` regressed because the v2 dataset deliberately starved a category that was
already at 100%. All three new failures pick the wrong subcommand, not the wrong flags.
Rebalancing away from a saturated category was not free.

### Reproduce any number here

```bash
python -m eval.run_eval --replay results/v2_finetune_generations.jsonl
```

Under a second, no model download, no GPU. Every run in `results/` is replayable this
way: the generations are committed, not just the summaries.

## Status

**The weights are not downloadable yet.** `MODEL_REGISTRY` in
`nlcli_wizard/model.py` points at `pranavkumaarofficial/nlcli-gemma3-docker` on
HuggingFace, and that repo is private as of 2026-10-08. Until it is published, the
only working paths are a GGUF you already have (`--model-path`) or training your own.
`translate` prints the repo id, the filename and every path it searched when the
download fails.

The evaluation harness needs no weights and no GPU. If you are here to check whether
the numbers hold, start there.

## Quickstart

CI runs on Python 3.10 and 3.12.

```bash
git clone https://github.com/pranavkumaarofficial/nlcli-wizard.git
cd nlcli-wizard
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install llama-cpp-python from the prebuilt CPU wheel index FIRST. Without this, pip
# resolves it to an sdist and compiles llama.cpp, which needs a C++ toolchain and
# fails on Windows on a long path.
pip install llama-cpp-python \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu

pip install -e .
```

Fourteen packages, no compiler. `torch` and `transformers` are not runtime
dependencies; the inference path is GGUF only.

With a GGUF in `models/`:

```bash
python -m nlcli_wizard.cli translate --cli-tool docker \
    "run nginx on port 8080 in background"
```

```
Loading model from models\docker_gemma3_4b_q4km.gguf...
Input: run nginx on port 8080 in background
Command: docker run -d -p 8080:80 nginx
Confidence: 95%
Runs nginx container in detached mode, mapping port 8080:80, container ID: nginx
```

That is the real output. The confidence is meaningless and the explanation degrades
after the first clause. Neither is scored; the evaluation reads only the `COMMAND:`
line.

Measured on an 11th-gen Core i5-1135G7, 4 threads, CPU only: 1.09 s to load the model,
1.85 s to generate, about 2.9 s for a cold invocation.

### Run the evaluation without weights

```bash
pip install -e ".[dev]"
python -m pytest -q tests                                           # 120 tests
python -m eval.contamination --train data/docker_training.jsonl \
                             --test  data/docker_test_handwritten.jsonl
python -m eval.run_eval --replay results/v2_finetune_generations.jsonl
```

`eval/` has zero third-party dependencies. It is pure standard library.

### Train your own

[Colab notebook](https://colab.research.google.com/drive/1uBJJ_EqCMT8bMnCnVQHeN8USKu1ABddL)
(free T4). The notebook runs baselines before training, aborts if the training file
leaks the held-out set, and emits the ablation table above.

```bash
python -m nlcli_wizard.dataset_v2 --out data/my_train.jsonl \
                                  --exclude data/docker_test_handwritten.jsonl
```

## How the evaluation works

This is the part worth reading. The original 94% survived nine months because the
evaluation logic lived in a private copy inside the training notebook, where nothing
ever compared the two files.

**Contamination is a gate, not a report.** The leakage audit runs before inference and
prints above the accuracy. `--strict` exits non-zero rather than emit a number on a
leaking pair. Four channels are checked: verbatim prompt overlap and near-duplicate
prompts are fatal, shared generator templates are high severity, and target-command
overlap is disclosed rather than failed, because generalizing to new phrasings of a
known command is the actual task.

**Three metrics, always together.** Exact match, flag-order-normalized match, and
functional equivalence (`docker ps` equals `docker container ls`). What is deliberately
not normalized away is `-d`, `-i`, `-t` and `-a`. Those change what the command does. A
scorer that forgave them would have rated the `exec` category at 90% while the model
was producing non-interactive shells.

**Results split by novelty.** Unseen command versus unseen phrasing of a known command
are reported separately, because a single blended number lets the easier partition
carry the harder one.

**The harness has its own tests.** Building it surfaced two real parser bugs: `-it` did
not expand into `-i -t`, and `-t` was globally value-consuming when it only takes a
value under `build`, so `docker run -t nginx` parsed as a tag with no image. Both would
have mis-scored every interactive example.

Two further bugs were found later, and both had biased results toward the fine-tune.
A markdown fence labelled ```` ```docker ```` was parsed as the command itself, scoring
65 of 116 baseline generations as zero and reading the zero-shot baseline as 3.4%
instead of 9.5%. And recorded runs could not be re-scored at all, because the replay
path keyed on a prompt string that differs between the notebook and the local harness.
Both are fixed, with regression tests.

## Design notes

**Why fine-tune a small local model instead of prompting a hosted one.** For most
people, asking a frontier model is the right answer. It stops being the right answer on
a host with no egress, where the security review of a new outbound endpoint costs more
than the feature. The commands you most want help composing are often the ones you run
in exactly those places. That constraint rules out GPUs and network calls, which rules
out large models, which is the entire problem this repo is about.

**Q4_K_M on a narrow model is not what the name suggests.** K-quants operate on
256-element superblocks, and this model's embedding width is 1152, which is not a
multiple of 256. Every tensor with a 1152-long row falls back to a non-K quant. Of 183
two-dimensional tensors, 117 are Q5_0 and 14 are Q8_0, and only 52 are actually Q4_K or
Q6_K. The file is roughly 60% Q5_0 by parameter count. The correlation with row length
holds for 183 of 183. Check it yourself with standard library only:

```bash
python scripts/gguf_header.py models/docker_gemma3_4b_q4km.gguf
```

**What I would do differently.** Write the held-out set first, before the generator,
and not in the same week by the same person. The first build of the v2 dataset leaked
12 test prompts verbatim because the author reached for the same phrasings twice. Care
does not prevent that; a generation-time blocklist does, and it is now the default.

## Limitations

- **Not usable for the two most common Docker verbs.** `run` is 48.3% and `exec` is
  54.5%. Better than v1, not good.
- **Three or more flags is unsolved.** 1 of 11 on unseen commands.
- **Ignore the `CONFIDENCE` field.** The dataset generator filled it with
  `random.uniform(0.90, 0.97)`, so the model was trained to predict a random number.
  Worse, `nlcli_wizard/agent.py` gates its success flag on that number, so a correct command with a
  low sampled confidence is reported as a failed translation.
- **Single seed.** No variance estimates. At n=116 the 95% interval is roughly plus or
  minus 9 points near 50%.
- **The test set is one author's phrasing.** Prompts collected from real users would be
  stronger evidence.
- **Functional equivalence is a heuristic, not execution.** Two commands scored
  equivalent may still differ in effect.
- **The CLI lowercases your input**, which destroys build-arg values and env var names.
- **The model is a 1B, not the 4B the filename claims.** Its header reports
  `size_label 1000M` and 999,885,952 parameters. The 46.6% and 62.9% both belong to a
  1B. Recorded in `docs/EVAL_METHODOLOGY.md`.
- **Docker only.** The venvy integration's accuracy was withdrawn for the same
  contamination reason.

## Repo layout

```
nlcli_wizard/     CLI, GGUF loading, prompt formatting, dataset generators
eval/             contamination audit, command-level splits, normalization,
                  scoring, inference backends, entry point. No third-party deps.
tests/            120 tests
data/             v1 (594 rows), v2 (5,000 rows), 116 hand-written held-out prompts
results/          every run, summaries and per-generation records, all replayable
scripts/          gguf_header.py, shell wrappers
training/         Colab notebook, generated by build_notebook.py
docs/             EVAL_METHODOLOGY.md
```

## Contributing

[CONTRIBUTING.md](CONTRIBUTING.md). Issues and PRs welcome at
[github.com/pranavkumaarofficial/nlcli-wizard/issues](https://github.com/pranavkumaarofficial/nlcli-wizard/issues).

## License

[MIT](LICENSE)
